"""AUROC / AUPRC with 95% CIs from bootstrapping *patients* (not hours)."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def auroc(y: np.ndarray, score: np.ndarray) -> float:
    return float(roc_auc_score(y, score))


def auprc(y: np.ndarray, score: np.ndarray) -> float:
    return float(average_precision_score(y, score))


METRICS: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {"auroc": auroc, "auprc": auprc}


def bootstrap_ci(
    y: np.ndarray,
    score: np.ndarray,
    patient_ids: np.ndarray,
    metric: str = "auroc",
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Point estimate and percentile CI, resampling patients with replacement.

    For sepsis there are many rows (hours) per patient; resampling rows would understate the
    uncertainty, so all rows of a sampled patient enter together.
    """
    fn = METRICS[metric]
    y, score, patient_ids = map(np.asarray, (y, score, patient_ids))
    _, inverse = np.unique(patient_ids, return_inverse=True)
    rows_of = [np.flatnonzero(inverse == k) for k in range(inverse.max() + 1)]
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(rows_of), size=len(rows_of))
        idx = np.concatenate([rows_of[k] for k in pick])
        if y[idx].min() == y[idx].max():  # only one class drawn, metric undefined
            continue
        stats.append(fn(y[idx], score[idx]))
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return {"value": fn(y, score), "lo": float(lo), "hi": float(hi), "n_boot": len(stats)}


def bootstrap_contrast(
    y: np.ndarray,
    scores: list[np.ndarray],
    weights: list[float],
    patient_ids: np.ndarray,
    metric: str = "auroc",
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Paired bootstrap of sum_k weights[k] * metric(y, scores[k]) (plan P3).

    All scores belong to the same rows (same patients, hours and labels), so every resample
    uses the *same* patients for every model. Differences between models are then much less
    noisy than two separate intervals suggest. Examples: Δmask = [all, values] with weights
    [1, -1]; M2 = [all_random, all_state, values_random, values_state] with [1, -1, -1, 1].
    """
    fn = METRICS[metric]
    y, patient_ids = np.asarray(y), np.asarray(patient_ids)
    scores = [np.asarray(s) for s in scores]
    assert len(scores) == len(weights) and all(len(s) == len(y) for s in scores)

    def value(idx: np.ndarray) -> float:
        return float(sum(w * fn(y[idx], s[idx]) for w, s in zip(weights, scores, strict=True)))

    _, inverse = np.unique(patient_ids, return_inverse=True)
    rows_of = [np.flatnonzero(inverse == k) for k in range(inverse.max() + 1)]
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(rows_of), size=len(rows_of))
        idx = np.concatenate([rows_of[k] for k in pick])
        if y[idx].min() == y[idx].max():
            continue
        stats.append(value(idx))
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    full = np.arange(len(y))
    return {"value": value(full), "lo": float(lo), "hi": float(hi), "n_boot": len(stats)}


def bootstrap_diff(
    y: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
    patient_ids: np.ndarray,
    metric: str = "auroc",
    n_boot: int = 1000,
    seed: int = 0,
) -> dict[str, float]:
    """metric(a) - metric(b) with a paired patient bootstrap (same resampled patients)."""
    return bootstrap_contrast(
        y, [score_a, score_b], [1.0, -1.0], patient_ids, metric, n_boot=n_boot, seed=seed
    )


def pre_onset_mask(
    patient_ids: np.ndarray, hours: np.ndarray, labels: np.ndarray, label_lead: float = 6.0
) -> np.ndarray:
    """Rows strictly before clinical sepsis onset (all rows for non-septic patients).

    PhysioNet 2019 labels turn positive 6 h before onset, so onset = first positive hour + 6.
    Hours after onset carry many extra tests and would inflate the pattern signal.
    """
    patient_ids, hours, labels = map(np.asarray, (patient_ids, hours, labels))
    keep = np.ones(len(labels), dtype=bool)
    for pid in np.unique(patient_ids[labels > 0]):
        rows = patient_ids == pid
        onset = hours[rows & (labels > 0)].min() + label_lead
        keep[rows & (hours >= onset)] = False
    return keep
