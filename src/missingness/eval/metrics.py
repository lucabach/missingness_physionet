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


def _weighted(metric: str, y: np.ndarray, group: np.ndarray, n_groups: int, w: np.ndarray) -> float:
    """AUROC or AUPRC with integer row weights, from rows grouped by distinct score.

    ``group[i]`` is the rank of row i's score among the distinct scores (ascending). A weight
    k counts the row k times, so a bootstrap resample of patients is a weight vector and the
    rows never have to be copied or sorted again. Ties count one half (AUROC), and AUPRC uses
    one threshold per distinct score, as in scikit-learn.
    """
    pos = np.bincount(group, weights=w * y, minlength=n_groups)
    neg = np.bincount(group, weights=w * (1.0 - y), minlength=n_groups)
    n_pos, n_neg = pos.sum(), neg.sum()
    if metric == "auroc":
        below = np.cumsum(neg) - neg  # negatives with a lower score
        return float((pos * (below + 0.5 * neg)).sum() / (n_pos * n_neg))
    pos, neg = pos[::-1], neg[::-1]  # thresholds from the highest score down
    tp, fp = np.cumsum(pos), np.cumsum(neg)
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=tp + fp > 0)
    return float((pos / n_pos * precision).sum())


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
    return bootstrap_contrast(y, [score], [1.0], patient_ids, metric, n_boot, seed, alpha)


def bootstrap_contrast(
    y: np.ndarray,
    scores: list[np.ndarray],
    weights: list[float],
    patient_ids: np.ndarray,
    metric: str = "auroc",
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
    keep_draws: bool = False,
) -> dict:
    """Paired bootstrap of sum_k weights[k] * metric(y, scores[k]) (plan P3).

    All scores belong to the same rows (same patients, hours and labels), so every resample
    uses the *same* patients for every model. Differences between models are then much less
    noisy than two separate intervals suggest. Examples: Δmask = [all, values] with weights
    [1, -1]; M2 = [all_random, all_state, values_random, values_state] with [1, -1, -1, 1].

    A resample is a weight per patient (how often it was drawn), applied to all its rows; this
    gives the same numbers as copying the rows, without re-sorting them for every draw.
    ``keep_draws`` adds the bootstrap values under ``draws`` (to combine independent samples,
    e.g. two hospitals).
    """
    fn = METRICS[metric]
    y, patient_ids = np.asarray(y, dtype=float), np.asarray(patient_ids)
    scores = [np.asarray(s) for s in scores]
    assert len(scores) == len(weights) and all(len(s) == len(y) for s in scores)
    _, patient = np.unique(patient_ids, return_inverse=True)
    n_patients = int(patient.max()) + 1
    groups = []
    for s in scores:
        distinct, group = np.unique(s, return_inverse=True)
        groups.append((group.ravel(), len(distinct)))
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        pick = rng.integers(0, n_patients, size=n_patients)
        w = np.bincount(pick, minlength=n_patients)[patient].astype(float)
        if (w * y).sum() == 0 or (w * (1.0 - y)).sum() == 0:  # one class only: undefined
            continue
        stats.append(
            sum(
                k * _weighted(metric, y, g, n, w) for k, (g, n) in zip(weights, groups, strict=True)
            )
        )
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    value = float(sum(k * fn(y, s) for k, s in zip(weights, scores, strict=True)))
    out = {"value": value, "lo": float(lo), "hi": float(hi), "n_boot": len(stats)}
    if keep_draws:
        out["draws"] = np.asarray(stats)
    return out


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
