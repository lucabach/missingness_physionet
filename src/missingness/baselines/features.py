"""Causal hand-made features for the classical baselines.

One row per labelled step (every hour for sepsis, the last hour for mortality), built only
from data up to that step. Feature sets (mirroring the deep-model input variants):

    pattern  per variable: measurements so far, hours since the last one, measured this hour
    values   per variable: last observed value (training mean before the first one) + statics
    time     hours since ICU admission (sepsis only; mortality always predicts at 48 h)
    all      pattern + values

Values are imputed with the training mean so that "never measured" is not visible to the
values-only models; otherwise a tree could read the pattern from the missing entries.
"""

from __future__ import annotations

import numpy as np

from missingness.data.preprocessing import forward_fill
from missingness.data.records import PatientRecord

FEATURE_SETS = ("pattern", "values", "time", "all")


def _pattern(r: PatientRecord) -> np.ndarray:
    obs = ~np.isnan(r.values)
    count = np.cumsum(obs, axis=0)
    elapsed = r.hours - r.hours[0]
    since = np.zeros(obs.shape, dtype=np.float32)
    for t in range(len(r.hours)):
        if t == 0:
            since[t] = np.where(obs[t], 0.0, 1.0)
        else:
            gap = r.hours[t] - r.hours[t - 1]
            since[t] = np.where(obs[t], 0.0, since[t - 1] + gap)
    # never measured so far: hours since admission + 1 (larger than any real gap so far)
    since = np.where(count == 0, elapsed[:, None] + 1.0, since)
    return np.concatenate([count, since, obs], axis=1).astype(np.float32)


def _values(r: PatientRecord, fill: np.ndarray, static_fill: np.ndarray) -> np.ndarray:
    x = forward_fill(r.values, fill)
    s = np.where(np.isnan(r.static), static_fill, r.static).astype(np.float32)
    return np.concatenate([x, np.broadcast_to(s, (len(r.hours), len(s)))], axis=1)


def feature_names(variables: list[str], static: list[str], kind: str) -> list[str]:
    pattern = (
        [f"count_{v}" for v in variables]
        + [f"since_{v}" for v in variables]
        + [f"measured_{v}" for v in variables]
    )
    values = [f"last_{v}" for v in variables] + [f"static_{s}" for s in static]
    return {"pattern": pattern, "values": values, "time": ["hour"], "all": pattern + values}[kind]


def build_features(
    records: list[PatientRecord],
    kind: str,
    task: str,
    fill: np.ndarray,
    static_fill: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return X, y, patient_id, hour for every labelled step of ``records``.

    ``fill`` / ``static_fill``: training means of each variable / static covariate
    (``Normalizer.fit(train).mean`` and ``.static_mean``).
    """
    if kind not in FEATURE_SETS:
        raise ValueError(f"kind must be one of {FEATURE_SETS}")
    X, y, pid, hour = [], [], [], []
    for r in records:
        parts = []
        if kind in ("pattern", "all"):
            parts.append(_pattern(r))
        if kind in ("values", "all"):
            parts.append(_values(r, fill, static_fill))
        if kind == "time":
            parts.append(r.hours[:, None].astype(np.float32))
        feats = np.concatenate(parts, axis=1)
        if task == "hourly":
            X.append(feats)
            y.append(r.labels)
            hour.append(r.hours)
            pid += [r.patient_id] * len(r.hours)
        else:
            X.append(feats[-1:])
            y.append(r.labels[-1:])
            hour.append(r.hours[-1:])
            pid.append(r.patient_id)
    return {
        "X": np.concatenate(X).astype(np.float32),
        "y": np.concatenate(y).astype(int),
        "patient_id": np.asarray(pid),
        "hour": np.concatenate(hour),
    }
