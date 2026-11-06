"""Turn raw values (NaN = missing) into model inputs: x (imputed), m (mask), delta (time gap).

Definitions follow GRU-D (Che et al. 2018):
    m[t, d]     = 1 if variable d was measured at step t
    delta[t, d] = hours since the last measurement of d before step t
                  (0 at the first step; grows while d is missing)
    x[t, d]     = last observed value of d (forward fill); the training mean before the
                  first measurement. In normalised units the training mean is 0.

Always recompute all three from ``values`` after any perturbation, never edit them directly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from missingness.data.records import PatientRecord


def compute_mask(values: np.ndarray) -> np.ndarray:
    return (~np.isnan(values)).astype(np.float32)


def compute_delta(mask: np.ndarray, hours: np.ndarray) -> np.ndarray:
    """Time since previous observation, GRU-D eq. (2). Shapes: mask (T, D), hours (T,)."""
    T, D = mask.shape
    delta = np.zeros((T, D), dtype=np.float32)
    for t in range(1, T):
        gap = hours[t] - hours[t - 1]
        delta[t] = gap + (1.0 - mask[t - 1]) * delta[t - 1]
    return delta


def forward_fill(values: np.ndarray, fill_value: np.ndarray | float = 0.0) -> np.ndarray:
    """Carry the last observation forward; use ``fill_value`` before the first one."""
    T, D = values.shape
    out = np.empty_like(values, dtype=np.float32)
    last = np.broadcast_to(np.asarray(fill_value, dtype=np.float32), (D,)).copy()
    for t in range(T):
        obs = ~np.isnan(values[t])
        last[obs] = values[t, obs]
        out[t] = last
    return out


@dataclass
class Normalizer:
    """Z-score with statistics from observed training values only."""

    mean: np.ndarray
    std: np.ndarray
    static_mean: np.ndarray
    static_std: np.ndarray

    @classmethod
    def fit(cls, records: list[PatientRecord]) -> Normalizer:
        vals = np.concatenate([r.values for r in records], axis=0)
        stat = np.stack([r.static for r in records], axis=0)

        def _stats(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            mean = np.nan_to_num(np.nanmean(a, axis=0))
            std = np.nan_to_num(np.nanstd(a, axis=0), nan=1.0)
            return mean, np.where(std < 1e-6, 1.0, std)

        mean, std = _stats(vals)
        smean, sstd = _stats(stat) if stat.size else (np.zeros(0), np.ones(0))
        return cls(mean, std, smean, sstd)

    def to_dict(self) -> dict:
        return {k: getattr(self, k).tolist() for k in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, d: dict) -> Normalizer:
        return cls(**{k: np.asarray(v, dtype=float) for k, v in d.items()})


def featurize(record: PatientRecord, norm: Normalizer) -> dict[str, np.ndarray]:
    """Model inputs for one record. Missing static values become 0 (= training mean)."""
    z = (record.values - norm.mean) / norm.std
    m = compute_mask(record.values)
    return {
        "x": forward_fill(z, 0.0),
        "m": m,
        "delta": compute_delta(m, record.hours),
        "hours": record.hours.astype(np.float32),
        "static": np.nan_to_num((record.static - norm.static_mean) / norm.static_std).astype(
            np.float32
        ),
    }
