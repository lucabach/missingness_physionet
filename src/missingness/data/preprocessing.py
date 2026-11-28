"""Turn raw values (NaN = missing) into model inputs: x (imputed), m (mask), delta (time gap).

Definitions follow GRU-D (Che et al. 2018):
    m[t, d]     = 1 if variable d was measured at step t
    delta[t, d] = hours since the last measurement of d before step t
                  (0 at the first step; grows while d is missing)
    x[t, d]     = last observed value of d (forward fill); the training mean before the
                  first measurement. In normalised units the training mean is 0.

Always recompute all three from ``values`` after any perturbation, never edit them directly.

Two other views of the same record exist for the RQ2 redesign (D29, proposed); both are
derived inputs like forward fill, never edits of ``values``:

    snapshot        a common schedule: one value per variable per fixed bin, shown at the bin
                    end, so every patient has the same measurement times (``snapshot_values``)
    decoded         m and delta rebuilt from the forward-filled x alone (``decode_mask``), to
                    measure how much of the pattern a values-only model can read
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


def decode_mask(x: np.ndarray, start: float = 0.0) -> np.ndarray:
    """Mask read off forward-filled values: 1 where x changes (or first leaves ``start``).

    This is what a values-only model can see of the pattern. It misses a measurement only
    when the new value equals the carried one exactly (or the first value equals the fill).
    """
    m = np.zeros_like(x, dtype=np.float32)
    if len(x):
        m[0] = x[0] != start
        m[1:] = x[1:] != x[:-1]
    return m


def bin_ends(hours: np.ndarray, width: int) -> np.ndarray:
    """Rows that close a bin of ``width`` hours, i.e. (hour + 1) is a multiple of the width.

    Bins are anchored at ICU hour 0, the same for every patient. A last bin cut short by the
    end of the stay has no end row and is never shown (its values would arrive at a
    patient-specific time).
    """
    h = np.round(hours).astype(int)
    return (h + 1) % width == 0


def snapshot_values(values: np.ndarray, hours: np.ndarray, widths: np.ndarray) -> np.ndarray:
    """Common-schedule view: per variable, the last value of each bin, placed at the bin end.

    ``widths[d]`` is the bin width of variable d in hours; 0 hides the variable (all NaN).
    Everything else is NaN, so within-bin timing, counts and busy hours are invisible. The
    value shown at a bin end was measured at or before that hour, so the view stays causal.
    Only an empty bin (no value at all) still differs between patients; its share is the leak
    that remains (``scripts/leak_audit.py`` reports it).
    """
    T, D = values.shape
    out = np.full((T, D), np.nan, dtype=np.float32)
    h = np.round(hours).astype(int)
    for d in range(D):
        w = int(widths[d])
        if w <= 0:
            continue
        ends = np.flatnonzero(bin_ends(hours, w))
        last = forward_fill_within_bins(values[:, d], h // w)
        out[ends, d] = last[ends]
    return out


def forward_fill_within_bins(col: np.ndarray, bins: np.ndarray) -> np.ndarray:
    """Carry the last value forward, restarting (NaN) at the start of every bin."""
    out = np.full(len(col), np.nan, dtype=np.float32)
    last, current = np.nan, None
    for t, (v, b) in enumerate(zip(col, bins, strict=True)):
        if b != current:
            last, current = np.nan, b
        if not np.isnan(v):
            last = v
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


FEATURE_VIEWS = ("forward_fill", "snapshot", "decoded")


def featurize(
    record: PatientRecord,
    norm: Normalizer,
    view: str = "forward_fill",
    snapshot_widths: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Model inputs for one record. Missing static values become 0 (= training mean).

    ``view``: ``forward_fill`` (default, Che et al. 2018), ``snapshot`` (common schedule with
    ``snapshot_widths`` per variable; m and delta then describe the schedule) or ``decoded``
    (x as usual, m and delta rebuilt from x only).
    """
    if view not in FEATURE_VIEWS:
        raise ValueError(f"view must be one of {FEATURE_VIEWS}, got {view!r}")
    values = record.values
    if view == "snapshot":
        if snapshot_widths is None:
            raise ValueError("the snapshot view needs snapshot_widths")
        values = snapshot_values(values, record.hours, snapshot_widths)
    z = (values - norm.mean) / norm.std
    x = forward_fill(z, 0.0)
    m = decode_mask(x) if view == "decoded" else compute_mask(values)
    return {
        "x": x,
        "m": m,
        "delta": compute_delta(m, record.hours),
        "hours": record.hours.astype(np.float32),
        "static": np.nan_to_num((record.static - norm.static_mean) / norm.static_std).astype(
            np.float32
        ),
    }
