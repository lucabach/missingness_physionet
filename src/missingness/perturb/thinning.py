"""Rate-matched thinning: make hospital 1 test patients look measured like hospital 2.

For each variable, rate = observed values per patient-hour. Where the source hospital
measures more than the target, each observation is kept with probability
target_rate / source_rate; where it measures less, nothing can be done (thinning only
removes) and the variable is left unchanged. This is why thinning runs from the hospital
that measures more to the one that measures less.

First version for the week-2 owner to review (e.g. per-variable vs per-patient matching).
"""

from __future__ import annotations

import numpy as np

from missingness.data.records import PatientRecord


def measurement_rates(records: list[PatientRecord]) -> np.ndarray:
    """Observed values per patient-hour for each variable, shape (D,)."""
    obs = np.concatenate([~np.isnan(r.values) for r in records], axis=0)
    return obs.mean(axis=0)


def thin_to_rates(
    records: list[PatientRecord], target_rates: np.ndarray, seed: int = 0
) -> list[PatientRecord]:
    rng = np.random.default_rng(seed)
    src = measurement_rates(records)
    keep_prob = np.where(src > target_rates, target_rates / np.maximum(src, 1e-12), 1.0)
    out = []
    for r in records:
        drop = (~np.isnan(r.values)) & (rng.random(r.values.shape) > keep_prob)
        v = r.values.copy()
        v[drop] = np.nan
        out.append(r.with_values(v))
    return out
