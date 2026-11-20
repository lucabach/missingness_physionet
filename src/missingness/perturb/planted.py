"""Planted pattern (experiment E5, plan P4): a measurement habit whose effect we know.

On real records, measurements of one variable are kept with probability ``q1`` for patients
with the outcome and ``q0`` for the others. With q1 > q0 the mask of that variable now
carries outcome information that we put there ourselves; with q1 = q0 it carries none. The
strength of the rule is q1 / q0 (plan: 1, 1.5, 2, 4; 1 is the control).

"Rule off" is the test condition where the same share of that variable is deleted for every
patient, whatever the outcome. A model that learned the planted habit should lose accuracy
when the rule is switched off; the ``values`` model should not.

Removal only, same patients (invariants 2 and 3). Code only, no runs before ``protocol-v1``.
"""

from __future__ import annotations

import numpy as np

from missingness.data.records import PatientRecord


def _keep(record: PatientRecord, d: int, prob: float, rng: np.random.Generator) -> PatientRecord:
    v = record.values.copy()
    drop = ~np.isnan(v[:, d]) & (rng.random(len(v)) >= prob)
    v[drop, d] = np.nan
    return record.with_values(v)


def planted_removal(
    records: list[PatientRecord], variable: int, q1: float, q0: float, seed: int = 0
) -> list[PatientRecord]:
    """Keep observations of ``variable`` with prob q1 (outcome) or q0 (no outcome)."""
    assert 0 <= q0 <= 1 and 0 <= q1 <= 1
    rng = np.random.default_rng(seed)
    return [_keep(r, variable, q1 if r.stay_label else q0, rng) for r in records]


def planted_keep_rate(records: list[PatientRecord], variable: int, q1: float, q0: float) -> float:
    """Expected share of ``variable``'s observations the rule keeps on these records."""
    n = np.array([(~np.isnan(r.values[:, variable])).sum() for r in records], dtype=float)
    q = np.array([q1 if r.stay_label else q0 for r in records])
    return float((n * q).sum() / max(n.sum(), 1))


def planted_off(
    records: list[PatientRecord], variable: int, q1: float, q0: float, seed: int = 0
) -> list[PatientRecord]:
    """Rule off: the same overall deletion rate as the rule, for every patient alike."""
    rate = planted_keep_rate(records, variable, q1, q0)
    rng = np.random.default_rng(seed)
    return [_keep(r, variable, rate, rng) for r in records]
