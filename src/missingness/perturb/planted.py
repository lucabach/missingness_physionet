"""Planted pattern (experiment E5, plan P4): a measurement habit whose effect we know.

On real records, measurements of one variable are kept with probability ``q1`` for patients
with the outcome and ``q0`` for the others. With q1 > q0 the mask of that variable now
carries outcome information that we put there ourselves; with q1 = q0 it carries none. The
strength of the rule is q1 / q0. D24: q1 = 0.8 and strength {1, 1.05, 1.1, 1.2} (1 is the
control). With q1 = 1 (D21) the patients with the outcome never lose a value, so an hour with
other vital signs but no HR gives the label away: the GRU reached AUROC 0.999 at strength 1.5.

"Rule off" is the test condition where the same share of that variable is deleted for every
patient, whatever the outcome. A model that learned the planted habit should lose accuracy
when the rule is switched off; the ``values`` model should not.

Removal only, same patients (invariants 2 and 3).
"""

from __future__ import annotations

import numpy as np

from missingness.data.records import PatientRecord


def planted_index(planted: dict, variables: list[str]) -> int:
    """Column of the planted variable; the config gives a name from data.variables or an index."""
    v = planted["variable"]
    return v if isinstance(v, int) else variables.index(v)


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
