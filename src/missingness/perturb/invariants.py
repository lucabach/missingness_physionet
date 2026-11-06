"""Guarantees every perturbed test set must satisfy (Sarah: same patients, always)."""

from __future__ import annotations

import numpy as np

from missingness.data.records import PatientRecord


def check_only_removed(original: list[PatientRecord], perturbed: list[PatientRecord]) -> None:
    """Raise AssertionError unless ``perturbed`` = ``original`` with some values removed.

    Checks: same patients in the same order, same hours and labels, every remaining value
    was observed in the original with the identical value, no new observations.
    """
    assert len(original) == len(perturbed), "number of patients changed"
    for o, p in zip(original, perturbed, strict=True):
        assert o.patient_id == p.patient_id, f"patient order changed at {o.patient_id}"
        assert np.array_equal(o.hours, p.hours), f"{o.patient_id}: hours changed"
        assert np.array_equal(o.labels, p.labels), f"{o.patient_id}: labels changed"
        assert np.array_equal(o.static, p.static, equal_nan=True), f"{o.patient_id}: static"
        obs_p = ~np.isnan(p.values)
        obs_o = ~np.isnan(o.values)
        assert not (obs_p & ~obs_o).any(), f"{o.patient_id}: new observations were added"
        assert np.array_equal(o.values[obs_p], p.values[obs_p]), f"{o.patient_id}: values changed"
