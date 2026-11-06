import numpy as np
import pytest

from missingness.perturb import (
    check_only_removed,
    measurement_rates,
    paired_removal,
    thin_to_rates,
)


def test_paired_removal_keeps_patients_and_matches_counts(records):
    rand, state, achieved = paired_removal(records, p=0.3, seed=0)
    check_only_removed(records, rand)
    check_only_removed(records, state)
    assert 0 < achieved <= 0.3 + 1e-9
    for o, r, s in zip(records, rand, state, strict=True):
        assert o.n_observed - r.n_observed == o.n_observed - s.n_observed


def test_thinning_matches_target_rates_and_only_removes(records):
    h1 = [r for r in records if r.source == "hospital1"]  # measures more (synthetic)
    h2 = [r for r in records if r.source == "hospital2"]
    target = measurement_rates(h2)
    thinned = thin_to_rates(h1, target, seed=0)
    check_only_removed(h1, thinned)
    assert [r.patient_id for r in thinned] == [r.patient_id for r in h1]
    np.testing.assert_allclose(measurement_rates(thinned), target, atol=0.08)


def test_invariant_check_catches_added_values(records):
    bad = [r.with_values(np.nan_to_num(r.values)) for r in records]  # fills NaNs
    with pytest.raises(AssertionError, match="new observations"):
        check_only_removed(records, bad)
