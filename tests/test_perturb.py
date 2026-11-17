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


def test_matched_random_removal_takes_same_count_per_variable(records):
    rand, state, _ = paired_removal(records, p=0.1, seed=1, match_variables=True)
    check_only_removed(records, rand)
    for o, r, s in zip(records, rand, state, strict=True):
        obs = ~np.isnan(o.values)
        removed_r = (obs & np.isnan(r.values)).sum(axis=0)
        removed_s = (obs & np.isnan(s.values)).sum(axis=0)
        np.testing.assert_array_equal(removed_r, removed_s)


def test_unmatched_random_removal_still_matches_total(records):
    rand, state, _ = paired_removal(records, p=0.1, seed=1, match_variables=False)
    check_only_removed(records, rand)
    for r, s in zip(rand, state, strict=True):
        assert r.n_observed == s.n_observed


def test_planted_rule_only_removes_and_links_mask_to_outcome(records):
    from missingness.perturb import planted_keep_rate, planted_off, planted_removal

    on = planted_removal(records, variable=0, q1=1.0, q0=0.25, seed=0)
    off = planted_off(records, variable=0, q1=1.0, q0=0.25, seed=0)
    check_only_removed(records, on)
    check_only_removed(records, off)

    def kept(recs, pos):
        o = [r for r, r0 in zip(recs, records, strict=True) if r0.stay_label == pos]
        base = [r for r in records if r.stay_label == pos]
        return sum((~np.isnan(r.values[:, 0])).sum() for r in o) / max(
            sum((~np.isnan(r.values[:, 0])).sum() for r in base), 1
        )

    assert kept(on, 1) == 1.0 and kept(on, 0) < 0.4  # rule on: outcome decides
    rate = planted_keep_rate(records, 0, 1.0, 0.25)
    assert abs(kept(off, 1) - rate) < 0.15 and abs(kept(off, 0) - rate) < 0.15  # rule off
    same = planted_removal(records, variable=0, q1=0.5, q0=0.5, seed=0)
    assert abs(kept(same, 1) - kept(same, 0)) < 0.15  # control: no link
    for r, r0 in zip(on, records, strict=True):  # other variables untouched
        np.testing.assert_array_equal(np.isnan(r.values[:, 1:]), np.isnan(r0.values[:, 1:]))
