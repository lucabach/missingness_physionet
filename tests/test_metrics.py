import numpy as np

from missingness.eval import bootstrap_ci, delta_mask, information_share, pre_onset_mask


def test_bootstrap_ci_contains_point_estimate():
    rng = np.random.default_rng(0)
    pids = np.repeat(np.arange(50), 4)
    y = np.repeat(rng.integers(0, 2, 50), 4)
    score = y + rng.normal(0, 1, len(y))
    res = bootstrap_ci(y, score, pids, n_boot=200)
    assert res["lo"] <= res["value"] <= res["hi"]


def test_pre_onset_mask_drops_hours_from_clinical_onset():
    pids = np.array(["a"] * 10 + ["b"] * 3)
    hours = np.array(list(range(10)) + [0, 1, 2])
    labels = np.array([0, 0, 1, 1, 1, 1, 1, 1, 1, 1] + [0, 0, 0])
    keep = pre_onset_mask(pids, hours, labels)
    # first positive hour 2 -> onset at 8: keep hours 0..7 of "a", all of "b"
    assert keep.tolist() == [True] * 8 + [False] * 2 + [True] * 3


def test_information_scores():
    assert abs(information_share(0.7, 0.9) - 0.5) < 1e-12
    assert abs(delta_mask(0.9, 0.85) - 0.05) < 1e-12
