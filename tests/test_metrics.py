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


def test_paired_bootstrap_identical_scores_give_zero():
    from missingness.eval import bootstrap_diff

    rng = np.random.default_rng(0)
    pids = np.repeat(np.arange(60), 3)
    y = np.repeat(rng.integers(0, 2, 60), 3)
    s = y + rng.normal(0, 1, len(y))
    res = bootstrap_diff(y, s, s, pids, n_boot=100)
    assert res["value"] == 0 and res["lo"] == 0 and res["hi"] == 0


def test_paired_bootstrap_is_narrower_than_separate_intervals():
    from missingness.eval import bootstrap_diff

    rng = np.random.default_rng(1)
    pids = np.repeat(np.arange(150), 2)
    y = np.repeat(rng.integers(0, 2, 150), 2)
    a = y + rng.normal(0, 1.2, len(y))
    b = a + rng.normal(0, 0.3, len(y))  # a close copy of model a
    diff = bootstrap_diff(y, a, b, pids, n_boot=300)
    ci_a, ci_b = bootstrap_ci(y, a, pids, n_boot=300), bootstrap_ci(y, b, pids, n_boot=300)
    assert diff["hi"] - diff["lo"] < (ci_a["hi"] - ci_a["lo"]) + (ci_b["hi"] - ci_b["lo"])
    assert diff["hi"] - diff["lo"] < ci_a["hi"] - ci_a["lo"]


def test_score_predictions_reports_pre_onset_without_positive_at_start():
    import pandas as pd

    from missingness.experiment import score_predictions

    rng = np.random.default_rng(0)
    rows = []
    for k in range(40):
        septic, at_start = k % 4 == 0, k % 8 == 0
        for h in range(1, 13):
            label = int(septic and (at_start or h >= 6))
            rows.append((f"p{k}", h, label, label + rng.normal()))
    pred = pd.DataFrame(rows, columns=["patient_id", "hour", "label", "score"])
    out = score_predictions(pred, "hourly", n_boot=20)
    assert {"all/auroc", "pre_onset/auroc", "pre_onset_excl_positive_at_start/auroc"} <= set(out)


def _copy_rows_bootstrap(y, scores, weights, pids, metric, n_boot, seed=0):
    """The first implementation: copy the rows of every drawn patient (reference)."""
    from missingness.eval.metrics import METRICS

    fn = METRICS[metric]
    _, inverse = np.unique(pids, return_inverse=True)
    rows_of = [np.flatnonzero(inverse == k) for k in range(inverse.max() + 1)]
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(rows_of), size=len(rows_of))
        idx = np.concatenate([rows_of[k] for k in pick])
        if y[idx].min() == y[idx].max():
            continue
        stats.append(sum(w * fn(y[idx], s[idx]) for w, s in zip(weights, scores, strict=True)))
    return np.quantile(stats, [0.025, 0.975]), len(stats)


def test_weighted_bootstrap_equals_copying_rows():
    from missingness.eval import bootstrap_contrast

    rng = np.random.default_rng(3)
    pids = np.repeat(np.arange(40), rng.integers(1, 6, 40))
    y = (rng.random(len(pids)) < 0.2).astype(float)
    a = np.round(y + rng.normal(0, 1, len(y)), 1)  # rounded: many ties
    b = np.round(a + rng.normal(0, 0.5, len(y)), 1)
    for metric in ("auroc", "auprc"):
        for scores, weights in (([a], [1.0]), ([a, b], [1.0, -1.0])):
            res = bootstrap_contrast(y, scores, weights, pids, metric, n_boot=200)
            (lo, hi), n = _copy_rows_bootstrap(y, scores, weights, pids, metric, 200)
            assert res["n_boot"] == n
            assert abs(res["lo"] - lo) < 1e-9 and abs(res["hi"] - hi) < 1e-9
