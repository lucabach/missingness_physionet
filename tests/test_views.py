"""RQ2 redesign (D29): decoded mask, common-schedule view, channel intervention, leak audit."""

import importlib.util

import numpy as np
import yaml

from missingness import paths, train
from missingness.config import load_config
from missingness.data.dataset import TimeSeriesDataset
from missingness.data.preprocessing import (
    Normalizer,
    compute_mask,
    decode_mask,
    featurize,
    forward_fill,
    snapshot_values,
)
from missingness.data.records import PatientRecord
from missingness.data.views import feature_kwargs, resolve_snapshot
from missingness.eval.leak import mask_recovery
from missingness.evaluate import evaluate_run
from missingness.perturb import check_only_removed

NAN = np.nan


def _record(values, pid="p0"):
    values = np.asarray(values, dtype=float)
    T = len(values)
    return PatientRecord(pid, "hospital1", np.arange(T, dtype=float), values, np.zeros(1),
                         np.zeros(T))  # fmt: skip


def _script(name):
    spec = importlib.util.spec_from_file_location(name, paths.REPO_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_forward_fill_leaks_the_mask_except_exact_repeats():
    values = np.array([[NAN], [1.0], [NAN], [2.0], [2.0], [NAN]])
    x = forward_fill(values, 0.0)
    assert decode_mask(x)[:, 0].tolist() == [0, 1, 0, 1, 0, 0]  # the repeated 2.0 is missed
    assert compute_mask(values)[:, 0].tolist() == [0, 1, 0, 1, 1, 0]


def test_snapshot_shows_the_last_value_of_each_bin_at_its_end():
    values = np.array([[1.0, 5.0], [NAN, NAN], [3.0, NAN], [4.0, NAN], [NAN, NAN], [NAN, 7.0]])
    out = snapshot_values(values, np.arange(6.0), np.array([2, 0]))
    assert np.isnan(out[[0, 2, 4], 0]).all()
    assert out[1, 0] == 1.0 and out[3, 0] == 4.0 and np.isnan(out[5, 0])  # bin 3 is empty
    assert np.isnan(out[:, 1]).all()  # width 0 hides the variable


def test_snapshot_drops_a_last_bin_cut_short_by_the_end_of_the_stay():
    out = snapshot_values(np.array([[1.0], [2.0], [3.0]]), np.arange(3.0), np.array([2]))
    assert out[1, 0] == 2.0 and np.isnan(out[2, 0])


def test_snapshot_view_is_blind_to_timing_and_extra_measurements_within_a_bin():
    widths = np.array([4])
    a = _record([[1.0], [NAN], [NAN], [2.0], [NAN], [3.0], [3.5], [2.5]])
    b = _record([[NAN], [1.5], [NAN], [2.0], [9.0], [NAN], [NAN], [2.5]])  # other times, counts
    norm = Normalizer(np.zeros(1), np.ones(1), np.zeros(1), np.ones(1))
    fa = featurize(a, norm, view="snapshot", snapshot_widths=widths)
    fb = featurize(b, norm, view="snapshot", snapshot_widths=widths)
    for key in ("x", "m", "delta"):
        np.testing.assert_array_equal(fa[key], fb[key])


def test_snapshot_view_is_causal(records):
    norm = Normalizer.fit(records)
    widths = np.array([2, 2, 4, 4, 4])
    r = records[0]
    later = r.values.copy()
    later[10:] = 5.0
    f1 = featurize(r, norm, view="snapshot", snapshot_widths=widths)
    f2 = featurize(r.with_values(later), norm, view="snapshot", snapshot_widths=widths)
    np.testing.assert_array_equal(f1["x"][:10], f2["x"][:10])
    np.testing.assert_array_equal(f1["m"][:10], f2["m"][:10])


def test_resolve_snapshot_keeps_covered_variables_and_names_the_run():
    cfg = load_config(paths.CONFIG_DIR / "experiment" / "smoke.yaml", ["features=snapshot"])
    from missingness.experiment import load_data

    resolve_snapshot(cfg, load_data(cfg)["train"])
    widths = cfg["data"]["snapshot"]["widths"]
    assert widths["v0"] == 2 and widths["v4"] == 4  # synthetic rates give coverage > 0.5
    assert feature_kwargs(cfg)["snapshot_widths"].tolist() == [2, 2, 4, 4, 4]
    assert train.run_name(cfg).endswith("_all_snapshot_seed0")
    cfg["data"]["snapshot"]["tag"] = "cov90"
    assert train.run_name(cfg).endswith("_all_snapshot-cov90_seed0")
    del cfg["data"]["snapshot"]["tag"]
    cfg["data"]["snapshot"] = dict(cfg["data"]["snapshot"], widths=None, min_coverage=1.01)
    resolve_snapshot(cfg, load_data(cfg)["train"])
    assert set(cfg["data"]["snapshot"]["widths"].values()) == {0}


def test_channel_dataset_takes_pattern_from_the_other_records(records):
    norm = Normalizer.fit(records)
    thinned = [r.with_values(np.where(np.arange(len(r.hours))[:, None] % 2 == 0, r.values, NAN))
               for r in records]  # fmt: skip
    check_only_removed(records, thinned)
    mixed = TimeSeriesDataset(records, norm, "hourly", pattern_records=thinned)
    clean = TimeSeriesDataset(records, norm, "hourly")
    other = TimeSeriesDataset(thinned, norm, "hourly")
    for i in range(3):
        np.testing.assert_array_equal(mixed[i]["x"], clean[i]["x"])
        np.testing.assert_array_equal(mixed[i]["m"], other[i]["m"])
        np.testing.assert_array_equal(mixed[i]["delta"], other[i]["delta"])


def test_mask_recovery_finds_every_measurement_of_continuous_values(records):
    norm = Normalizer.fit(records)
    names = [f"v{i}" for i in range(records[0].values.shape[1])]
    table = mask_recovery(records, norm, names).set_index("variable")
    assert table.loc["ALL", "recall"] > 0.99 and table.loc["ALL", "false_hits"] == 0
    snap = mask_recovery(records, norm, names, {"view": "snapshot",
                                                 "snapshot_widths": np.full(5, 2)})  # fmt: skip
    assert 0 <= snap.set_index("variable").loc["ALL", "empty_bin_share"] < 0.5


def test_d29_runs_channel_evaluation_and_measures(tmp_path, monkeypatch):
    monkeypatch.setattr(train, "OUTPUT_DIR", tmp_path)
    smoke = paths.CONFIG_DIR / "experiment" / "smoke.yaml"
    base = ["train.epochs=1", "eval.n_boot=20"]
    for seed in (0, 1):
        for extra in (
            ["inputs=all"],
            ["inputs=values"],
            ["inputs=pattern"],
            ["inputs=pattern", "features=decoded"],
            ["inputs=values", "features=snapshot"],
            ["inputs=all", "features=snapshot"],
        ):
            train.train(load_config(smoke, base + extra + [f"seed={seed}"]))
    saved = yaml.safe_load((tmp_path / "synthetic_gru_values_snapshot_seed0" / "config.yaml")
                           .read_text())  # fmt: skip
    assert saved["data"]["snapshot"]["widths"]  # evaluation reuses the resolved widths

    ev = _script("evaluate_runs")
    jobs = [("random", 0.1, 0), ("state", 0.1, 0), ("thin", None, 0), ("thin", None, 1),
            ("thin", None, 2)]  # fmt: skip
    for run in sorted(tmp_path.iterdir()):
        cfg = yaml.safe_load((run / "config.yaml").read_text())
        if cfg["inputs"] == "pattern":
            continue
        evaluate_run(run, jobs)
        if ev.channel_jobs(cfg) and cfg.get("features", "forward_fill") == "forward_fill":
            for channel in ("pattern", "values"):
                evaluate_run(run, jobs, channel=channel)
    assert (tmp_path / "synthetic_gru_all_seed0" / "perturbed" / "state_p0.1_chpattern_seed0.json"
            ).exists()  # fmt: skip

    measures = _script("measures")
    idx = measures.runs_index(tmp_path, None)
    found = {r["measure"] for r in measures.measure_rows(idx, ["auroc"], n_boot=20)}
    assert found >= {
        "M1_decoded_gap",
        "M1_bound_snapshot",
        "M2_ref_snapshot",
        "M4_thinning_ref_snapshot",
        "M2_channel_pattern",
        "M2_channel_values",
        "M4_thinning_channel_pattern",
    }
    table = _script("collect_results").collect(tmp_path)
    assert {"forward_fill", "snapshot", "decoded"} <= set(table["view"])
    assert "state_p0.1_chpattern_seed0" in set(table["perturb"])
