"""End-to-end smoke test: synthetic data -> train GRU -> metrics, for every input variant."""

import json

import pytest
import yaml

from missingness import paths, train
from missingness.config import load_config


@pytest.mark.parametrize("inputs", ["values", "pattern", "all"])
def test_smoke_training_run(tmp_path, monkeypatch, inputs):
    monkeypatch.setattr(train, "OUTPUT_DIR", tmp_path)
    cfg = load_config(
        paths.CONFIG_DIR / "experiment" / "smoke.yaml",
        [f"inputs={inputs}", "train.epochs=1", "eval.n_boot=20"],
    )
    metrics = train.train(cfg)
    assert set(metrics) >= {"val_auroc", "test_hospital1", "test_hospital2"}
    run = next(tmp_path.iterdir())
    assert (run / "model.pt").exists()
    assert json.loads((run / "metrics.json").read_text())["test_hospital1"]["all/auroc"]["value"]
    assert yaml.safe_load((run / "config.yaml").read_text())["git_commit"]

    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "collect_results", paths.REPO_ROOT / "scripts" / "collect_results.py"
    )
    collector = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(collector)
    table = collector.collect(tmp_path)
    assert set(table["test_set"]) == {"test_hospital1", "test_hospital2"}
    assert (table["perturb"] == "none").all() and table["inputs"].eq(inputs).all()


def test_config_overrides():
    cfg = load_config(paths.CONFIG_DIR / "experiment" / "sepsis_gru.yaml", ["train.lr=0.5"])
    assert cfg["train"]["lr"] == 0.5
    assert cfg["data"]["name"] == "physionet2019" and cfg["model"]["name"] == "gru"
    assert len(cfg["data"]["variables"]) == 34


def test_planted_run_and_rule_off_evaluation(tmp_path, monkeypatch):
    import subprocess
    import sys

    monkeypatch.setattr(train, "OUTPUT_DIR", tmp_path)
    cfg = load_config(
        paths.CONFIG_DIR / "experiment" / "smoke.yaml",
        ["train.epochs=1", "eval.n_boot=10", "run_name=planted"],
    )
    cfg["planted"] = {"variable": 0, "q1": 1.0, "q0": 0.25}
    train.train(cfg)
    out = subprocess.run(
        [
            sys.executable,
            "-m",
            "missingness.evaluate",
            str(tmp_path / "planted"),
            "--perturb",
            "planted_off",
        ],
        capture_output=True,
        text=True,
    )
    assert out.returncode == 0, out.stderr
    res = json.loads((tmp_path / "planted" / "perturbed" / "planted_off_seed0.json").read_text())
    assert "test_hospital1_rule_off" in res


def test_run_names_keep_planted_and_plain_runs_apart():
    cfg = load_config(paths.CONFIG_DIR / "experiment" / "mortality_gru.yaml", ["seed=1"])
    assert train.run_name(cfg) == "physionet2012_gru_all_seed1"
    cfg["planted"] = {"variable": "HR", "q1": 1.0, "q0": 1 / 1.5}
    assert train.run_name(cfg) == "physionet2012_gru_all_plantedHR-s1.5_seed1"
    cfg["planted"] = {"variable": "HR", "q1": 0.8, "q0": 0.8 / 1.5}
    assert train.run_name(cfg) == "physionet2012_gru_all_plantedHR-q0.8-s1.5_seed1"
    comparable = load_config(paths.CONFIG_DIR / "experiment" / "sepsis_comparable_gru.yaml")
    assert train.run_name(comparable) == "physionet2019_comparable_gru_all_seed0"


def test_tuning_run_leaves_test_sets_unscored(tmp_path, monkeypatch):
    monkeypatch.setattr(train, "OUTPUT_DIR", tmp_path)
    cfg = load_config(
        paths.CONFIG_DIR / "experiment" / "smoke.yaml",
        ["train.epochs=1", "eval.test=false", "run_name=tune_smoke"],
    )
    metrics = train.train(cfg)
    assert set(metrics) == {"val_auroc", "epochs"}


def test_evaluate_run_does_several_perturbations_and_skips_done_ones(tmp_path, monkeypatch):
    from missingness.evaluate import evaluate_run

    monkeypatch.setattr(train, "OUTPUT_DIR", tmp_path)
    cfg = load_config(
        paths.CONFIG_DIR / "experiment" / "smoke.yaml",
        ["train.epochs=1", "eval.n_boot=10", "run_name=run"],
    )
    train.train(cfg)
    jobs = [("random", 0.1, 0), ("state", 0.1, 0), ("thin", None, 1)]
    written = evaluate_run(tmp_path / "run", jobs)
    assert [f.name for f in written] == [
        "random_p0.1_seed0.json",
        "state_p0.1_seed0.json",
        "thin_seed1.json",
    ]
    preds = tmp_path / "run" / "perturbed" / "state_p0.1_seed0" / "test_hospital2.parquet"
    assert preds.exists()
    assert evaluate_run(tmp_path / "run", jobs) == []  # already done


def test_tuned_block_sets_values_and_command_line_still_wins(tmp_path):
    exp = tmp_path / "exp.yaml"
    exp.write_text(
        "data: synthetic\nmodel: gru\ninputs: all\nseed: 0\n"
        "train: {lr: 1.0e-3, weight_decay: 0.0}\n"
        "tuned:\n  model.hidden_size: 128\n  train.lr: 3.0e-4\n  train.weight_decay: 1.0e-5\n"
    )
    cfg = load_config(exp)
    assert cfg["model"]["hidden_size"] == 128 and cfg["train"]["lr"] == 3e-4
    assert cfg["train"]["weight_decay"] == 1e-5
    assert load_config(exp, ["train.lr=1.0e-3"])["train"]["lr"] == 1e-3
