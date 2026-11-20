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
