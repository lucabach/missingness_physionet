"""End-to-end smoke test: synthetic data -> train GRU -> metrics, for every input variant."""

import json

import pytest

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


def test_config_overrides():
    cfg = load_config(paths.CONFIG_DIR / "experiment" / "sepsis_gru.yaml", ["train.lr=0.5"])
    assert cfg["train"]["lr"] == 0.5
    assert cfg["data"]["name"] == "physionet2019" and cfg["model"]["name"] == "gru"
    assert len(cfg["data"]["variables"]) == 34
