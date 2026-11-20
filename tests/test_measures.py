"""The standard evaluation and the measures M1–M4 run end to end on synthetic data."""

import importlib.util

import yaml

from missingness import paths, train
from missingness.config import load_config
from missingness.evaluate import evaluate_run


def _script(name):
    spec = importlib.util.spec_from_file_location(name, paths.REPO_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_standard_evaluation_and_measures(tmp_path, monkeypatch):
    monkeypatch.setattr(train, "OUTPUT_DIR", tmp_path)
    smoke = paths.CONFIG_DIR / "experiment" / "smoke.yaml"
    base = ["train.epochs=1", "eval.n_boot=20"]
    for inputs in ("all", "values", "pattern"):
        for seed in (0, 1):
            train.train(load_config(smoke, base + [f"inputs={inputs}", f"seed={seed}"]))
    for inputs in ("all", "values"):
        for q1 in (1.0, 0.8):  # two designs of the rule (D21, D24), strengths 1 and 2
            for q0 in (q1, q1 / 2):
                cfg = load_config(smoke, base + [f"inputs={inputs}"])
                cfg["planted"] = {"variable": 0, "q1": q1, "q0": q0}
                train.train(cfg)

    jobs_of = _script("evaluate_runs").standard_jobs
    for run in sorted(tmp_path.iterdir()):
        evaluate_run(run, jobs_of(yaml.safe_load((run / "config.yaml").read_text())))

    measures = _script("measures")
    idx = measures.runs_index(tmp_path, None)
    table = measures.measure_rows(idx, ["auroc"], n_boot=20)
    found = {r["measure"] for r in table}
    assert found == {
        "M1_delta_mask",
        "M2_state_vs_random",
        "M4_thinning",
        "M4_transfer",
        "M3_gain",
        "M3_loss_off",
    }
    m1 = [r for r in table if r["measure"] == "M1_delta_mask" and r["subset"] == "all"]
    assert all(r["n_seeds"] == 2 and r["lo"] <= r["value"] <= r["hi"] for r in m1)
    assert all("S_all_rows" in r for r in m1)
    m3 = [r for r in table if r["measure"] == "M3_gain" and r["subset"] == "all"]
    assert sorted({r["planted_q1"] for r in m3}) == [0.8, 1.0]  # designs kept apart
