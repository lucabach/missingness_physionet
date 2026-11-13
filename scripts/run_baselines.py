"""Week-2 baselines: logistic regression and gradient boosting on causal hand-made features.

    python scripts/run_baselines.py               # both tasks + SOFA check (~15 min on 4 CPUs)
    python scripts/run_baselines.py --task mortality --n-boot 200

Sepsis trains on ``train_source`` (hospital 1, D11) and is tested on both hospitals' test
splits; mortality on the PhysioNet 2012 test split. Writes results/baselines.csv and, for
sepsis, results/sofa_check.csv: gradient boosting on all features with the pattern features
of the SOFA variables removed, against removing the pattern features of the same number of
randomly chosen other variables (open question from the proposal: how much of the sepsis
signal comes from the label being computed from lab values).
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from missingness.baselines.features import FEATURE_SETS, build_features, feature_names
from missingness.config import load_config
from missingness.data.preprocessing import Normalizer
from missingness.eval.metrics import auroc, pre_onset_mask
from missingness.experiment import load_data, score_predictions
from missingness.paths import CONFIG_DIR, REPO_ROOT

EXPERIMENTS = {"sepsis": "sepsis_gru.yaml", "mortality": "mortality_gru.yaml"}


def make_model(name: str, seed: int = 0):
    if name == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    return HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.1, early_stopping=True, random_state=seed
    )


def predict_frame(model, data: dict, cols: np.ndarray | None = None) -> pd.DataFrame:
    X = data["X"] if cols is None else data["X"][:, cols]
    return pd.DataFrame(
        {
            "patient_id": data["patient_id"],
            "hour": data["hour"],
            "label": data["y"],
            "score": model.predict_proba(X)[:, 1],
        }
    )


def flatten(task, test_set, features, model, metrics) -> list[dict]:
    rows = []
    for key, m in metrics.items():
        subset, metric = key.split("/")
        rows.append(
            {
                "task": task,
                "test_set": test_set,
                "features": features,
                "model": model,
                "subset": subset,
                "metric": metric,
                **m,
            }
        )
    return rows


def run_task(task: str, n_boot: int) -> tuple[pd.DataFrame, dict]:
    cfg = load_config(CONFIG_DIR / "experiment" / EXPERIMENTS[task])
    step = cfg["data"]["task"]
    sets = load_data(cfg)
    norm = Normalizer.fit(sets["train"])
    kinds = [k for k in FEATURE_SETS if not (k == "time" and step == "stay")]
    rows, cache = [], {}
    for kind in kinds:
        feats = {
            name: build_features(recs, kind, step, norm.mean, norm.static_mean)
            for name, recs in sets.items()
            if name == "train" or name.startswith("test")
        }
        cache[kind] = feats
        for model_name in ("logreg", "hgb"):
            t0 = time.time()
            model = make_model(model_name).fit(feats["train"]["X"], feats["train"]["y"])
            for name, data in feats.items():
                if not name.startswith("test"):
                    continue
                pred = predict_frame(model, data)
                metrics = score_predictions(pred, step, n_boot=n_boot)
                rows += flatten(task, name, kind, model_name, metrics)
                print(
                    f"{task:9s} {kind:8s} {model_name:6s} {name:15s} "
                    f"AUROC {metrics['all/auroc']['value']:.3f}  ({time.time() - t0:.0f}s)"
                )
    return pd.DataFrame(rows), {"cfg": cfg, "feats": cache.get("all")}


def sofa_check(cfg: dict, feats: dict, n_random: int = 5, seed: int = 0) -> pd.DataFrame:
    variables, static = cfg["data"]["variables"], cfg["data"]["static"]
    names = np.array(feature_names(variables, static, "all"))
    sofa = cfg["data"]["sofa_variables"]
    others = [v for v in variables if v not in sofa]

    def drop_pattern(vs: list[str]) -> np.ndarray:
        drop = {f"{p}_{v}" for v in vs for p in ("count", "since", "measured")}
        return np.flatnonzero([n not in drop for n in names])

    rng = np.random.default_rng(seed)
    variants = [("none", "", np.arange(len(names))), ("sofa", ",".join(sofa), drop_pattern(sofa))]
    for k in range(n_random):
        pick = sorted(rng.choice(others, size=len(sofa), replace=False).tolist())
        variants.append((f"random{k}", ",".join(pick), drop_pattern(pick)))

    rows = []
    for name, dropped, cols in variants:
        model = make_model("hgb").fit(feats["train"]["X"][:, cols], feats["train"]["y"])
        for test in [k for k in feats if k.startswith("test")]:
            pred = predict_frame(model, feats[test], cols)
            pre = pre_onset_mask(pred["patient_id"], pred["hour"], pred["label"])
            rows.append(
                {
                    "variant": name,
                    "pattern_dropped_for": dropped,
                    "test_set": test,
                    "auroc_all": auroc(pred["label"], pred["score"]),
                    "auroc_pre_onset": auroc(pred["label"][pre], pred["score"][pre]),
                }
            )
            print(f"sofa-check {name:8s} {test:15s} AUROC {rows[-1]['auroc_all']:.3f}")
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["sepsis", "mortality", "both"], default="both")
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--no-sofa", action="store_true")
    args = ap.parse_args()
    out = REPO_ROOT / "results"
    out.mkdir(exist_ok=True)
    tasks = ["sepsis", "mortality"] if args.task == "both" else [args.task]
    tables = []
    for task in tasks:
        table, extra = run_task(task, args.n_boot)
        tables.append(table)
        if task == "sepsis" and not args.no_sofa:
            sofa_check(extra["cfg"], extra["feats"]).to_csv(out / "sofa_check.csv", index=False)
    pd.concat(tables).to_csv(out / "baselines.csv", index=False)
    print(f"-> {out / 'baselines.csv'}")


if __name__ == "__main__":
    main()
