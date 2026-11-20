"""Select a predictor from the six full-input Week 3 checkpoints for one task.

    python scripts/select_predictor.py mortality
    python scripts/select_predictor.py sepsis

Candidates are the six individual seeds, the mean of each model's three seeds, and
three mixtures of those means. Choose by validation AUROC (pre-onset hours for sepsis).
The JSON report records the run names and probability weights needed for inference.
"""

import argparse
import json

import numpy as np
import pandas as pd
import torch
import yaml

from missingness.data.preprocessing import Normalizer
from missingness.eval.metrics import auroc, bootstrap_diff
from missingness.experiment import eval_subsets, load_data, predict, score_predictions
from missingness.models import build_model
from missingness.paths import OUTPUT_DIR, REPO_ROOT
from missingness.utils import get_device


def candidates() -> dict[str, np.ndarray]:
    weights = {f"single_{i}": w for i, w in enumerate(np.eye(6))}
    for fraction in (0, 0.25, 0.5, 0.75, 1):
        weights[f"grud_fraction_{fraction}"] = np.repeat([fraction, 1 - fraction], 3) / 3
    return weights


def select(validation: pd.DataFrame, scores: np.ndarray, task: str) -> tuple[dict, str, str]:
    subset = "pre_onset" if task == "hourly" else "all"
    keep = eval_subsets(validation, task)[subset]
    aucs = {
        name: auroc(validation.label[keep], scores[keep] @ weights)
        for name, weights in candidates().items()
    }
    best = max(aucs, key=aucs.get)
    best_single = max((k for k in aucs if k.startswith("single_")), key=aucs.get)
    return aucs, best, best_single


def aligned_scores(frame: pd.DataFrame, reference: pd.DataFrame) -> np.ndarray:
    keys = ["patient_id", "hour", "label"]
    if not frame[keys].equals(reference[keys]):
        raise ValueError("predictions must have identical patients, hours and labels")
    return frame.score.to_numpy(dtype=float)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("task", choices=["mortality", "sepsis"])
    args = ap.parse_args()
    dataset = "physionet2012" if args.task == "mortality" else "physionet2019"
    runs = [
        OUTPUT_DIR / f"{dataset}_{model}_all_seed{seed}"
        for model in ("grud", "odernn")
        for seed in range(3)
    ]
    device = get_device()
    val_scores = []
    reference = None
    for run in runs:
        cfg = yaml.safe_load((run / "config.yaml").read_text())
        task = cfg["data"]["task"]
        if reference is None:
            validation_records = load_data(cfg)["val"]
        norm = Normalizer.from_dict(json.loads((run / "normalizer.json").read_text()))
        model = build_model(cfg["model"], len(norm.mean), len(norm.static_mean), "all").to(device)
        model.load_state_dict(torch.load(run / "model.pt", map_location=device, weights_only=True))
        pred = predict(model, validation_records, norm, task, device)
        if reference is None:
            reference = pred
        val_scores.append(aligned_scores(pred, reference))

    aucs, best, best_single = select(reference, np.column_stack(val_scores), task)
    weights = candidates()[best]
    report = {
        "weights": {r.name: float(w) for r, w in zip(runs, weights, strict=True) if w},
        "selection": best,
        "validation_auroc": aucs,
        "checkpoint_order": [r.name for r in runs],
        "best_individual": best_single,
        "test": {},
    }
    for split in ["test"] if task == "stay" else ["test_hospital1", "test_hospital2"]:
        frames = [pd.read_parquet(r / "predictions" / f"{split}.parquet") for r in runs]
        scores = np.column_stack([aligned_scores(f, frames[0]) for f in frames])
        pred = frames[0].assign(score=scores @ weights)
        subset = "all" if task == "stay" else "pre_onset"
        keep = eval_subsets(pred, task)[subset]
        baseline = scores @ candidates()[best_single]
        metrics = score_predictions(pred, task)
        metrics["headline_gain_vs_best_individual"] = bootstrap_diff(
            pred.label.to_numpy()[keep],
            pred.score.to_numpy()[keep],
            baseline[keep],
            pred.patient_id.to_numpy()[keep],
            metric="auroc",
        )
        report["test"][split] = metrics
    path = REPO_ROOT / "results" / f"predictor_{args.task}.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(path)


if __name__ == "__main__":
    main()
