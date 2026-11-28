"""Collect every run's metrics into one table (plan, rule 4: figures come from this file only).

    python scripts/collect_results.py                    # -> results/runs.csv
    python scripts/collect_results.py --models grud odernn   # only these; other rows stay
    python scripts/collect_results.py --out results/runs_mortality.csv --outputs outputs/

One row per run, test set, subset and metric; tuning runs (``tune_*``) have one row with
``test_set = none`` that carries their validation AUROC. Clean test sets come from
``outputs/<run>/metrics.json``; perturbed ones from ``outputs/<run>/perturbed/*.json``
(column ``perturb``: ``none``, ``random_p0.1``, ``state_p0.1``, ``thin``,
``state_p0.1_chpattern`` ...; column ``view``: the run's input view, D29).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from missingness.data.views import view_name
from missingness.paths import OUTPUT_DIR, REPO_ROOT
from missingness.results import write_table

RUN_KEYS = {
    "dataset": ("data", "name"),
    "data_tag": ("data", "tag"),  # e.g. physionet2019_comparable (25 variables); else = dataset
    "train_source": ("data", "train_source"),
    "model": ("model", "name"),
    "inputs": ("inputs",),
    "seed": ("seed",),
    "hidden_size": ("model", "hidden_size"),
    "lr": ("train", "lr"),
    "weight_decay": ("train", "weight_decay"),
    "git_commit": ("git_commit",),
}


def run_info(cfg: dict, run_dir: Path) -> dict:
    """Columns that say which run a row belongs to (enough to group runs without the name)."""
    info = {"run": run_dir.name} | {k: _get(cfg, p) for k, p in RUN_KEYS.items()}
    info["data_tag"] = info["data_tag"] or info["dataset"]
    info["view"] = view_name(cfg)  # D29: forward_fill, decoded, snapshot[-<tag>]
    info["n_vars"] = len(_get(cfg, ("data", "variables")) or [])
    pl = cfg.get("planted") or {}
    info["planted_variable"] = pl.get("variable")
    info["planted_strength"] = pl["q1"] / pl["q0"] if pl else None
    info["planted_q1"] = pl.get("q1")
    info["tuning"] = run_dir.name.startswith("tune_")
    metrics = run_dir / "metrics.json"
    m = json.loads(metrics.read_text()) if metrics.exists() else {}
    info["val_auroc"], info["epochs"] = m.get("val_auroc"), m.get("epochs")
    return info


def _get(cfg: dict, path: tuple[str, ...]):
    for k in path:
        if not isinstance(cfg, dict) or k not in cfg:
            return None
        cfg = cfg[k]
    return cfg


def metric_rows(metrics: dict) -> list[dict]:
    """Flatten {test_set: {"subset/metric": {value, lo, hi}}} into rows."""
    rows = []
    for test_set, block in metrics.items():
        if not isinstance(block, dict):
            continue
        for key, m in block.items():
            if not (isinstance(m, dict) and "value" in m and "/" in key):
                continue
            subset, metric = key.split("/", 1)
            rows.append(
                {"test_set": test_set, "subset": subset, "metric": metric}
                | {k: m.get(k) for k in ("value", "lo", "hi")}
            )
    return rows


def collect(outputs: Path, models: list[str] | None = None) -> pd.DataFrame:
    rows = []
    for cfg_path in sorted(outputs.glob("*/config.yaml")):
        run_dir = cfg_path.parent
        cfg = yaml.safe_load(cfg_path.read_text())
        if models and cfg["model"]["name"] not in models:
            continue
        run = run_info(cfg, run_dir)
        sources = []
        if (run_dir / "metrics.json").exists():
            sources.append(("none", json.loads((run_dir / "metrics.json").read_text())))
        for f in sorted((run_dir / "perturbed").glob("*.json")):
            res = json.loads(f.read_text())
            tag = res.get("perturb", "?")
            if tag not in ("thin", "planted_off"):
                tag = f"{tag}_p{res.get('p')}"
            if res.get("channel"):  # channel intervention (D29)
                tag = f"{tag}_ch{res['channel']}"
            sources.append((f"{tag}_seed{res.get('seed', 0)}", res))
        n_before = len(rows)
        for perturb, metrics in sources:
            rows += [run | {"perturb": perturb} | r for r in metric_rows(metrics)]
        if len(rows) == n_before:  # tuning run or unfinished run: keep its validation AUROC
            rows.append(run | {"perturb": "none", "test_set": "none"})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", type=Path, default=OUTPUT_DIR)
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "runs.csv")
    ap.add_argument(
        "--models", nargs="*", help="only these models; the other models' rows in --out stay"
    )
    args = ap.parse_args()
    df = collect(args.outputs, args.models)
    write_table(df, args.out, args.models)
    print(f"{df['run'].nunique() if len(df) else 0} runs, {len(df)} rows -> {args.out}")


if __name__ == "__main__":
    main()
