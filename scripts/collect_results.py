"""Collect every run's metrics into one table (plan, rule 4: figures come from this file only).

    python scripts/collect_results.py                    # -> results/runs.csv
    python scripts/collect_results.py --out results/runs_mortality.csv --outputs outputs/

One row per run, test set, subset and metric. Clean test sets come from
``outputs/<run>/metrics.json``; perturbed ones from ``outputs/<run>/perturbed/*.json``
(column ``perturb``: ``none``, ``random_p0.1``, ``state_p0.1``, ``thin`` ...).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from missingness.paths import OUTPUT_DIR, REPO_ROOT

RUN_KEYS = {
    "dataset": ("data", "name"),
    "model": ("model", "name"),
    "inputs": ("inputs",),
    "seed": ("seed",),
    "git_commit": ("git_commit",),
}


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


def collect(outputs: Path) -> pd.DataFrame:
    rows = []
    for cfg_path in sorted(outputs.glob("*/config.yaml")):
        run_dir = cfg_path.parent
        cfg = yaml.safe_load(cfg_path.read_text())
        run = {"run": run_dir.name} | {k: _get(cfg, p) for k, p in RUN_KEYS.items()}
        sources = []
        if (run_dir / "metrics.json").exists():
            sources.append(("none", json.loads((run_dir / "metrics.json").read_text())))
        for f in sorted((run_dir / "perturbed").glob("*.json")):
            res = json.loads(f.read_text())
            tag = res.get("perturb", "?")
            if tag not in ("thin", "planted_off"):
                tag = f"{tag}_p{res.get('p')}"
            sources.append((f"{tag}_seed{res.get('seed', 0)}", res))
        for perturb, metrics in sources:
            rows += [run | {"perturb": perturb} | r for r in metric_rows(metrics)]
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", type=Path, default=OUTPUT_DIR)
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "runs.csv")
    args = ap.parse_args()
    df = collect(args.outputs)
    args.out.parent.mkdir(exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"{df['run'].nunique() if len(df) else 0} runs, {len(df)} rows -> {args.out}")


if __name__ == "__main__":
    main()
