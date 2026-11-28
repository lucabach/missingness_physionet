"""Leak audit for the RQ2 redesign (D29, proposed): how much pattern the values-only input shows.

    python scripts/leak_audit.py                    # -> results/leak_audit/leak_audit.csv
    python scripts/leak_audit.py --coverage         # -> results/leak_audit/snapshot_coverage.csv
    python scripts/leak_audit.py --print-commands   # training and evaluation commands of D29

No training. For each task (mortality, sepsis) and each test set:

- ``forward_fill``: share of the true measurements that show up as a change of the
  forward-filled x (``recall``). 1.0 would mean a values-only model sees the whole mask.
- ``snapshot``: the same in the common-schedule view, plus the share of empty bins
  (``empty_bin_share``), the part of the pattern that is still patient-specific there.

``--coverage`` lists, per variable, the share of complete bins with a value on the training
split for several bin widths. Use it to fix the snapshot block of the data config (bin widths,
``min_coverage``) before any snapshot run; it reads no labels.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from missingness.config import load_config
from missingness.data.preprocessing import Normalizer
from missingness.data.views import (
    bin_coverage,
    feature_kwargs,
    nominal_widths,
    resolve_snapshot,
    variable_names,
)
from missingness.eval.leak import mask_recovery
from missingness.experiment import load_data
from missingness.paths import CONFIG_DIR, REPO_ROOT

TASKS = {"mortality": "mortality_gru.yaml", "sepsis": "sepsis_gru.yaml"}
WIDTHS = (1, 2, 4, 6, 8, 12, 24)
MODELS = ("gru", "grud", "odernn")
SEEDS = (0, 1, 2)


def audit(cfg: dict, task: str) -> pd.DataFrame:
    sets = load_data(cfg, apply_planted=False)
    norm = Normalizer.fit(sets["train"])
    names = variable_names(cfg, len(norm.mean))
    snap = dict(cfg, features="snapshot")
    snap["data"] = dict(cfg["data"], snapshot=dict(cfg["data"]["snapshot"]))
    resolve_snapshot(snap, sets["train"])
    frames = []
    for name, records in sets.items():
        if not name.startswith("test"):
            continue
        for view, c in (("forward_fill", cfg), ("snapshot", snap)):
            t = mask_recovery(records, norm, names, feature_kwargs(c))
            frames.append(t.assign(task=task, test_set=name, view=view))
    cols = ["task", "test_set", "view"]
    out = pd.concat(frames, ignore_index=True)
    return out[cols + [c for c in out.columns if c not in cols]]


def coverage(cfg: dict, task: str) -> pd.DataFrame:
    train = load_data(cfg, apply_planted=False)["train"]
    names = variable_names(cfg, train[0].values.shape[1])
    nominal = nominal_widths(cfg["data"]["snapshot"], names)
    rows = []
    for d, name in enumerate(names):
        row = {"task": task, "variable": name, "configured_width": nominal[name]}
        row |= {f"w{w}": round(bin_coverage(train, d, w), 4) for w in WIDTHS}
        rows.append(row)
    return pd.DataFrame(rows)


def commands() -> list[str]:
    """D29 runs: decoded-pattern models, snapshot values and all models, channel evaluation."""
    cmds = []
    for task in TASKS:
        for model in MODELS:
            exp = f"configs/experiment/{task}_{model}.yaml"
            for seed in SEEDS:
                cmds.append(
                    f"python -m missingness.train {exp} inputs=pattern features=decoded "
                    f"seed={seed} --skip-done"
                )
                for inputs in ("values", "all"):
                    cmds.append(
                        f"python -m missingness.train {exp} inputs={inputs} features=snapshot "
                        f"seed={seed} --skip-done"
                    )
    cmds.append("python scripts/evaluate_runs.py --runs '*_snapshot_*'  # removal + thinning")
    cmds.append("python scripts/evaluate_runs.py --channel  # channel intervention, `all` runs")
    cmds.append("python scripts/collect_results.py && python scripts/measures.py")
    return cmds


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS))
    ap.add_argument("--experiment", type=Path, help="one experiment config instead (tests)")
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results" / "leak_audit")
    ap.add_argument("--coverage", action="store_true", help="bin coverage per width instead")
    ap.add_argument("--print-commands", action="store_true")
    args = ap.parse_args()
    if args.print_commands:
        print("\n".join(commands()))
        return
    if args.experiment:
        jobs = [(load_config(args.experiment), args.experiment.stem)]
    else:
        jobs = [(load_config(CONFIG_DIR / "experiment" / TASKS[t]), t) for t in args.tasks]
    fn, stem = (coverage, "snapshot_coverage") if args.coverage else (audit, "leak_audit")
    table = pd.concat([fn(cfg, task) for cfg, task in jobs], ignore_index=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"{stem}.csv"
    table.to_csv(out, index=False)
    with pd.option_context("display.max_rows", 200, "display.width", 160):
        if "recall" in table:
            keep = ["task", "test_set", "view", "n_measured", "recall", "empty_bin_share"]
            print(table.loc[table["variable"] == "ALL", keep].to_string(index=False))
        else:
            print(table.to_string(index=False))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
