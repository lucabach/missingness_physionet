"""Pick the tuned configuration of one model and task from the tuning sweep (plan section 4).

    python scripts/pick_config.py mortality gru                 # show the 8 runs, best first
    python scripts/pick_config.py mortality gru --write         # freeze it in the configs
    python scripts/pick_config.py mortality gru --inputs values # the plan's check (GRU)

Reads ``outputs/tune_<task>_<model>_<inputs>_*`` (made by ``slurm/submit_tuning.sh``) and
ranks them by validation AUROC; the test split is never read. ``--write`` puts the best
configuration of the ``all`` sweep as a ``tuned`` block into
``configs/experiment/<task>_<model>.yaml`` (for sepsis also into the 25-variable config), so
every input variant and seed of that model uses it. Commit the changed configs.

With ``--inputs values`` the script also prints the values model's validation AUROC under the
configuration chosen for ``all``. If the best values configuration beats it by more than
0.01, tune per variant (plan section 4, D20).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from pathlib import Path

import yaml

from missingness.paths import CONFIG_DIR, OUTPUT_DIR

KEYS = {"model.hidden_size": ("model", "hidden_size"), "train.lr": ("train", "lr"),
        "train.weight_decay": ("train", "weight_decay")}  # fmt: skip


def sweep(outputs: Path, task: str, model: str, inputs: str) -> list[dict]:
    rows = []
    for run in sorted(outputs.glob(f"tune_{task}_{model}_{inputs}_*")):
        metrics = run / "metrics.json"
        if not metrics.exists():
            continue
        cfg = yaml.safe_load((run / "config.yaml").read_text())
        row = {k: cfg[a][b] for k, (a, b) in KEYS.items()}
        rows.append(
            row | {"run": run.name, "val_auroc": json.loads(metrics.read_text())["val_auroc"]}
        )
    return sorted(rows, key=lambda r: -r["val_auroc"])


def _fmt(key: str, value) -> str:
    return str(int(value)) if key == "model.hidden_size" else f"{float(value):.1e}"


def write_tuned(path: Path, best: dict, n: int) -> None:
    """Replace (or add) the ``tuned`` block at the end of an experiment config, keeping comments."""
    text = path.read_text()
    text = re.sub(r"\n# Chosen by the tuning sweep.*", "", text, flags=re.S).rstrip("\n")
    block = [
        f"# Chosen by the tuning sweep ({dt.date.today()}, scripts/pick_config.py): best",
        f"# validation AUROC {best['val_auroc']:.4f} of {n} configurations, `all` inputs, seed 0.",
        "tuned:",
    ] + [f"  {k}: {_fmt(k, best[k])}" for k in KEYS]
    path.write_text(text + "\n" + "\n".join(block) + "\n")
    print(f"wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("task", choices=["sepsis", "mortality"])
    ap.add_argument("model")
    ap.add_argument("--inputs", default="all")
    ap.add_argument("--outputs", type=Path, default=OUTPUT_DIR)
    ap.add_argument("--write", action="store_true", help="freeze the best `all` configuration")
    args = ap.parse_args()

    rows = sweep(args.outputs, args.task, args.model, args.inputs)
    if not rows:
        raise SystemExit(f"no finished tuning runs for {args.task} {args.model} {args.inputs}")
    for r in rows:
        print(f"{r['val_auroc']:.4f}  " + "  ".join(f"{k}={_fmt(k, r[k])}" for k in KEYS))
    best = rows[0]

    if args.inputs != "all":
        chosen = sweep(args.outputs, args.task, args.model, "all")
        if chosen:
            same = [r for r in rows if all(r[k] == chosen[0][k] for k in KEYS)]
            if same:
                gap = best["val_auroc"] - same[0]["val_auroc"]
                verdict = "tune per variant" if gap > 0.01 else "keep the `all` configuration"
                print(f"{args.inputs}: best {best['val_auroc']:.4f}, with the `all` choice "
                      f"{same[0]['val_auroc']:.4f}, gap {gap:+.4f} -> {verdict}")  # fmt: skip
        return
    if args.write:
        if len(rows) < 8:
            raise SystemExit(f"only {len(rows)} of 8 tuning runs finished; not writing")
        names = [f"{args.task}_{args.model}.yaml"]
        if args.task == "sepsis":
            names.append(f"sepsis_comparable_{args.model}.yaml")
        for name in names:
            write_tuned(CONFIG_DIR / "experiment" / name, best, len(rows))


if __name__ == "__main__":
    main()
