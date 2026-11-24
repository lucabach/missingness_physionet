"""The standard evaluation of every finished run: the same for every model (protocol-v1).

    python scripts/evaluate_runs.py                          # every run in outputs/
    python scripts/evaluate_runs.py --runs 'physionet2012_gru_*' --dry-run
    python scripts/evaluate_runs.py --hours        # only the whole-busy-hour removal (D27)

Per run (tuning runs are skipped; evaluations that exist are skipped):

- `all` or `values`, no planted rule: random and state-dependent removal at p 0.05, 0.10 and
  0.15, removal seed 0 (D17).
- The same runs for sepsis: also thinning of hospital 1 to hospital 2's rates, thinning seeds
  0, 1, 2 (D14).
- Planted rule (E5): rule off, seed 0.
- `pattern`: nothing (train.py scores the clean test sets).

Every model sees the same perturbed test sets (same seeds), so the paired measures in
``scripts/measures.py`` compare models on identical inputs.
"""

from __future__ import annotations

import argparse

import yaml

from missingness.evaluate import evaluate_run
from missingness.paths import OUTPUT_DIR

REMOVAL_P = (0.05, 0.10, 0.15)
HOURS_P = (0.20, 0.25, 0.30)  # whole busy hours (D27), state_hours vs random_hours
THIN_SEEDS = (0, 1, 2)


def hours_jobs(cfg: dict) -> list[tuple[str, float | None, int]]:
    """Whole-busy-hour removal (D27) for the `all` and `values` runs without a planted rule."""
    if cfg.get("planted") or cfg["inputs"] not in ("all", "values"):
        return []
    return [(m, p, 0) for p in HOURS_P for m in ("random_hours", "state_hours")]


def standard_jobs(cfg: dict) -> list[tuple[str, float | None, int]]:
    if cfg.get("planted"):
        return [("planted_off", None, 0)]
    if cfg["inputs"] not in ("all", "values"):
        return []
    jobs = [(m, p, 0) for p in REMOVAL_P for m in ("random", "state")]
    if "hospital2" in (cfg["data"].get("test_sources") or []):
        jobs += [("thin", None, s) for s in THIN_SEEDS]
    return jobs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", default="*", help="glob of run folders inside outputs/")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--hours", action="store_true", help="only whole-busy-hour removal (D27)")
    args = ap.parse_args()
    for cfg_path in sorted(OUTPUT_DIR.glob(f"{args.runs}/config.yaml")):
        run = cfg_path.parent
        if run.name.startswith("tune_") or not (run / "metrics.json").exists():
            continue
        cfg = yaml.safe_load(cfg_path.read_text())
        jobs = hours_jobs(cfg) if args.hours else standard_jobs(cfg)
        if args.dry_run:
            print(run.name, jobs)
        elif jobs:
            print(f"== {run.name}")
            evaluate_run(run, jobs)


if __name__ == "__main__":
    main()
