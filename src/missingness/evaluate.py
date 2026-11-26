"""Re-evaluate a trained run on perturbed test sets (no retraining).

    python -m missingness.evaluate outputs/<run> --perturb random --p 0.10
    python -m missingness.evaluate outputs/<run> --perturb random state --p 0.05 0.10 0.15
    python -m missingness.evaluate outputs/<run> --perturb random_hours state_hours --p 0.2 0.25 0.3
    python -m missingness.evaluate outputs/<run> --perturb thin --seed 0 1 2   # sepsis only
    python -m missingness.evaluate outputs/<run> --perturb planted_off         # E5 runs only

``random`` and ``state`` remove the same number of values per patient (paired_removal).
``random_hours`` and ``state_hours`` do the same with whole busy hours as the state-dependent
candidates (D27); random removal is again matched per variable.
Random removal takes the same count per variable as state-dependent removal (D16).
``thin`` thins test_hospital1 to the measurement rates of test_hospital2.
``planted_off`` (runs trained with a ``planted`` rule, E5): the rule switched off, i.e. the
same share of the planted variable deleted for every patient.

All combinations of the given perturbations, fractions and seeds are evaluated with one load
of the data and the model. Each writes outputs/<run>/perturbed/<tag>_seed<seed>.json and the
predictions next to it, in perturbed/<tag>_seed<seed>/<test set>.parquet (the paired
measures M2 and M4 are computed from these). Existing results are skipped unless --force.
The standard evaluation of every run is ``scripts/evaluate_runs.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from missingness.data.preprocessing import Normalizer
from missingness.data.records import PatientRecord
from missingness.experiment import (
    load_data,
    predict,
    score_predictions,
    with_pattern_drop,
    with_planted,
)
from missingness.models import build_model
from missingness.perturb import (
    check_only_removed,
    measurement_rates,
    paired_removal,
    planted_off,
    thin_to_rates,
)
from missingness.perturb.planted import planted_index
from missingness.utils import get_device

PERTURBS = ("random", "state", "random_hours", "state_hours", "thin", "planted_off")
REMOVAL_PERTURBS = ("random", "state", "random_hours", "state_hours")


def result_tag(perturb: str, p: float | None, seed: int) -> str:
    """File stem of one evaluation, e.g. ``state_p0.1_seed0`` or ``thin_seed2``."""
    base = perturb if perturb in ("thin", "planted_off") else f"{perturb}_p{p}"
    return f"{base}_seed{seed}"


def perturb_sets(
    cfg: dict,
    tests: dict[str, list[PatientRecord]],
    clean: dict[str, list[PatientRecord]] | None,
    perturb: str,
    p: float | None,
    seed: int,
    cache: dict | None = None,
) -> tuple[dict[str, list[PatientRecord]], dict]:
    """Perturbed test sets for one evaluation, each checked with ``check_only_removed``.

    ``tests`` are the run's test sets as trained (planted rule on for E5 runs); ``clean`` are
    the same sets without the planted rule (only needed for ``planted_off``). ``cache`` keeps
    the paired removal of a (p, seed), so random and state-dependent share one computation.
    """
    extra: dict = {}
    if perturb == "planted_off":
        pl = cfg.get("planted")
        if not pl or clean is None:
            raise SystemExit("planted_off needs a run trained with a `planted` block (E5)")
        d = planted_index(pl, cfg["data"].get("variables", []))
        out = {
            f"{name}_rule_off": planted_off(recs, d, pl["q1"], pl["q0"], seed=seed)
            for name, recs in clean.items()
        }
        for name, recs in clean.items():
            check_only_removed(recs, out[f"{name}_rule_off"])
        return out, extra
    if perturb == "thin":
        src, tgt = tests["test_hospital1"], tests["test_hospital2"]
        thinned = thin_to_rates(src, measurement_rates(tgt), seed=seed)
        check_only_removed(src, thinned)
        return {"test_hospital1_thinned": thinned}, extra
    out = {}
    cache = {} if cache is None else cache
    whole_hours = perturb.endswith("_hours")
    for name, records in tests.items():
        key = (name, p, seed, whole_hours)
        if key not in cache:
            cache[key] = paired_removal(records, p, seed=seed, whole_hours=whole_hours)
        rand, state, achieved = cache[key]
        chosen = rand if perturb.startswith("random") else state
        check_only_removed(records, chosen)
        out[name] = chosen
        extra[f"{name}/achieved_fraction"] = achieved
    return out, extra


def evaluate_run(
    run_dir: Path, jobs: list[tuple[str, float | None, int]], force: bool = False
) -> list[Path]:
    """Evaluate one trained run under each (perturb, p, seed) in ``jobs``; return the files."""
    todo = [
        j for j in jobs if force or not (run_dir / "perturbed" / f"{result_tag(*j)}.json").exists()
    ]
    if not todo:
        return []
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    task = cfg["data"]["task"]
    device = get_device()
    norm = Normalizer.from_dict(json.loads((run_dir / "normalizer.json").read_text()))
    model = build_model(cfg["model"], len(norm.mean), len(norm.static_mean), cfg["inputs"])
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location=device, weights_only=True))
    model = with_pattern_drop(model.to(device), cfg)

    # One load. For E5 runs, ``clean`` is without the planted rule (for planted_off) and
    # ``tests`` with it, rebuilt exactly as in training.
    planted = bool(cfg.get("planted"))
    everything = load_data(cfg, apply_planted=False)
    clean = {k: v for k, v in everything.items() if k.startswith("test")}
    tests = clean
    if planted:
        tests = {k: v for k, v in with_planted(cfg, everything).items() if k.startswith("test")}

    written, cache = [], {}
    for perturb, p, seed in todo:
        sets, results = perturb_sets(
            cfg, tests, clean if planted else None, perturb, p, seed, cache
        )
        results = {"perturb": perturb, "p": p, "seed": seed} | results
        pred_dir = run_dir / "perturbed" / result_tag(perturb, p, seed)
        pred_dir.mkdir(parents=True, exist_ok=True)
        for name, records in sets.items():
            pred = predict(model, records, norm, task, device)
            pred.to_parquet(pred_dir / f"{name}.parquet", index=False)
            results[name] = score_predictions(pred, task, n_boot=cfg["eval"]["n_boot"])
        out = run_dir / "perturbed" / f"{result_tag(perturb, p, seed)}.json"
        out.write_text(json.dumps(results, indent=2))
        print(f"done -> {out}")
        written.append(out)
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--perturb", choices=PERTURBS, nargs="+", required=True)
    ap.add_argument("--p", type=float, nargs="+", default=[0.10], help="fractions removed, D17")
    ap.add_argument("--seed", type=int, nargs="+", default=[0])
    ap.add_argument("--force", action="store_true", help="redo evaluations that exist")
    args = ap.parse_args()
    jobs = []
    for perturb in args.perturb:
        ps = args.p if perturb in REMOVAL_PERTURBS else [None]
        jobs += [(perturb, p, s) for p in ps for s in args.seed]
    evaluate_run(args.run_dir, jobs, force=args.force)


if __name__ == "__main__":
    main()
