"""Re-evaluate a trained run on perturbed test sets (no retraining).

    python -m missingness.evaluate outputs/<run> --perturb random --p 0.3
    python -m missingness.evaluate outputs/<run> --perturb state --p 0.3
    python -m missingness.evaluate outputs/<run> --perturb thin      # sepsis only

``random`` and ``state`` remove the same number of values per patient (paired_removal).
``thin`` thins test_hospital1 to the measurement rates of test_hospital2.
Results are written to outputs/<run>/perturbed/<perturb>[_p<p>]_seed<seed>.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from missingness.data.preprocessing import Normalizer
from missingness.experiment import load_data, predict, score_predictions
from missingness.models import build_model
from missingness.perturb import check_only_removed, measurement_rates, paired_removal, thin_to_rates
from missingness.utils import get_device


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--perturb", choices=["random", "state", "thin"], required=True)
    ap.add_argument("--p", type=float, default=0.3, help="fraction removed (random/state)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = yaml.safe_load((args.run_dir / "config.yaml").read_text())
    task = cfg["data"]["task"]
    device = get_device()
    norm = Normalizer.from_dict(json.loads((args.run_dir / "normalizer.json").read_text()))
    model = build_model(cfg["model"], len(norm.mean), len(norm.static_mean), cfg["inputs"])
    model.load_state_dict(torch.load(args.run_dir / "model.pt", map_location=device))
    model.to(device)

    sets = {k: v for k, v in load_data(cfg).items() if k.startswith("test")}
    results = {"perturb": args.perturb, "p": args.p, "seed": args.seed}
    if args.perturb == "thin":
        src, tgt = sets["test_hospital1"], sets["test_hospital2"]
        thinned = thin_to_rates(src, measurement_rates(tgt), seed=args.seed)
        check_only_removed(src, thinned)
        sets = {"test_hospital1_thinned": thinned}
    else:
        new = {}
        for name, records in sets.items():
            rand, state, achieved = paired_removal(records, args.p, seed=args.seed)
            chosen = rand if args.perturb == "random" else state
            check_only_removed(records, chosen)
            new[name] = chosen
            results[f"{name}/achieved_fraction"] = achieved
        sets = new

    for name, records in sets.items():
        pred = predict(model, records, norm, task, device)
        results[name] = score_predictions(pred, task, n_boot=cfg["eval"]["n_boot"])

    tag = args.perturb if args.perturb == "thin" else f"{args.perturb}_p{args.p}"
    out = args.run_dir / "perturbed" / f"{tag}_seed{args.seed}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"done -> {out}")


if __name__ == "__main__":
    main()
