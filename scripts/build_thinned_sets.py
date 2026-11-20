"""Build the thinned sepsis test set: hospital-1 test patients thinned to hospital 2's
measurement rates (per variable, removal only, same patients). Week-2 deliverable.

    python scripts/build_thinned_sets.py --seeds 0 1 2

Checks the invariants (same patients, hours and labels; only removed values), prints the
achieved vs target rate per variable, and saves
data/processed/physionet2019/thinned_test_hospital1_seed<k>.parquet (not committed) plus a
small summary table results/thinning_rates.csv.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from missingness.config import load_config
from missingness.data.records import records_to_frame
from missingness.experiment import load_data
from missingness.paths import CONFIG_DIR, PROCESSED_DIR, REPO_ROOT
from missingness.perturb import check_only_removed, measurement_rates, thin_to_rates


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    args = ap.parse_args()
    cfg = load_config(CONFIG_DIR / "experiment" / "sepsis_gru.yaml")
    variables, static = cfg["data"]["variables"], cfg["data"]["static"]
    sets = load_data(cfg)
    src, tgt = sets["test_hospital1"], sets["test_hospital2"]
    r_src, r_tgt = measurement_rates(src), measurement_rates(tgt)

    table = pd.DataFrame({"variable": variables, "rate_hospital1": r_src, "rate_hospital2": r_tgt})
    for seed in args.seeds:
        thinned = thin_to_rates(src, r_tgt, seed=seed)
        check_only_removed(src, thinned)
        table[f"rate_thinned_seed{seed}"] = measurement_rates(thinned)
        removed = 1 - sum(r.n_observed for r in thinned) / sum(r.n_observed for r in src)
        out = PROCESSED_DIR / "physionet2019" / f"thinned_test_hospital1_seed{seed}.parquet"
        records_to_frame(thinned, variables, static).to_parquet(out, index=False)
        print(f"seed {seed}: {len(thinned)} patients, {removed:.1%} of values removed -> {out}")

    table["matched"] = table["rate_hospital1"] >= table["rate_hospital2"]
    s0 = f"rate_thinned_seed{args.seeds[0]}"
    table["abs_error_vs_target"] = np.where(
        table["matched"], (table[s0] - table["rate_hospital2"]).abs(), np.nan
    )
    REPO_ROOT.joinpath("results").mkdir(exist_ok=True)
    table.round(5).to_csv(REPO_ROOT / "results" / "thinning_rates.csv", index=False)
    m = table[table["matched"]]
    print(
        f"matched variables: {len(m)}/{len(table)}; "
        f"max |thinned - target| rate error: {m['abs_error_vs_target'].max():.4f}"
    )
    print(table[~table["matched"]][["variable", "rate_hospital1", "rate_hospital2"]].to_string())


if __name__ == "__main__":
    main()
