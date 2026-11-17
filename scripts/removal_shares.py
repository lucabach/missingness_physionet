"""How much can state-dependent removal remove? Evidence for the removal fractions p (Q4, D17).

    python scripts/removal_shares.py            # -> results/removal_shares.csv

For each test set (sepsis after the cut, D15, and mortality): the share of observed values that are
state-dependent candidates (D5), the share of patients with enough candidates for each p, the
fraction actually removed by ``paired_removal`` (it caps at the candidates), and the share of
vital signs among removed values: state-dependent, matched random (D16) and unmatched random.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from missingness.config import load_config
from missingness.experiment import load_data
from missingness.paths import CONFIG_DIR, REPO_ROOT
from missingness.perturb.removal import paired_removal, state_dependent_candidates

P_LEVELS = [0.05, 0.10, 0.15, 0.20, 0.30]
# Bedside vital signs (charted often); everything else is a lab value or chart item.
VITALS = {"HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2"}  # PhysioNet 2019
VITALS |= {"DiasABP", "SysABP", "NIDiasABP", "NIMAP", "NISysABP", "RespRate"}  # 2012 extras
EXPERIMENTS = {"sepsis": "sepsis_gru.yaml", "mortality": "mortality_gru.yaml"}


def vital_share(orig, pert, is_vital: np.ndarray) -> float:
    removed = sum(
        (~np.isnan(o.values) & np.isnan(p.values)).sum(axis=0)
        for o, p in zip(orig, pert, strict=True)
    )
    return float(removed[is_vital].sum() / max(removed.sum(), 1))


def main() -> None:
    rows = []
    for task, exp in EXPERIMENTS.items():
        rows += task_rows(task, load_config(CONFIG_DIR / "experiment" / exp))
    out = REPO_ROOT / "results" / "removal_shares.csv"
    pd.DataFrame(rows).round(4).to_csv(out, index=False)
    print(f"-> {out}")


def task_rows(task: str, cfg: dict) -> list[dict]:
    is_vital = np.array([v in VITALS for v in cfg["data"]["variables"]])
    sets = {k: v for k, v in load_data(cfg).items() if k.startswith("test")}
    rows = []
    for name, recs in sets.items():
        rng = np.random.default_rng(0)
        n_cand = np.array([len(state_dependent_candidates(r, rng)) for r in recs])
        n_obs = np.array([r.n_observed for r in recs])
        row = {
            "task": task,
            "test_set": name,
            "patients": len(recs),
            "candidate_share": n_cand.sum() / n_obs.sum(),
        }
        for p in P_LEVELS:
            enough = n_cand >= np.round(p * n_obs)
            rand_m, state, achieved = paired_removal(recs, p, seed=0, match_variables=True)
            rand_u, _, _ = paired_removal(recs, p, seed=0, match_variables=False)
            row |= {
                f"p{p}_patients_with_enough": enough.mean(),
                f"p{p}_achieved": achieved,
                f"p{p}_vital_share_state": vital_share(recs, state, is_vital),
                f"p{p}_vital_share_random_matched": vital_share(recs, rand_m, is_vital),
                f"p{p}_vital_share_random_unmatched": vital_share(recs, rand_u, is_vital),
            }
        rows.append(row)
        print(pd.Series(row).to_string(), "\n")
    return rows


if __name__ == "__main__":
    main()
