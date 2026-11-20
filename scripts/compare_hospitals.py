"""Week-1 check: how differently do hospital 1 and hospital 2 measure? (open question Q2)

    python scripts/compare_hospitals.py

For every variable and hospital: measurement rate (observed values per patient-hour) and
median time between consecutive measurements, for all hours, for septic and non-septic
patients, and for pre-onset hours only (hours after sepsis onset carry many extra tests).
Writes results/hospital_rates.csv and results/hospital_rates.pdf.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from missingness.eval.metrics import pre_onset_mask  # noqa: E402
from missingness.paths import CONFIG_DIR, PROCESSED_DIR, REPO_ROOT  # noqa: E402

HOSPITALS = ("hospital1", "hospital2")


def rates_table(df: pd.DataFrame, variables: list[str], group: str) -> pd.DataFrame:
    rows = []
    for hosp in HOSPITALS:
        d = df[df["source"] == hosp]
        n_hours = len(d)
        for var in variables:
            obs = d.loc[d[var].notna(), ["patient_id", "hour"]]
            same = obs["patient_id"].eq(obs["patient_id"].shift())
            gaps = obs["hour"].diff()[same]
            rows.append(
                {
                    "group": group,
                    "hospital": hosp,
                    "variable": var,
                    "rate": len(obs) / max(n_hours, 1),
                    "median_gap_h": float(gaps.median()) if len(gaps) else np.nan,
                    "n_obs": len(obs),
                    "n_patient_hours": n_hours,
                    "n_patients": d["patient_id"].nunique(),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    cfg = yaml.safe_load((CONFIG_DIR / "data" / "physionet2019.yaml").read_text())
    variables = cfg["variables"]
    df = pd.read_parquet(PROCESSED_DIR / "physionet2019" / "timeseries.parquet")
    df = df.sort_values(["patient_id", "hour"]).reset_index(drop=True)
    septic = df.groupby("patient_id")["label"].transform("max") > 0
    pre = pre_onset_mask(df["patient_id"].to_numpy(), df["hour"].to_numpy(), df["label"].to_numpy())

    table = pd.concat(
        [
            rates_table(df, variables, "all"),
            rates_table(df[septic], variables, "septic"),
            rates_table(df[~septic], variables, "non_septic"),
            rates_table(df[pre], variables, "pre_onset"),
        ],
        ignore_index=True,
    )
    out_dir = REPO_ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    table.to_csv(out_dir / "hospital_rates.csv", index=False)

    # Summary on all hours
    wide = table[table["group"] == "all"].pivot(index="variable", columns="hospital", values="rate")
    eps = 1e-5  # a variable never measured in one hospital (EtCO2 in hospital 1) gives rate 0
    wide["log2_ratio"] = np.log2((wide["hospital1"] + eps) / (wide["hospital2"] + eps))
    wide = wide.loc[variables].sort_values("log2_ratio")
    for group in ("all", "pre_onset", "septic", "non_septic"):
        g = table[table["group"] == group]
        tot = g.groupby("hospital")[["n_obs", "n_patient_hours"]].sum()
        per_hour = (tot["n_obs"] / tot["n_patient_hours"]).round(2).to_dict()
        print(f"{group:11s} observed values per patient-hour: {per_hour}")
    more1 = int((wide["log2_ratio"] > 0).sum())
    print(
        f"hospital 1 measures more often for {more1}/{len(wide)} variables, "
        f"hospital 2 for {len(wide) - more1}"
    )
    print("\nLargest differences (log2 rate ratio hospital1/hospital2):")
    print(wide.round(3).iloc[np.r_[0:5, -5:0]].to_string())

    fig, ax = plt.subplots(figsize=(6, 8))
    y = np.arange(len(wide))
    ax.scatter(wide["hospital1"].clip(lower=eps), y, label="hospital 1", s=18)
    ax.scatter(wide["hospital2"].clip(lower=eps), y, label="hospital 2", s=18, marker="s")
    ax.set_yticks(y, wide.index, fontsize=7)
    ax.set_xscale("log")
    ax.set_xlabel("observed values per patient-hour (log scale; 0 drawn at 1e-5)")
    ax.set_title(
        "PhysioNet 2019: measurement rate per variable\n(sorted by hospital 1 / hospital 2 ratio)"
    )
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "hospital_rates.pdf")
    print(f"\n-> {out_dir / 'hospital_rates.csv'}, {out_dir / 'hospital_rates.pdf'}")


if __name__ == "__main__":
    main()
