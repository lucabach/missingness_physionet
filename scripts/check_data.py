"""Sanity checks on the processed datasets (week 1). Prints a table to compare with the
expected numbers in data/README.md.

    python scripts/check_data.py physionet2019 physionet2012
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import yaml

from missingness.paths import CONFIG_DIR, PROCESSED_DIR


def check(name: str) -> None:
    cfg = yaml.safe_load((CONFIG_DIR / "data" / f"{name}.yaml").read_text())
    df = pd.read_parquet(PROCESSED_DIR / name / "timeseries.parquet")
    print(f"\n=== {name}: {df['patient_id'].nunique()} patients, {len(df)} rows ===")
    missing_cols = [c for c in cfg["variables"] + cfg["static"] if c not in df.columns]
    print(f"missing columns: {missing_cols or 'none'}")

    per_pat = df.groupby("patient_id").agg(
        source=("source", "first"),
        T=("hour", "size"),
        pos=("label", "max"),
        pos_hours=("label", "sum"),
        dup=("hour", lambda h: h.duplicated().sum()),
    )
    by = per_pat.groupby("source").agg(
        patients=("T", "size"),
        positive_patients=("pos", "mean"),
        median_T=("T", "median"),
        min_T=("T", "min"),
        max_T=("T", "max"),
    )
    if cfg["task"] == "hourly":
        hours = df.groupby("source")["label"].mean().rename("positive_hours")
        by = by.join(hours)
    print(by.round(4).to_string())

    labels = set(np.unique(df["label"].dropna()))
    print(f"label values: {sorted(labels)}  (NaN labels: {df['label'].isna().sum()})")
    print(f"patients with duplicate hours: {(per_pat['dup'] > 0).sum()}")
    obs = df[cfg["variables"]].notna()
    print(f"observed share of all cells: {obs.to_numpy().mean():.3f}")
    never = [v for v in cfg["variables"] if not obs[v].any()]
    print(f"variables never observed: {never or 'none'}")
    stat_nan = df.groupby("patient_id")[cfg["static"]].first().isna().mean().round(3)
    print(f"share of patients with unknown static value:\n{stat_nan.to_string()}")
    if cfg.get("max_hour") is not None:
        print(hour_cut_table(df, cfg["max_hour"]).to_string())


def hour_cut_table(df: pd.DataFrame, max_hour: int) -> pd.DataFrame:
    """Effect of the cut at ``max_hour`` per hospital (D15).

    Expected for max_hour = 59 (Matteo's check, 13 Nov): records past the cut, septic / not
    septic: 644 / 8 (hospital 1), 387 / 224 (hospital 2); septic patients with a positive hour
    up to the cut: 1,222 of 1,790 and 797 of 1,142; positive share of hours after the cut
    1.51 % and 1.01 %; share of hours kept 94.8 % and 94.6 %.
    """
    per = df.groupby("patient_id").agg(
        source=("source", "first"), last=("hour", "max"), septic=("label", "max")
    )
    kept = df[df["hour"] <= max_hour]
    per["pos_before_cut"] = kept.groupby("patient_id")["label"].max().reindex(per.index)
    rows = {}
    for src, p in per.groupby("source"):
        past = p[p["last"] > max_hour]
        k = kept[kept["source"] == src]
        rows[src] = {
            f"past_h{max_hour}_septic": int((past["septic"] > 0).sum()),
            f"past_h{max_hour}_not_septic": int((past["septic"] == 0).sum()),
            "septic_patients": int((p["septic"] > 0).sum()),
            "septic_with_positive_hour_kept": int((p["pos_before_cut"] > 0).sum()),
            "records_dropped_whole": int(p["pos_before_cut"].isna().sum()),
            "positive_hours_after_cut_%": round(100 * k["label"].mean(), 2),
            "hours_kept_%": round(100 * len(k) / (df["source"] == src).sum(), 1),
        }
    print(f"\n--- cut at ICU hour {max_hour} (D15) ---")
    return pd.DataFrame(rows).astype(object)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="+")
    for name in ap.parse_args().datasets:
        check(name)


if __name__ == "__main__":
    main()
