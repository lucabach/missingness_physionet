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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="+")
    for name in ap.parse_args().datasets:
        check(name)


if __name__ == "__main__":
    main()
