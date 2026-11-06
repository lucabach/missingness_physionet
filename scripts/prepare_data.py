"""Convert raw PhysioNet files to the processed format and (optionally) create splits.

    python scripts/prepare_data.py physionet2019 physionet2012
    python scripts/prepare_data.py physionet2019 --make-splits   # only once, then commit!

The processed table goes to data/processed/<dataset>/timeseries.parquet (not committed).
Splits go to splits/<dataset>_seed0.json and ARE committed; if the file already exists it
is never overwritten, so everybody keeps using the same patients.
"""

from __future__ import annotations

import argparse

import yaml

from missingness.data import physionet2012, physionet2019
from missingness.data.records import load_processed, save_processed
from missingness.data.splits import make_splits, save_splits, split_path
from missingness.paths import CONFIG_DIR, RAW_DIR

LOADERS = {"physionet2019": physionet2019.load_raw, "physionet2012": physionet2012.load_raw}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="+", choices=sorted(LOADERS))
    ap.add_argument("--make-splits", action="store_true")
    ap.add_argument("--split-seed", type=int, default=0)
    args = ap.parse_args()

    for name in args.datasets:
        cfg = yaml.safe_load((CONFIG_DIR / "data" / f"{name}.yaml").read_text())
        df = LOADERS[name](RAW_DIR / name, cfg["variables"], cfg["static"])
        path = save_processed(df, name)
        n_pat = df["patient_id"].nunique()
        print(f"{name}: {n_pat} patients, {len(df)} rows -> {path}")
        print(df.groupby("source")["patient_id"].nunique().to_string())

        if args.make_splits:
            if split_path(name, args.split_seed).exists():
                print(f"{split_path(name, args.split_seed)} exists, not overwriting")
                continue
            records = load_processed(cfg)
            out = save_splits(make_splits(records, seed=args.split_seed), name, args.split_seed)
            print(f"splits -> {out}  (commit this file)")


if __name__ == "__main__":
    main()
