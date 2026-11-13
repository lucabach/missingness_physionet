"""PhysioNet/CinC Challenge 2012 (in-hospital mortality), Silva et al. 2012.

Raw format: one ``<RecordID>.txt`` file per stay with rows ``Time,Parameter,Value`` (Time is
``HH:MM`` since ICU admission, first 48 h), in folders set-a/, set-b/, set-c/, and outcome
files Outcomes-a.txt etc. with the column ``In-hospital_death``. Static descriptors (Age,
Gender, Height, ICUType, Weight) appear at time 00:00; -1 means unknown.

We put the 48 h on an hourly grid (hour 0..47), taking the mean of several measurements of
the same variable within one hour (decision D3 in docs/decisions.md). Stays without a single
time-series value (12 of 12,000) are dropped, as in Horn et al. 2020 and later benchmarks (D10).

NOTE: written against the documented file format; the week-1 owner checks it on the real
files (12,000 stays, about 14% in-hospital deaths).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

N_HOURS = 48
SETS = ("a", "b", "c")


def _read_stay(path: Path, variables: list[str], static: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    hh, mm = df["Time"].str.split(":", expand=True).astype(int).T.to_numpy()
    df["hour"] = np.minimum((hh * 60 + mm) // 60, N_HOURS - 1)

    at_zero = df[(hh == 0) & (mm == 0)]
    static_vals = {}
    for col in static:
        v = at_zero.loc[at_zero["Parameter"] == col, "Value"]
        static_vals[col] = np.nan if v.empty or v.iloc[0] < 0 else float(v.iloc[0])

    ts = df[df["Parameter"].isin(variables)]
    # Static "Weight" at 00:00 is a descriptor, not a measurement.
    ts = ts[~((ts["Parameter"] == "Weight") & (hh[ts.index] == 0) & (mm[ts.index] == 0))]
    ts = ts[ts["Value"] >= 0]  # -1 encodes "unknown"
    grid = ts.pivot_table(index="hour", columns="Parameter", values="Value", aggfunc="mean")
    grid = grid.reindex(index=range(N_HOURS), columns=variables)

    out = pd.DataFrame({"patient_id": path.stem, "source": "p2012", "hour": np.arange(N_HOURS)})
    for col in static:
        out[col] = static_vals[col]
    out[variables] = grid.to_numpy()
    return out


def load_raw(raw_root: Path, variables: list[str], static: list[str]) -> pd.DataFrame:
    frames = []
    for s in SETS:
        set_dirs = sorted(raw_root.rglob(f"set-{s}"))
        outcome_files = sorted(raw_root.rglob(f"Outcomes-{s}.txt"))
        if not set_dirs or not outcome_files:
            print(f"[physionet2012] set-{s} or Outcomes-{s}.txt not found, skipping")
            continue
        outcomes = pd.read_csv(outcome_files[0]).set_index("RecordID")["In-hospital_death"]
        for f in tqdm(sorted(set_dirs[0].glob("*.txt")), desc=f"set-{s}"):
            stay = _read_stay(f, variables, static)
            if stay[variables].notna().sum().sum() == 0:
                continue  # only static descriptors, no measurements (D10)
            stay["label"] = int(outcomes.loc[int(f.stem)])
            frames.append(stay)
    if not frames:
        raise FileNotFoundError(f"No PhysioNet 2012 files under {raw_root}. See data/README.md.")
    return pd.concat(frames, ignore_index=True)
