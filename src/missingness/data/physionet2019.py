"""PhysioNet/CinC Challenge 2019 (early sepsis prediction), Reyna et al. 2020.

Raw format: one pipe-separated ``.psv`` file per patient with one row per ICU hour, 40
variable columns (NaN = not measured) and ``SepsisLabel`` (1 from 6 h before clinical onset).
The challenge releases two hospital systems as ``training_setA`` and ``training_setB``.
We call them **hospital 1** and **hospital 2** everywhere (never "A"/"B", see CLAUDE.md).

NOTE: written against the documented file format; the week-1 owner checks it on the real
files (row counts, label prevalence: about 7% of patients septic, 2% of hours positive).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from tqdm import tqdm

SOURCE_DIRS = {"hospital1": "training_setA", "hospital2": "training_setB"}


def find_raw_dir(raw_root: Path) -> Path:
    """Locate the folder that contains training_setA/ and training_setB/ after download."""
    hits = sorted(raw_root.rglob("training_setA"))
    if not hits:
        raise FileNotFoundError(f"No training_setA/ under {raw_root}. See data/README.md.")
    return hits[0].parent


def load_raw(raw_root: Path, variables: list[str], static: list[str]) -> pd.DataFrame:
    """Read every .psv file into the processed long table (see records.py)."""
    base = find_raw_dir(raw_root)
    frames = []
    for source, folder in SOURCE_DIRS.items():
        files = sorted((base / folder).glob("*.psv"))
        for f in tqdm(files, desc=f"{source} ({folder})"):
            df = pd.read_csv(f, sep="|")
            out = pd.DataFrame(
                {
                    "patient_id": f.stem,
                    "source": source,
                    "hour": df["ICULOS"].to_numpy(),
                }
            )
            for col in static:
                out[col] = df[col].iloc[0]
            for col in variables:
                out[col] = df[col].to_numpy()
            out["label"] = df["SepsisLabel"].to_numpy()
            frames.append(out)
    return pd.concat(frames, ignore_index=True)
