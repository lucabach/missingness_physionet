"""Writing the committed result tables (``results/runs.csv``, ``results/measures.csv``).

The runs of different models may live on different machines (week 2: GRU on Matteo's laptop;
week 3: GRU-D and ODE-RNN elsewhere). With ``models`` given, a table is updated for those
models only and keeps the rows of every other model, so nobody overwrites another owner's
numbers by running a script on an ``outputs/`` folder that lacks their runs.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_table(new: pd.DataFrame, out: Path, models: list[str] | None = None) -> pd.DataFrame:
    """Write ``new`` to ``out``; with ``models``, replace only those models' rows in ``out``."""
    if models and out.exists():
        old = pd.read_csv(out)
        if "model" in old:
            new = pd.concat([old[~old["model"].isin(models)], new], ignore_index=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    new.to_csv(out, index=False)
    return new
