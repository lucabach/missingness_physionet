"""The one data format everything else uses.

A ``PatientRecord`` holds one ICU stay on an hourly grid. Missing measurements are NaN in
``values``; the mask and time-since-last-measurement are *derived* from those NaNs (see
``preprocessing.py``) and never stored. That way, removing a measurement (setting it to NaN)
automatically keeps values, mask and time gaps consistent.

On disk, one dataset is one Parquet table, ``data/processed/<dataset>/timeseries.parquet``:

    patient_id | source | hour | <static columns> | <variable columns> | label

* ``source``: ``hospital1`` / ``hospital2`` for PhysioNet 2019, ``p2012`` for PhysioNet 2012.
* ``label``: hourly sepsis label (PhysioNet 2019) or the stay-level in-hospital death label
  repeated on every row (PhysioNet 2012).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd

from missingness.paths import PROCESSED_DIR


@dataclass
class PatientRecord:
    patient_id: str
    source: str  # "hospital1", "hospital2" (PhysioNet 2019) or "p2012"
    hours: np.ndarray  # (T,) hours since ICU admission
    values: np.ndarray  # (T, D) raw values, NaN = not measured
    static: np.ndarray  # (S,) static covariates (age, sex, ...), NaN = unknown
    labels: np.ndarray  # (T,) hourly labels, or (1,) for a stay-level label
    meta: dict = field(default_factory=dict)

    @property
    def n_observed(self) -> int:
        return int((~np.isnan(self.values)).sum())

    @property
    def stay_label(self) -> int:
        """1 if the stay is ever positive (septic patient / died)."""
        return int(np.nanmax(self.labels) > 0)

    def with_values(self, values: np.ndarray) -> PatientRecord:
        """Copy of this record with new values (used by every perturbation)."""
        assert values.shape == self.values.shape
        return replace(self, values=values)


def records_from_frame(
    df: pd.DataFrame,
    variables: list[str],
    static: list[str],
    stay_label: bool,
) -> list[PatientRecord]:
    """Turn the processed long table into a list of PatientRecords (sorted by patient_id)."""
    records = []
    for pid, g in df.sort_values(["patient_id", "hour"]).groupby("patient_id", sort=True):
        labels = g["label"].to_numpy(dtype=float)
        records.append(
            PatientRecord(
                patient_id=str(pid),
                source=str(g["source"].iloc[0]),
                hours=g["hour"].to_numpy(dtype=float),
                values=g[variables].to_numpy(dtype=float),
                static=g[static].iloc[0].to_numpy(dtype=float),
                labels=labels[-1:] if stay_label else labels,
            )
        )
    return records


def save_processed(df: pd.DataFrame, dataset: str, root: Path = PROCESSED_DIR) -> Path:
    path = root / dataset / "timeseries.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def load_processed(dataset_cfg: dict, root: Path = PROCESSED_DIR) -> list[PatientRecord]:
    """Load all records of a dataset described by a ``configs/data/*.yaml`` config."""
    path = root / dataset_cfg["name"] / "timeseries.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run scripts/download_data.sh and "
            f"`python scripts/prepare_data.py {dataset_cfg['name']}` first (see data/README.md)."
        )
    df = pd.read_parquet(path)
    return records_from_frame(
        df,
        variables=dataset_cfg["variables"],
        static=dataset_cfg["static"],
        stay_label=dataset_cfg["task"] == "stay",
    )
