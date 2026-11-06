"""Tiny fake ICU data for tests and smoke runs. Never use it for results.

Sicker patients get measured more often, so the measurement pattern is informative,
which is exactly the effect the project studies.
"""

from __future__ import annotations

import numpy as np

from missingness.data.records import PatientRecord


def make_synthetic(
    n_patients: int = 60,
    n_vars: int = 5,
    n_static: int = 2,
    task: str = "hourly",
    seed: int = 0,
    sources: tuple[str, ...] = ("hospital1", "hospital2"),
) -> list[PatientRecord]:
    rng = np.random.default_rng(seed)
    records = []
    for i in range(n_patients):
        T = int(rng.integers(12, 40)) if task == "hourly" else 48
        sick = rng.random() < 0.3
        source = sources[i % len(sources)]
        base_rate = 0.5 if source == sources[0] else 0.3
        rate = np.clip(base_rate + (0.3 if sick else 0.0), 0, 0.95)
        values = rng.normal(1.0 if sick else 0.0, 1.0, size=(T, n_vars))
        values[rng.random((T, n_vars)) > rate] = np.nan
        if task == "hourly":
            labels = np.zeros(T)
            if sick:
                labels[int(T * 0.6) :] = 1
        else:
            labels = np.array([float(sick)])
        records.append(
            PatientRecord(
                patient_id=f"syn{i:04d}",
                source=source,
                hours=np.arange(T, dtype=float),
                values=values,
                static=rng.normal(size=n_static),
                labels=labels,
            )
        )
    return records
