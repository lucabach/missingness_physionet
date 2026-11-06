"""Patient-level train/val/test splits, stratified by outcome (and by hospital for 2019).

Splits are written to ``splits/<dataset>_seed<k>.json`` and committed, so every teammate
trains and evaluates on exactly the same patients. Never re-split silently: if a split must
change, make a new seed file and log it in docs/decisions.md.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from missingness.data.records import PatientRecord
from missingness.paths import SPLITS_DIR


def make_splits(
    records: list[PatientRecord],
    fractions: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 0,
) -> dict[str, list[str]]:
    rng = np.random.default_rng(seed)
    out: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    strata: dict[tuple[str, int], list[str]] = {}
    for r in records:
        strata.setdefault((r.source, r.stay_label), []).append(r.patient_id)
    for key in sorted(strata):
        ids = np.array(sorted(strata[key]))
        rng.shuffle(ids)
        n_train = int(round(fractions[0] * len(ids)))
        n_val = int(round(fractions[1] * len(ids)))
        out["train"] += ids[:n_train].tolist()
        out["val"] += ids[n_train : n_train + n_val].tolist()
        out["test"] += ids[n_train + n_val :].tolist()
    return {k: sorted(v) for k, v in out.items()}


def split_path(dataset: str, seed: int) -> Path:
    return SPLITS_DIR / f"{dataset}_seed{seed}.json"


def save_splits(splits: dict[str, list[str]], dataset: str, seed: int) -> Path:
    path = split_path(dataset, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(splits, indent=0))
    return path


def load_splits(dataset: str, seed: int) -> dict[str, list[str]]:
    return json.loads(split_path(dataset, seed).read_text())


def select(
    records: list[PatientRecord], ids: list[str], source: str | None = None
) -> list[PatientRecord]:
    """Records whose id is in ``ids`` (optionally only one hospital), in a stable order."""
    keep = set(ids)
    return [r for r in records if r.patient_id in keep and (source is None or r.source == source)]
