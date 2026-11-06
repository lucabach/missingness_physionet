"""Shared plumbing for train.py and evaluate.py: load data, split, predict, score."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from missingness.data.dataset import TimeSeriesDataset, collate
from missingness.data.preprocessing import Normalizer
from missingness.data.records import PatientRecord, load_processed
from missingness.data.splits import load_splits, make_splits, select
from missingness.data.synthetic import make_synthetic
from missingness.eval.metrics import bootstrap_ci, pre_onset_mask


def load_data(cfg: dict) -> dict[str, list[PatientRecord]]:
    """Return named record sets: train, val, and one or more test sets.

    PhysioNet 2019: train/val come from ``data.train_source``; test sets are
    ``test_hospital1`` and ``test_hospital2`` (the test split of each hospital).
    PhysioNet 2012 and synthetic data: a single ``test`` set.
    """
    data_cfg = cfg["data"]
    seed = cfg.get("split_seed", 0)
    if data_cfg["name"] == "synthetic":
        records = make_synthetic(task=data_cfg["task"], seed=seed, **data_cfg.get("kwargs", {}))
        splits = make_splits(records, seed=seed)
    else:
        records = load_processed(data_cfg)
        splits = load_splits(data_cfg["name"], seed)

    train_source = data_cfg.get("train_source")
    sets = {
        "train": select(records, splits["train"], train_source),
        "val": select(records, splits["val"], train_source),
    }
    sources = data_cfg.get("test_sources")
    if sources:
        for s in sources:
            sets[f"test_{s}"] = select(records, splits["test"], s)
    else:
        sets["test"] = select(records, splits["test"])
    return sets


def make_loader(
    records: list[PatientRecord], norm: Normalizer, task: str, batch_size: int, shuffle: bool
) -> DataLoader:
    ds = TimeSeriesDataset(records, norm, task)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)


def to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


@torch.no_grad()
def predict(
    model: nn.Module,
    records: list[PatientRecord],
    norm: Normalizer,
    task: str,
    device: torch.device,
    batch_size: int = 128,
) -> pd.DataFrame:
    """One row per labelled step: patient_id, hour, label, score (probability)."""
    model.eval()
    rows = []
    for batch in make_loader(records, norm, task, batch_size, shuffle=False):
        batch = to_device(batch, device)
        prob = torch.sigmoid(model(batch)).cpu().numpy()
        mask = batch["label_mask"].cpu().numpy()
        labels = batch["labels"].cpu().numpy()
        hours = batch["hours"].cpu().numpy()
        for i, pid in enumerate(batch["patient_id"]):
            sel = mask[i]
            rows.append(
                pd.DataFrame(
                    {
                        "patient_id": pid,
                        "hour": hours[i, sel],
                        "label": labels[i, sel],
                        "score": prob[i, sel],
                    }
                )
            )
    return pd.concat(rows, ignore_index=True)


def score_predictions(pred: pd.DataFrame, task: str, n_boot: int = 1000) -> dict:
    """AUROC/AUPRC with patient-bootstrap CIs; for sepsis also on pre-onset hours."""
    out = {}
    subsets = {"all": np.ones(len(pred), dtype=bool)}
    if task == "hourly":
        subsets["pre_onset"] = pre_onset_mask(
            pred["patient_id"].to_numpy(), pred["hour"].to_numpy(), pred["label"].to_numpy()
        )
    for name, keep in subsets.items():
        p = pred[keep]
        for metric in ("auroc", "auprc"):
            out[f"{name}/{metric}"] = bootstrap_ci(
                p["label"].to_numpy(),
                p["score"].to_numpy(),
                p["patient_id"].to_numpy(),
                metric=metric,
                n_boot=n_boot,
            )
    return out
