"""Torch dataset and padding collate function.

A batch is a dict of tensors, all padded to the longest stay in the batch:
    x, m, delta   (B, T, D)   imputed values, mask, time since last measurement
    hours         (B, T)
    static        (B, S)
    labels        (B, T)      hourly labels; for stay-level tasks only the last real step
    label_mask    (B, T)      True where a label should enter loss and metrics
    lengths       (B,)
    patient_id    list[str]

Every model therefore outputs one logit per time step, and the stay-level task (mortality)
simply reads the logit at the last observed hour. Predictions at step t only use data up
to t, as long as the model is causal (all our RNNs are).
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from missingness.data.preprocessing import Normalizer, featurize
from missingness.data.records import PatientRecord


class TimeSeriesDataset(Dataset):
    def __init__(self, records: list[PatientRecord], norm: Normalizer, task: str):
        assert task in ("hourly", "stay"), task
        self.records = records
        self.task = task
        self.items = [self._build(r, norm) for r in records]

    def _build(self, r: PatientRecord, norm: Normalizer) -> dict:
        item = featurize(r, norm)
        T = len(r.hours)
        if self.task == "hourly":
            labels = r.labels.astype(np.float32)
            label_mask = np.ones(T, dtype=bool)
        else:
            labels = np.zeros(T, dtype=np.float32)
            labels[-1] = r.labels[-1]
            label_mask = np.zeros(T, dtype=bool)
            label_mask[-1] = True
        item.update(labels=labels, label_mask=label_mask, patient_id=r.patient_id)
        return item

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict:
        return self.items[i]


def collate(batch: list[dict]) -> dict:
    lengths = torch.tensor([len(b["hours"]) for b in batch])
    T = int(lengths.max())

    def pad(key: str, dtype=torch.float32) -> torch.Tensor:
        first = batch[0][key]
        out = torch.zeros((len(batch), T, *first.shape[1:]), dtype=dtype)
        for i, b in enumerate(batch):
            out[i, : len(b[key])] = torch.as_tensor(b[key], dtype=dtype)
        return out

    return {
        "x": pad("x"),
        "m": pad("m"),
        "delta": pad("delta"),
        "hours": pad("hours"),
        "labels": pad("labels"),
        "label_mask": pad("label_mask", torch.bool),
        "static": torch.as_tensor(np.stack([b["static"] for b in batch]), dtype=torch.float32),
        "lengths": lengths,
        "patient_id": [b["patient_id"] for b in batch],
    }
