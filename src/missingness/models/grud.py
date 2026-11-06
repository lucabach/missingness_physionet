"""GRU-D (Che et al. 2018): GRU with learned decay of inputs and hidden state.

TODO (week 3 owner): port an open-source implementation and make it follow the common
interface (``forward(batch) -> logits (B, T)``). Starting points:
* original Keras code: https://github.com/PeterChe1990/GRU-D
* PyTorch ports, e.g. the one in Raindrop's baselines (Zhang et al. 2022) or
  https://github.com/zhiyongc/GRU-D

GRU-D consumes x, m and delta itself, so ``inputs`` means:
* "all":     standard GRU-D;
* "values":  no mask/decay, which makes it a plain GRU (state this in the report);
* "pattern": m and delta only, x replaced by zeros.
"""

from __future__ import annotations

import torch
from torch import nn


class GRUD(nn.Module):
    def __init__(self, n_vars: int, n_static: int, inputs: str = "all", **cfg):
        super().__init__()
        self.inputs = inputs
        self.cfg = cfg

    def forward(self, batch: dict) -> torch.Tensor:
        raise NotImplementedError("GRU-D is a week-3 deliverable, see the module docstring.")
