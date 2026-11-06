"""ODE-RNN (Rubanova et al. 2019): hidden state evolves by a neural ODE between observations.

TODO (week 3 owner): port from the authors' code, https://github.com/YuliaRubanova/latent_ode
(``lib/ode_rnn.py``), using ``torchdiffeq``. Requirements for this project:
* causal: the logit at hour t uses observations up to t only;
* one logit per hour, same interface as the GRU (``forward(batch) -> (B, T)``);
* same ``inputs`` variants as the other models (see models/inputs.py).
Expect roughly 1.6x the cost of a GRU (proposal, compute table).
"""

from __future__ import annotations

import torch
from torch import nn


class ODERNN(nn.Module):
    def __init__(self, n_vars: int, n_static: int, inputs: str = "all", **cfg):
        super().__init__()
        self.inputs = inputs
        self.cfg = cfg

    def forward(self, batch: dict) -> torch.Tensor:
        raise NotImplementedError("ODE-RNN is a week-3 deliverable, see the module docstring.")
