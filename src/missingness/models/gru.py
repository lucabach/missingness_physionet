"""GRU baseline on forward-filled inputs (Che et al. 2018, "GRU-Forward"/"GRU-Simple")."""

from __future__ import annotations

import torch
from torch import nn

from missingness.models.inputs import build_inputs, n_features


class GRUModel(nn.Module):
    def __init__(
        self,
        n_vars: int,
        n_static: int,
        inputs: str = "all",
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.inputs = inputs
        self.gru = nn.GRU(
            n_features(n_vars, n_static, inputs),
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, batch: dict) -> torch.Tensor:
        h, _ = self.gru(build_inputs(batch, self.inputs))  # causal: step t sees inputs <= t
        return self.head(h).squeeze(-1)
