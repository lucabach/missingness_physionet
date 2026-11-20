"""Causal ODE-RNN following Rubanova et al. (2019) on the study's hourly grid."""

import torch
from torch import nn

from missingness.models.inputs import INPUT_VARIANTS


class ODERNN(nn.Module):
    def __init__(
        self,
        n_vars: int,
        n_static: int,
        inputs: str = "all",
        hidden_size: int = 64,
        ode_hidden: int = 64,
        solver: str = "euler",
    ):
        super().__init__()
        if solver != "euler":
            raise ValueError("the hourly ODE-RNN uses the fixed Euler solver")
        if inputs not in INPUT_VARIANTS:
            raise ValueError(f"unknown inputs {inputs!r}")
        self.inputs = inputs
        width = n_vars if inputs == "pattern" else n_vars + n_static
        if inputs == "all":
            width += n_vars
        self.update = nn.GRUCell(width, hidden_size)
        self.drift = nn.Sequential(
            nn.Linear(hidden_size, ode_hidden),
            nn.Tanh(),
            nn.Linear(ode_hidden, hidden_size),
            nn.Tanh(),
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, batch: dict) -> torch.Tensor:
        features = batch["m"]
        if self.inputs != "pattern":
            x = batch["x"]
            static = batch["static"][:, None].expand(-1, x.size(1), -1)
            parts = [x, static]
            if self.inputs == "all":
                parts.append(batch["m"])
            features = torch.cat(parts, dim=-1)

        hidden = features.new_zeros(features.size(0), self.update.hidden_size)
        states = []
        for t, observation in enumerate(features.unbind(1)):
            if t:
                # One forward Euler step over the interval since the preceding hourly row.
                dt = (batch["hours"][:, t] - batch["hours"][:, t - 1]).clamp_min(0)
                hidden = hidden + dt[:, None] * self.drift(hidden)
            hidden = self.update(observation, hidden)
            states.append(hidden)
        return self.head(torch.stack(states, dim=1)).squeeze(-1)
