"""Hourly GRU-D following Che et al. (2018), with the study's input ablations."""

import torch
from torch import nn

from missingness.models.gru import GRUModel
from missingness.models.inputs import INPUT_VARIANTS


class GRUD(nn.Module):
    def __init__(self, n_vars: int, n_static: int, inputs: str = "all", hidden_size: int = 64):
        super().__init__()
        if inputs not in INPUT_VARIANTS:
            raise ValueError(f"unknown inputs {inputs!r}")
        self.inputs = inputs
        if inputs == "values":
            self.values_model = GRUModel(n_vars, n_static, inputs, hidden_size)
            return

        # A constant zero value contributes nothing to any gate in the pattern ablation.
        width = n_vars
        if inputs == "all":
            width += n_vars + n_static
            self.input_scale = nn.Parameter(torch.full((n_vars,), 0.1))
            self.input_bias = nn.Parameter(torch.zeros(n_vars))
        self.hidden_decay = nn.Linear(n_vars, hidden_size)
        self.input_gates = nn.Linear(width, 3 * hidden_size)
        self.hidden_gates = nn.Linear(hidden_size, 2 * hidden_size, bias=False)
        self.hidden_candidate = nn.Linear(hidden_size, hidden_size, bias=False)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, batch: dict) -> torch.Tensor:
        if self.inputs == "values":
            return self.values_model(batch)
        mask, gaps = batch["m"], batch["delta"]
        features = mask
        if self.inputs == "all":
            decay = torch.exp(-torch.relu(gaps * self.input_scale + self.input_bias))
            # Missing values are already forward-filled. The normalized training mean is zero.
            values = batch["x"] * torch.where(mask.bool(), 1.0, decay)
            static = batch["static"][:, None].expand(-1, mask.size(1), -1)
            features = torch.cat((values, mask, static), dim=-1)

        gates = self.input_gates(features)
        decay_h = torch.exp(-torch.relu(self.hidden_decay(gaps)))
        hidden = torch.zeros_like(decay_h[:, 0])
        states = []
        for projected, decay in zip(gates.unbind(1), decay_h.unbind(1), strict=True):
            previous = hidden * decay
            gate_inputs, candidate_input = projected.split(
                (2 * hidden.size(-1), hidden.size(-1)), dim=-1
            )
            reset, update = (gate_inputs + self.hidden_gates(previous)).sigmoid().chunk(2, dim=-1)
            candidate = (candidate_input + self.hidden_candidate(reset * previous)).tanh()
            hidden = torch.lerp(previous, candidate, update)
            states.append(hidden)
        return self.head(torch.stack(states, dim=1)).squeeze(-1)
