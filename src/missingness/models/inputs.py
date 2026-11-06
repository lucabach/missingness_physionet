"""Input variants for the information ablation (RQ1).

    values   x and static covariates       -> AUROC(x)
    pattern  m and delta                   -> AUROC(m, delta)
    all      x, m, delta and static        -> AUROC(x, m, delta)

Time gaps enter as log(1 + delta) to keep their scale close to the other inputs.
"""

from __future__ import annotations

import torch

INPUT_VARIANTS = ("values", "pattern", "all")


def n_features(n_vars: int, n_static: int, inputs: str) -> int:
    return {
        "values": n_vars + n_static,
        "pattern": 2 * n_vars,
        "all": 3 * n_vars + n_static,
    }[inputs]


def build_inputs(batch: dict, inputs: str) -> torch.Tensor:
    """Concatenate the per-step features of one input variant, shape (B, T, F)."""
    if inputs not in INPUT_VARIANTS:
        raise ValueError(f"inputs must be one of {INPUT_VARIANTS}, got {inputs!r}")
    T = batch["x"].shape[1]
    static = batch["static"].unsqueeze(1).expand(-1, T, -1)
    log_delta = torch.log1p(batch["delta"])
    parts = {
        "values": [batch["x"], static],
        "pattern": [batch["m"], log_delta],
        "all": [batch["x"], batch["m"], log_delta, static],
    }[inputs]
    return torch.cat(parts, dim=-1)
