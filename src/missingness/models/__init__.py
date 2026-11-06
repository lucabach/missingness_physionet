"""Sequence models. Every model maps a batch (see data/dataset.py) to logits of shape (B, T).

Core: GRU (forward-filled inputs), GRU-D, ODE-RNN. Extensions: Neural CDE, Transformer.
Add a model by writing a class with ``__init__(self, n_vars, n_static, inputs, **cfg)`` and
``forward(batch) -> logits`` and registering it in ``MODELS``.
"""

from __future__ import annotations

from torch import nn

from missingness.models.gru import GRUModel
from missingness.models.grud import GRUD
from missingness.models.odernn import ODERNN

MODELS: dict[str, type[nn.Module]] = {
    "gru": GRUModel,
    "grud": GRUD,
    "odernn": ODERNN,
}


def build_model(model_cfg: dict, n_vars: int, n_static: int, inputs: str) -> nn.Module:
    cfg = dict(model_cfg)
    name = cfg.pop("name")
    if name not in MODELS:
        raise KeyError(f"Unknown model {name!r}; available: {sorted(MODELS)}")
    return MODELS[name](n_vars=n_vars, n_static=n_static, inputs=inputs, **cfg)
