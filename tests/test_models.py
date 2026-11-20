"""Interface checks every registered model must pass (CLAUDE.md, invariant 5).

For each model and input variant: one logit per hour, shape (B, T), and causality: the logit
at hour t does not change when the record is changed after hour t. The variant must also
matter: `values` must not react to a change of the mask alone, `pattern` must not react to a
change of the values alone.
"""

from __future__ import annotations

import pytest
import torch

from missingness.data.dataset import TimeSeriesDataset, collate
from missingness.data.preprocessing import Normalizer
from missingness.data.synthetic import make_synthetic
from missingness.models import MODELS, build_model
from missingness.models.inputs import INPUT_VARIANTS

T_CUT = 6


def _cfg(name: str) -> dict:
    import yaml

    from missingness.paths import CONFIG_DIR

    return yaml.safe_load((CONFIG_DIR / "model" / f"{name}.yaml").read_text())


def _batch(records):
    ds = TimeSeriesDataset(records, Normalizer.fit(records), task="hourly")
    return collate([ds[i] for i in range(len(ds))])


def _cases():
    for name in sorted(MODELS):
        for inputs in INPUT_VARIANTS:
            yield pytest.param(name, inputs, id=f"{name}-{inputs}")


@pytest.mark.parametrize(("name", "inputs"), list(_cases()))
def test_model_shape_and_causality(name, inputs):
    torch.manual_seed(0)
    records = make_synthetic(n_patients=6, n_vars=4, n_static=2, seed=3)
    batch = _batch(records)
    model = build_model(_cfg(name), n_vars=4, n_static=2, inputs=inputs).eval()
    with torch.no_grad():
        out = model(batch)
    B, T = batch["x"].shape[:2]
    assert out.shape == (B, T)
    assert torch.isfinite(out).all()

    later = {k: v.clone() for k, v in batch.items() if torch.is_tensor(v)}
    later["patient_id"] = batch["patient_id"]
    later["x"][:, T_CUT + 1 :] += 3.0
    later["m"][:, T_CUT + 1 :] = 1.0 - later["m"][:, T_CUT + 1 :]
    later["delta"][:, T_CUT + 1 :] += 2.0
    with torch.no_grad():
        out_later = model(later)
    assert torch.allclose(out[:, : T_CUT + 1], out_later[:, : T_CUT + 1], atol=1e-5), (
        "the logit at hour t must only use the record up to hour t"
    )


@pytest.mark.parametrize(("name", "inputs"), list(_cases()))
def test_model_ignores_what_its_variant_hides(name, inputs):
    if inputs == "all":
        pytest.skip("`all` sees everything")
    torch.manual_seed(0)
    records = make_synthetic(n_patients=6, n_vars=4, n_static=2, seed=4)
    batch = _batch(records)
    model = build_model(_cfg(name), n_vars=4, n_static=2, inputs=inputs).eval()
    changed = dict(batch)
    if inputs == "values":  # mask and gaps hidden
        changed["m"] = 1.0 - batch["m"]
        changed["delta"] = batch["delta"] + 5.0
    else:  # pattern: values and static covariates hidden
        changed["x"] = batch["x"] + 3.0
        changed["static"] = batch["static"] + 3.0
    with torch.no_grad():
        assert torch.allclose(model(batch), model(changed), atol=1e-5)


@pytest.mark.parametrize("name", ["grud", "odernn"])
def test_time_dynamics_receive_training_gradients(name):
    """A plain GRU could pass the shape and causality checks without learning any decay/drift."""
    torch.manual_seed(4)
    batch = _batch(make_synthetic(n_patients=4, n_vars=4, n_static=2, seed=5))
    model = build_model(_cfg(name), n_vars=4, n_static=2, inputs="all")
    model(batch)[batch["label_mask"]].sum().backward()
    params = (
        [model.hidden_decay.weight, model.input_scale]
        if name == "grud"
        else list(model.drift.parameters())
    )
    for param in params:
        assert param.grad is not None and torch.isfinite(param.grad).all()
        assert param.grad.abs().sum() > 0
