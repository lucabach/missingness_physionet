"""Tiny YAML config system.

An experiment file names a data config and a model config; they are merged in this order
(later wins): configs/data/<data>.yaml -> configs/model/<model>.yaml -> experiment file ->
command-line overrides like ``train.lr=1e-3``.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from missingness.paths import CONFIG_DIR


def _deep_update(base: dict, new: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in new.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = value
    return out


def _parse_override(item: str) -> tuple[list[str], Any]:
    key, _, raw = item.partition("=")
    if not _:
        raise ValueError(f"Override must look like key.sub=value, got {item!r}")
    return key.split("."), yaml.safe_load(raw)


def load_config(experiment: str | Path, overrides: list[str] | None = None) -> dict:
    """Load an experiment YAML and merge its data/model configs and CLI overrides."""
    exp = yaml.safe_load(Path(experiment).read_text())
    cfg: dict = {}
    for kind in ("data", "model"):
        name = exp.get(kind)
        if isinstance(name, str):
            part = yaml.safe_load((CONFIG_DIR / kind / f"{name}.yaml").read_text())
            cfg = _deep_update(cfg, {kind: part})
            exp = {k: v for k, v in exp.items() if k != kind}
    cfg = _deep_update(cfg, exp)
    for item in overrides or []:
        keys, value = _parse_override(item)
        node = cfg
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = value
    return cfg


def save_config(cfg: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
