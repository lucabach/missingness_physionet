"""Which view of the records a run feeds its model (D29, proposed): config -> featurize kwargs.

    features: forward_fill   default: x forward-filled, m and delta from the NaNs
    features: snapshot       common schedule (``preprocessing.snapshot_values``); bin widths
                             from the data config's ``snapshot`` block
    features: decoded        m and delta rebuilt from x alone (leak audit, ``inputs=pattern``)

The snapshot block of a data config names a bin width for fast variables (vital signs) and
one for slow ones (labs), and a minimum coverage. At training time each variable keeps its
width only if at least ``min_coverage`` of the complete bins of the training records hold a
value; otherwise it is hidden (width 0). The resolved widths and coverages are written into
the run's ``config.yaml``, so evaluation uses exactly the same view without recomputing it.
"""

from __future__ import annotations

import numpy as np

from missingness.data.preprocessing import FEATURE_VIEWS, bin_ends
from missingness.data.records import PatientRecord


def variable_names(cfg: dict, n_vars: int) -> list[str]:
    names = cfg["data"].get("variables")
    return list(names) if names else [f"v{i}" for i in range(n_vars)]


def bin_coverage(records: list[PatientRecord], d: int, width: int) -> float:
    """Share of complete bins of ``width`` hours in which variable d has at least one value."""
    filled = total = 0
    for r in records:
        b = np.round(r.hours).astype(int) // width
        b -= b.min()
        complete = np.unique(b[bin_ends(r.hours, width)])
        has = np.zeros(b.max() + 1, dtype=bool)
        has[b[~np.isnan(r.values[:, d])]] = True
        filled += int(has[complete].sum())
        total += len(complete)
    return filled / total if total else 0.0


def nominal_widths(spec: dict, names: list[str]) -> dict[str, int]:
    """Bin width of each variable before the coverage rule: fast, slow or set by hand."""
    fast = set(spec.get("fast_variables") or [])
    by_hand = spec.get("variable_widths") or {}
    return {
        v: int(by_hand.get(v, spec["fast_width"] if v in fast else spec["slow_width"]))
        for v in names
    }


def resolve_snapshot(cfg: dict, train_records: list[PatientRecord]) -> dict:
    """Fill ``cfg['data']['snapshot']['widths']`` (and ``coverage``) from the training records.

    Does nothing when the view is not ``snapshot`` or the widths are already set (evaluation,
    or widths fixed by hand in the config).
    """
    if cfg.get("features", "forward_fill") != "snapshot":
        return cfg
    spec = cfg["data"].get("snapshot")
    if not spec:
        raise KeyError(f"features=snapshot needs a `snapshot` block in data config {cfg['data']}")
    if spec.get("widths"):
        return cfg
    names = variable_names(cfg, train_records[0].values.shape[1])
    widths, coverage = {}, {}
    for d, (name, w) in enumerate(nominal_widths(spec, names).items()):
        coverage[name] = round(bin_coverage(train_records, d, w), 4)
        widths[name] = w if coverage[name] >= spec["min_coverage"] else 0
    spec["widths"], spec["coverage"] = widths, coverage
    kept = [v for v, w in widths.items() if w]
    print(f"snapshot view: {len(kept)} of {len(names)} variables kept: {kept}")
    return cfg


def feature_kwargs(cfg: dict) -> dict:
    """Keyword arguments of ``preprocessing.featurize`` for this run's config."""
    view = cfg.get("features", "forward_fill")
    if view not in FEATURE_VIEWS:
        raise ValueError(f"features must be one of {FEATURE_VIEWS}, got {view!r}")
    if view != "snapshot":
        return {"view": view}
    widths = cfg["data"]["snapshot"].get("widths")
    if not widths:
        raise ValueError("snapshot widths not resolved; call views.resolve_snapshot first")
    names = variable_names(cfg, len(widths))
    return {"view": view, "snapshot_widths": np.array([widths[v] for v in names], dtype=int)}


def feature_tag(cfg: dict) -> str:
    """'' for forward fill; '_snapshot' or '_decoded' in the run name otherwise."""
    view = cfg.get("features", "forward_fill")
    return "" if view == "forward_fill" else f"_{view}"
