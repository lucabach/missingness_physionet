"""How much of the measurement pattern a values-only model can read (leak audit, D29).

With forward fill, a new measurement shows up as a change of x, so the mask can be rebuilt
from x (``preprocessing.decode_mask``). ``mask_recovery`` counts, per variable, which share of
the true measurements a values-only model can see this way. For the snapshot view it also
reports the share of empty bins, the only part of the pattern that still differs between
patients there.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from missingness.data.preprocessing import (
    Normalizer,
    bin_ends,
    compute_mask,
    decode_mask,
    featurize,
)
from missingness.data.records import PatientRecord


def mask_recovery(
    records: list[PatientRecord],
    norm: Normalizer,
    names: list[str],
    features: dict | None = None,
) -> pd.DataFrame:
    """One row per variable and one ``ALL`` row.

    Columns: ``n_measured`` (true mask entries in the view), ``recall`` (share of them that the
    decoded mask finds), ``false_hits`` (decoded but not measured; 0 by construction),
    ``n_bin_ends`` and ``empty_bin_share`` (snapshot view only: complete bins without a value).
    """
    features = features or {"view": "forward_fill"}
    widths = features.get("snapshot_widths")
    D = len(names)
    measured, hits, false, ends, empty = (np.zeros(D) for _ in range(5))
    for r in records:
        item = featurize(r, norm, **features)
        m = item["m"]  # for snapshot: the schedule's mask; else from the NaNs
        if features.get("view") == "decoded":
            m = compute_mask(r.values)
        decoded = decode_mask(item["x"])
        measured += m.sum(0)
        hits += (decoded * m).sum(0)
        false += (decoded * (1 - m)).sum(0)
        if widths is not None:
            for d in range(D):
                if widths[d] > 0:
                    e = bin_ends(r.hours, int(widths[d]))
                    ends[d] += e.sum()
                    empty[d] += (1 - m[e, d]).sum()
    rows = []
    for d, name in enumerate(names):
        rows.append(_row(name, measured[d], hits[d], false[d], ends[d], empty[d], widths, d))
    rows.append(_row("ALL", measured.sum(), hits.sum(), false.sum(), ends.sum(), empty.sum()))
    return pd.DataFrame(rows)


def _row(name, measured, hits, false, ends, empty, widths=None, d=None) -> dict:
    return {
        "variable": name,
        "width": int(widths[d]) if widths is not None and d is not None else None,
        "n_measured": int(measured),
        "recall": round(hits / measured, 4) if measured else None,
        "false_hits": int(false),
        "n_bin_ends": int(ends),
        "empty_bin_share": round(empty / ends, 4) if ends else None,
    }
