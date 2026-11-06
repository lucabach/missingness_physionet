"""How much predictive information the measurement pattern carries (proposal, RQ1).

    S        = (AUROC(m, delta) - 0.5) / (AUROC(x, m, delta) - 0.5)
    Delta_mask = AUROC(x, m, delta) - AUROC(x)

Delta_mask is a lower bound: forward-filled values partly reveal when measurements occurred.
For GRU-D the values-only variant is a plain GRU; say so wherever Delta_mask is reported.
"""

from __future__ import annotations


def information_share(auroc_pattern: float, auroc_all: float) -> float:
    return (auroc_pattern - 0.5) / (auroc_all - 0.5)


def delta_mask(auroc_all: float, auroc_values: float) -> float:
    return auroc_all - auroc_values
