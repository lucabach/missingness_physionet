"""Week-2 check: cheap models on hand-made causal features, before any deep model.

`features.py` builds one feature row per prediction step from data up to that step;
`scripts/run_baselines.py` fits logistic regression and gradient boosting on each feature
set and writes results/baselines.csv and results/sofa_check.csv.
"""

from missingness.baselines.features import FEATURE_SETS, build_features

__all__ = ["FEATURE_SETS", "build_features"]
