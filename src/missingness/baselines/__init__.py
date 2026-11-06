"""Week-1/2 check: cheap models on hand-made features, before any deep model.

TODO (week 2 owner, see docs/work_plan.md). For every prediction hour t, build causal
features from data up to t and fit logistic regression and gradient boosting
(sklearn ``HistGradientBoostingClassifier``) on three feature sets:
* pattern-only:     per variable, number of measurements so far and hours since last one;
* values-only:      per variable, last observed value (forward fill);
* time-only:        hours since ICU admission.
SOFA-variable check: drop the pattern features of the SOFA variables
(``sofa_variables`` in configs/data/physionet2019.yaml) and compare with dropping the same
number of randomly chosen other variables.
Reuse ``missingness.eval`` for metrics so numbers are comparable with the deep models.
"""
