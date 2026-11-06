"""Values or Habits? Sensitivity of clinical time-series models to measurement changes.

Package layout (see CLAUDE.md for the full map):
    data/       raw PhysioNet files -> PatientRecord -> padded torch batches
    perturb/    remove observations from real records (random, state-dependent, thinning)
    models/     GRU, GRU-D, ODE-RNN (core); Neural CDE (extension)
    baselines/  logistic regression / gradient boosting on hand-made features
    eval/       AUROC/AUPRC, patient bootstrap, information scores
"""

__version__ = "0.1.0"
