# Busy-hour removal at 20 / 25 / 30% (retrained models, 24 Nov 2026)

M2 = (random − state-dependent AUROC loss) for `all` minus the same for `values`. Positive means the pattern inputs make the model lose more under state-dependent removal than under matched random removal. Mean of 3 seeds, 95% paired patient bootstrap. * = interval excludes 0. Sepsis: hours before onset, trained on hospital 1. Models are fresh cloud retrains of the frozen configs, so numbers can differ slightly from the slides.

Achieved removal: the full 20 / 25 / 30% for every patient (sepsis hospital 1: 29.9% at 30%).

Note: the one significant hospital-1 result (GRU-D, 20%) comes from the values-only model doing better under state-dependent removal (AUROC drop −0.004 vs +0.008 random), not from the pattern model losing more (+0.013 vs +0.012).

| where          | rule                   | setting   | GRU                     | GRU-D                     | ODE-RNN                   |
|:---------------|:-----------------------|:----------|:------------------------|:--------------------------|:--------------------------|
| mortality      | extras (D5)            | p=0.05    | +0.000 [-0.003, +0.003] | +0.000 [-0.003, +0.003]   | +0.001 [-0.001, +0.002]   |
| mortality      | extras (D5)            | p=0.1     | -0.002 [-0.006, +0.002] | +0.003 [-0.002, +0.008]   | -0.001 [-0.004, +0.002]   |
| mortality      | extras (D5)            | p=0.15    | -0.000 [-0.005, +0.004] | +0.001 [-0.005, +0.007]   | -0.001 [-0.004, +0.002]   |
| mortality      | whole busy hours (D27) | p=0.2     | +0.000 [-0.004, +0.004] | -0.001 [-0.007, +0.006]   | +0.003 [-0.001, +0.007]   |
| mortality      | whole busy hours (D27) | p=0.25    | -0.002 [-0.007, +0.003] | +0.002 [-0.004, +0.008]   | +0.000 [-0.003, +0.004]   |
| mortality      | whole busy hours (D27) | p=0.3     | +0.001 [-0.004, +0.006] | +0.004 [-0.004, +0.011]   | -0.000 [-0.006, +0.006]   |
| sepsis hosp. 1 | whole busy hours (D27) | p=0.2     | +0.009 [-0.003, +0.021] | +0.014 [+0.003, +0.025] * | +0.005 [-0.003, +0.013]   |
| sepsis hosp. 1 | whole busy hours (D27) | p=0.25    | +0.000 [-0.011, +0.012] | +0.000 [-0.010, +0.012]   | +0.003 [-0.006, +0.012]   |
| sepsis hosp. 1 | whole busy hours (D27) | p=0.3     | -0.003 [-0.017, +0.012] | +0.002 [-0.013, +0.017]   | +0.006 [-0.004, +0.017]   |
| sepsis hosp. 2 | whole busy hours (D27) | p=0.2     | -0.008 [-0.018, +0.004] | -0.003 [-0.015, +0.012]   | +0.009 [-0.001, +0.018]   |
| sepsis hosp. 2 | whole busy hours (D27) | p=0.25    | -0.004 [-0.018, +0.008] | -0.003 [-0.020, +0.013]   | -0.003 [-0.014, +0.007]   |
| sepsis hosp. 2 | whole busy hours (D27) | p=0.3     | +0.006 [-0.008, +0.020] | -0.006 [-0.025, +0.010]   | +0.012 [+0.002, +0.021] * |
