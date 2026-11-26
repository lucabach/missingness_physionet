# Removal: state-dependent vs random, 5% to 30% (retrained models, 24 Nov 2026)

M2 = (random − state-dependent AUROC loss) for `all` minus the same for `values`. Positive means the pattern inputs make the model lose more under state-dependent removal than under matched random removal. Mean of 3 seeds, 95% paired patient bootstrap. * = interval excludes 0.

Sepsis: all models are trained on hospital 1 only. "Tested on hosp. 2" means the same hospital-1 models scored on hospital-2 patients; nothing is trained on hospital 2. Scores use hours before onset.

Rules: extras (D5) removes only the extra tests in busy hours and runs out above 15%; whole busy hours (D27, proposed) removes complete busy hours and reaches the full 20 / 25 / 30% for every patient. All rows come from the same cloud retrain of the frozen configs, which reproduces the week-3 numbers closely.

| task / test set           | rule                   | removed   | GRU                     | GRU-D                     | ODE-RNN                   |
|:--------------------------|:-----------------------|:----------|:------------------------|:--------------------------|:--------------------------|
| mortality                 | extras (D5)            | 5%        | +0.000 [-0.003, +0.003] | +0.000 [-0.003, +0.003]   | +0.001 [-0.001, +0.002]   |
| mortality                 | extras (D5)            | 10%       | -0.002 [-0.006, +0.002] | +0.003 [-0.002, +0.008]   | -0.001 [-0.004, +0.002]   |
| mortality                 | extras (D5)            | 15%       | -0.000 [-0.005, +0.004] | +0.001 [-0.005, +0.007]   | -0.001 [-0.004, +0.002]   |
| mortality                 | whole busy hours (D27) | 20%       | +0.000 [-0.004, +0.004] | -0.001 [-0.007, +0.006]   | +0.003 [-0.001, +0.007]   |
| mortality                 | whole busy hours (D27) | 25%       | -0.002 [-0.007, +0.003] | +0.002 [-0.004, +0.008]   | +0.000 [-0.003, +0.004]   |
| mortality                 | whole busy hours (D27) | 30%       | +0.001 [-0.004, +0.006] | +0.004 [-0.004, +0.011]   | -0.000 [-0.006, +0.006]   |
| sepsis, tested on hosp. 1 | extras (D5)            | 5%        | -0.003 [-0.010, +0.003] | -0.002 [-0.009, +0.005]   | -0.003 [-0.008, +0.002]   |
| sepsis, tested on hosp. 1 | extras (D5)            | 10%       | +0.003 [-0.004, +0.009] | +0.001 [-0.006, +0.007]   | -0.000 [-0.006, +0.006]   |
| sepsis, tested on hosp. 1 | extras (D5)            | 15%       | +0.006 [-0.003, +0.014] | +0.010 [+0.002, +0.018] * | +0.006 [-0.001, +0.012]   |
| sepsis, tested on hosp. 1 | whole busy hours (D27) | 20%       | +0.009 [-0.003, +0.021] | +0.014 [+0.003, +0.025] * | +0.005 [-0.003, +0.013]   |
| sepsis, tested on hosp. 1 | whole busy hours (D27) | 25%       | +0.000 [-0.011, +0.012] | +0.000 [-0.010, +0.012]   | +0.003 [-0.006, +0.012]   |
| sepsis, tested on hosp. 1 | whole busy hours (D27) | 30%       | -0.003 [-0.017, +0.012] | +0.002 [-0.013, +0.017]   | +0.006 [-0.004, +0.017]   |
| sepsis, tested on hosp. 2 | extras (D5)            | 5%        | -0.002 [-0.009, +0.005] | -0.006 [-0.014, +0.001]   | +0.002 [-0.003, +0.008]   |
| sepsis, tested on hosp. 2 | extras (D5)            | 10%       | +0.002 [-0.007, +0.009] | +0.006 [-0.004, +0.018]   | -0.005 [-0.012, +0.001]   |
| sepsis, tested on hosp. 2 | extras (D5)            | 15%       | +0.005 [-0.005, +0.014] | +0.004 [-0.006, +0.014]   | -0.002 [-0.008, +0.004]   |
| sepsis, tested on hosp. 2 | whole busy hours (D27) | 20%       | -0.008 [-0.018, +0.004] | -0.003 [-0.015, +0.012]   | +0.009 [-0.001, +0.018]   |
| sepsis, tested on hosp. 2 | whole busy hours (D27) | 25%       | -0.004 [-0.018, +0.008] | -0.003 [-0.020, +0.013]   | -0.003 [-0.014, +0.007]   |
| sepsis, tested on hosp. 2 | whole busy hours (D27) | 30%       | +0.006 [-0.008, +0.020] | -0.006 [-0.025, +0.010]   | +0.012 [+0.002, +0.021] * |
