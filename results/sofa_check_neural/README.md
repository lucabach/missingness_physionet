# SOFA label check for GRU, GRU-D and ODE-RNN (26 Nov 2026)

The sepsis label (Reyna et al. 2020) is computed from SOFA lab values, so part of the pattern
signal may be linked to the label by construction. As for gradient boosting
(`results/sofa_check.csv`), we retrain the sepsis `all` model (frozen configs, seeds 0–2,
trained on hospital 1) with the measurement pattern (mask and time gaps) of the five SOFA
variables hidden, keeping their values, and compare with hiding the pattern of five random
other variables (the same five draws as gradient boosting). See D28 (proposed).

Loss = AUROC(nothing hidden) − AUROC(pattern hidden), mean of 3 seeds, 95% paired patient
bootstrap (1000 draws). * = interval excludes 0. "SOFA minus random mean" is the SOFA loss
minus the mean loss of the five random draws on the same bootstrap draws; its interval covers
the test patients only, not the choice of the five random draws.

Full table, per seed: `sofa_check_neural.csv`. Rerun:

```bash
python scripts/sofa_check_neural.py --print-commands | xargs -d '\n' -P 4 -n 1 bash -c   # 63 runs, ~1 h on 4 CPUs
python scripts/sofa_check_neural.py
```

### hours before onset (headline)

| model | hospital | SOFA pattern hidden | 5 random draws (range) | SOFA minus random mean |
| --- | --- | --- | --- | --- |
| GRU | 1 | +0.009 [-0.004, +0.024] | -0.002 to +0.006 | +0.007 [-0.005, +0.021] |
| GRU | 2 | +0.012 [-0.006, +0.030] | -0.008 to +0.008 | +0.012 [+0.003, +0.020] * |
| GRU-D | 1 | +0.015 [+0.002, +0.027] * | -0.001 to +0.003 | +0.014 [+0.001, +0.026] * |
| GRU-D | 2 | +0.012 [-0.003, +0.027] | -0.018 to +0.003 | +0.021 [+0.010, +0.032] * |
| ODE-RNN | 1 | +0.013 [+0.003, +0.024] * | +0.007 to +0.016 | +0.003 [-0.010, +0.015] |
| ODE-RNN | 2 | +0.019 [-0.000, +0.036] | +0.002 to +0.028 | +0.002 [-0.014, +0.016] |

### all hours

| model | hospital | SOFA pattern hidden | 5 random draws (range) | SOFA minus random mean |
| --- | --- | --- | --- | --- |
| GRU | 1 | +0.007 [-0.006, +0.021] | -0.004 to +0.003 | +0.007 [-0.005, +0.020] |
| GRU | 2 | +0.006 [-0.011, +0.022] | -0.009 to +0.005 | +0.008 [-0.000, +0.016] |
| GRU-D | 1 | +0.013 [+0.002, +0.025] * | -0.002 to +0.006 | +0.013 [+0.002, +0.023] * |
| GRU-D | 2 | +0.012 [-0.001, +0.026] | -0.016 to +0.001 | +0.021 [+0.010, +0.032] * |
| ODE-RNN | 1 | +0.014 [+0.004, +0.024] * | +0.005 to +0.013 | +0.006 [-0.005, +0.017] |
| ODE-RNN | 2 | +0.013 [-0.008, +0.032] | +0.000 to +0.024 | -0.000 [-0.017, +0.014] |
## Reading

- **GRU-D:** hiding the SOFA pattern costs more than hiding five random variables, at both
  hospitals (+0.014 and +0.021 before onset). At hospital 2 part of this contrast comes from the
  random draws *improving* GRU-D (losses down to −0.018).
- **GRU:** the SOFA loss (+0.009 at hospital 1) is larger than every random draw but not
  significant at hospital 1; the contrast is significant at hospital 2 (+0.012).
- **ODE-RNN:** hiding any five variables' pattern costs about as much as hiding the SOFA
  pattern (random +0.007 to +0.016 at hospital 1): no SOFA-specific effect.
- The losses are small next to the pattern's gain (Δmask +0.026 to +0.043 before onset at hospital 1): part of
  the pattern signal is linked to the label by construction, most is not. Mortality stays the
  clean contrast.
