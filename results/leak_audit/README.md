# Leak audit (D29, proposed)

`python scripts/leak_audit.py` and `python scripts/leak_audit.py --coverage`, 28 Nov 2026, on
the committed splits (sepsis cut at hour 59, D15). No training, no labels.

**How much of the pattern a forward-fill `values` model can see.** Share of all measurements
of the test set that show up as a change of the forward-filled x (`recall`, `view =
forward_fill`, row `ALL`):

| Test set | measurements | visible as a change of x |
| --- | --- | --- |
| mortality | 654,763 | 80.9% |
| sepsis, hospital 1 | 785,453 | 85.3% |
| sepsis, hospital 2 | 703,862 | 88.3% |

The rest are exact repeats of the carried value (Weight 4%, FiO2 38%, O2Sat 66% at hospital 1;
most vital signs 85–97%).

**Snapshot view** (vitals 4 h, labs 24 h, variables kept if ≥ 95% of the complete bins of the
training records hold a value): mortality keeps 8 of 37 variables, sepsis 14 of 34. The only
pattern left is an empty bin: 2.9% (mortality), 2.0% (hospital 1), 8.2% (hospital 2, whose
rates were not used to pick the variables).

`snapshot_coverage.csv`: share of complete bins with a value per variable for widths 1–24 h,
on the training split (hospital 1 for sepsis).

**Preview of the decoded-pattern model** (GRU, mortality, seed 0, tuned configuration, 28 Nov,
not yet the 3-seed protocol): the `pattern` model on the true m and delta reaches test AUROC
0.770 [0.743, 0.798]; on m and delta rebuilt from the forward-filled x alone 0.757 [0.728,
0.788]. Paired difference (`M1_decoded_gap`) +0.013 [−0.005, 0.032]. So about 95% of the
pattern model's gain over chance is available to a model that only sees forward-filled values.
