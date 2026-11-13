# Week 2 handoff (dates: 16–22 Nov) — started early by Luca on 13 Nov

Branch `luca_branchmann`, same pull request as week 1.

## Done
- Causal hand-made features (`src/missingness/baselines/features.py`, tested): pattern-only
  (count so far, hours since last, measured now), values-only (last value, training mean
  before the first one, plus statics), time-only (hours in ICU), all.
- Logistic regression and gradient boosting on every feature set, both tasks:
  `scripts/run_baselines.py` → `results/baselines.csv` (AUROC/AUPRC with patient-bootstrap CIs,
  pre-onset hours for sepsis).
- SOFA-variable check → `results/sofa_check.csv`.
- Removal rule reviewed (Q4 in `docs/decisions.md`).
- Thinned hospital-1 test sets for seeds 0–2 (`scripts/build_thinned_sets.py`), invariants
  checked, rates in `results/thinning_rates.csv` (D14).

## Results (test AUROC, sepsis trained on hospital 1)
| Features | Gradient boosting h1 | Gradient boosting h2 | Logistic regression h1 | Logistic regression h2 | Mortality GB | Mortality LR |
| --- | --- | --- | --- | --- | --- | --- |
| pattern only | 0.793 | 0.744 | 0.729 | 0.490 | 0.770 | 0.758 |
| values only | 0.750 | 0.666 | 0.703 | 0.671 | 0.853 | 0.851 |
| time only | 0.712 | 0.696 | 0.668 | 0.624 | – | – |
| all | 0.814 | 0.753 | 0.773 | 0.665 | 0.866 | 0.860 |

- Sepsis: the pattern alone beats the values (0.79 vs 0.75); S ≈ 0.93, Δmask ≈ +0.06 (GB, h1).
  Mortality is the opposite: values dominate (0.85 vs 0.77); S ≈ 0.74, Δmask ≈ +0.01.
- Transfer: the pattern-only logistic regression collapses at hospital 2 (0.73 → 0.49). EtCO2
  explains part of it (never seen in training); without EtCO2 it still drops to 0.61, the
  largest loss of any feature set.
- SOFA check (GB, all features): dropping the SOFA variables' pattern costs 0.011 AUROC at
  hospital 1 (0.814 → 0.803); dropping 5 random other variables costs ~0 (0.811–0.817). At
  hospital 2 the random draws vary too much (0.712–0.755) to read the SOFA effect (0.728).
- Removal: the state-dependent rule finds only 24% / 17% "extra" values, so p ≤ 0.15 (Q4).

## Not done / open items
- Q4: confirm p ∈ {0.05, 0.10, 0.15}.
- Baseline numbers use one seed and default hyperparameters; fine as a sanity check.

## How to run
```bash
python scripts/run_baselines.py              # ~15 min on 4 CPUs
python scripts/build_thinned_sets.py --seeds 0 1 2
```
