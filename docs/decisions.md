# Decision log

Every choice that can change results gets an entry: number, date, who, what, why.
Never renumber; supersede an entry with a new one that says which it replaces.

| # | Date | Who | Decision | Why |
| --- | --- | --- | --- | --- |
| D1 | 2026-11-06 | team + Sarah | Core models GRU, GRU-D, ODE-RNN; Neural CDE and Transformer are extensions | Scope too large; Sarah suggested CDE may be too much and asked for simpler ODE-type models |
| D2 | 2026-11-06 | team + Sarah | Tasks: sepsis (PhysioNet 2019) and mortality (PhysioNet 2012) | Sepsis has few positives and a measurement-linked label; mortality gives a cleaner contrast |
| D3 | 2026-11-06 | repo setup | PhysioNet 2012 on an hourly grid (0–47 h), mean of repeated values within an hour | Same input format and models for both tasks; revisit in week 1 if it loses too much |
| D4 | 2026-11-06 | repo setup | Patient splits 70/15/15, stratified by outcome and hospital, `seed0`, committed in `splits/` | Same patients for everyone; split per hospital so each has its own test set |
| D5 | 2026-11-06 | repo setup | State-dependent removal, first version: baseline = patient's median measurements per hour; observations above it are candidates; random removal takes the same count per patient | Follows the proposal text; week-2 owner reviews |
| D6 | 2026-11-06 | repo setup | Thinning: per-variable keep probability target_rate / source_rate (only where source measures more) | Simplest rate matching that only removes; week-2 owner reviews |
| D7 | 2026-11-06 | repo setup | Static covariates count as values (in `values` and `all`, not in `pattern`); missing statics → training mean | Keeps the pattern variant strictly about measurement timing |
| D8 | 2026-11-13 | Luca | Splits `seed0` created and committed for both datasets (2019: 14,235/3,050/3,051 hospital-1 and 14,000/3,000/3,000 hospital-2 patients; 2012: 8,400/1,800/1,800 stays). Outcome rates are equal across splits | Implements D4 on the real data |
| D9 | 2026-11-13 | Luca | 2019 download fetches the ~40k single files in parallel with retries (PhysioNet offers no ZIP for 2019) | The sequential wget took too long and the server drops some requests |
| D10 | 2026-11-13 | Luca | Drop the 12 PhysioNet 2012 stays without any time-series value (IDs 140501, 140936, 141264, 142731, 142998, 143656, 145611, 147514, 150309, 150649, 155655, 156254): 11,988 stays, 1,707 deaths (14.2%). The committed split keeps its IDs; the loader skips them (8,390 / 1,798 / 1,800) | Standard practice: Horn et al. 2020 (SeFT) remove exactly these 12, Raindrop (Zhang et al. 2022) and PyPOTS/ehrdata follow; a pattern-only model sees nothing for them |
| D11 | 2026-11-13 | Luca | Train sepsis models on hospital 1 (`train_source: hospital1`); thin hospital-1 test patients toward hospital 2. Keep all 34 variables, including EtCO2 and TroponinI | Hospital 1 measures 23 of 34 variables more often (up to 45x); the 11 variables it measures less (most under 2x; EtCO2 never, TroponinI 15x less) cannot be matched by thinning and are reported as a limitation |
| D12 | 2026-11-13 | Luca | Outcomes: sepsis (PhysioNet 2019) and in-hospital mortality (PhysioNet 2012); no kidney failure | 2,932 septic patients are enough; mortality is the contrast whose label is not built from measurements; a kidney-failure label would have to be built from creatinine values |
| D13 | 2026-11-13 | Luca | SOFA variables for the label-construction check: MAP, Platelets, Bilirubin_total, Creatinine, FiO2 (SaO2 removed) | SaO2 is not part of SOFA; PaO2, GCS, urine output and vasopressors are not in the data |
| D14 | 2026-11-13 | Luca | Thinning target = per-variable measurement rates of the hospital-2 test split; thinned hospital-1 test sets saved for seeds 0, 1, 2 (`scripts/build_thinned_sets.py`) | Same rates `evaluate.py --perturb thin` uses; 23 matched variables land within 0.001 of the target rate, about 15% of values removed |

## Open questions

- **Q1–Q3, Q6:** resolved by D10–D13 (13 Nov).
- **Q4 (week 2):** Removal rule (D5). Measured on the test sets: the "extra" measurements the
  rule can remove are only 23.8% (hospital 1) and 17.2% (hospital 2) of all values. A stricter
  per-variable rule ("measured sooner than this patient's usual gap") gives even fewer (5.8% /
  4.0%). So the rule is fine, but p = 0.3 or 0.5 cannot be reached. Recommendation: keep D5 and
  use p in {0.05, 0.10, 0.15}, which both hospitals can reach in full.
- **Q5 (week 3):** Which open-source GRU-D and ODE-RNN implementations to port.
