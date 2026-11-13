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

## Open questions

- **Q1 (week 1):** Final label set: sepsis + mortality, or add kidney failure? Facts (13 Nov):
  2,932 septic patients (hospital 1: 1,790 = 8.8%; hospital 2: 1,142 = 5.7%); 1.8% positive
  hours overall, 1.1% on pre-onset hours; 426 septic patients are already positive at their
  first recorded hour, so they have no negative pre-onset hours. Mortality: 1,707 deaths in
  12,000 stays (14.2%). Kidney failure has no label in either dataset; it would have to be
  built from creatinine (and urine output, only in 2012), so the label would come from
  measurements even more directly than sepsis. Suggestion: keep sepsis + mortality.
- **Q2 (week 1):** Which hospital trains for thinning? Facts (`results/hospital_rates.csv`):
  overall both measure about equally (0.21 vs 0.19 observed values per patient-hour), but
  per variable they differ a lot. Hospital 1 measures 23 of 34 variables more often, mostly
  blood gases and chemistry (BaseExcess ~50x, HCO3 ~40x, Chloride ~14x, FiO2 ~6x, pH ~5x,
  PaCO2 and PTT ~4x). Hospital 2 measures 11 more often: EtCO2 (never recorded in
  hospital 1), TroponinI (~15x), Glucose and DBP (~1.7x). Since thinning only removes,
  training on hospital 1 lets us match 23 variables; the other 11 stay unmatched.
- **Q3 (week 1):** SOFA variables. SOFA has 6 parts: respiration (PaO2/FiO2), coagulation
  (platelets), liver (bilirubin), cardiovascular (MAP or vasopressors), brain (GCS) and kidney
  (creatinine or urine output). PhysioNet 2019 has no PaO2, GCS, urine or vasopressors.
  Proposed list: MAP, Platelets, Bilirubin_total, Creatinine, FiO2. Drop SaO2 from the
  current config: it is not part of SOFA.
- **Q4 (week 2):** Is the state-dependent baseline (D5) the right one, or a causal rolling
  baseline? On real data with p = 0.3 it only finds enough candidates for 23% (hospital 1) and
  16% (hospital 2) of values, so both methods remove less than asked.
- **Q5 (week 3):** Which open-source GRU-D and ODE-RNN implementations to port.
- **Q6:** resolved by D10.
