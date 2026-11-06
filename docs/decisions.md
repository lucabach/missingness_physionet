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

## Open questions

- **Q1 (week 1):** Final label set: sepsis + mortality, or add kidney failure? (Sarah raised it.)
- **Q2 (week 1):** Which hospital measures more, i.e. which one trains for thinning?
- **Q3 (week 1):** Confirm the SOFA variable list in `configs/data/physionet2019.yaml`.
- **Q4 (week 2):** Is the state-dependent baseline (D5) the right one, or a causal rolling baseline?
- **Q5 (week 3):** Which open-source GRU-D and ODE-RNN implementations to port.
