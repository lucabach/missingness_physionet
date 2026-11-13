# Study plan: same analysis on sepsis and mortality

Proposal by Matteo, 13 Nov 2026. Matteo says yes to every proposed fix; the rest of the team
confirms or objects in the pull request. `docs/work_plan.md` already follows this plan (who does what, by which week);
`docs/experiments.md` is updated once the team has confirmed. Why two tasks:
`docs/study_design.md`. Builds on the state of branch `luca_branchmann` (D8–D14, baselines,
thinned test sets).

## 1. Goal

Run the core experiments (RQ1–RQ3) so that every result is comparable between the two tasks,
reproducible from its config, and safe against the problems found in the first data checks.

## 2. Problems and fix decisions

Status: **decided** (in `docs/decisions.md`), **proposed** (Matteo: yes, 13 Nov; gets a D-number
once the team has confirmed and `luca_branchmann` is merged, to keep the numbering in one
line), **to do** (no decision needed, only work).

| # | Problem | Fix | Status |
| --- | --- | --- | --- |
| P1 | Records of patients without sepsis end at ICU hour 59, so the hour of the stay predicts the label (AUROC 0.71 from the hour alone). | Cut all PhysioNet 2019 records at ICU hour 59, in training, validation and test. Repeat the baselines after the cut. | proposed |
| P2 | The sepsis label is built from clinicians' orders and lab results. | Same analysis on mortality (D2, D12). State what remains as a limitation. | decided |
| P3 | Few patients with the outcome in a test set (about 180, 120 and 256), so one AUROC is uncertain by about ±0.04. | Report differences between two models on the same resampled patients (paired bootstrap). Seeds 0–2; add seeds 3–4 only where a primary measure is unclear. | to do |
| P4 | Nothing shows that our sensitivity measure detects a model that depends on the pattern. | New experiment E5: a planted pattern on real records (section 6). | proposed |
| P5 | Thinning to hospital 2's rates mostly deletes variables that hospital 2 hardly records. | Keep D11 and D14 (all 34 variables). Add a contrast on the 25 variables both hospitals record at a comparable rate (section 6). | proposed |
| P6 | State-dependent removal deletes mostly labs, random removal mostly vitals, so they differ in more than the choice of hours. | Random removal takes the same number of values **per variable** as state-dependent removal; only the hours differ. p in {0.05, 0.10, 0.15} (Q4). | proposed |
| P7 | "Values only" and "pattern only" are not defined for GRU-D and ODE-RNN. | Write the definitions (table in section 4) before the two models are implemented. | to do |
| P8 | 426 septic patients are positive from their first hour: no earlier hours, onset time unknown. | Keep them in training and in the all-hours numbers. Report pre-onset numbers with and without them. | proposed |
| P9 | Three motivating numbers come from 2026 preprints nobody has checked. | Read Nie et al., Gupta et al. and Tranchellini et al. before the report. | to do |
| P10 | The title and some wording promise more than the design can show. | Soften the title; say "change in measurement density" for thinning; say that irregular sampling times are out of scope. | to do |

**Cannot be fixed, must be stated in the report:** the label caveat; the missing long-stay
patients without sepsis; mortality being a different dataset; thinning not being a real
change of policy; the hourly grid.

## 3. Where we are

| Done (13 Nov) | By |
| --- | --- |
| Both datasets load; splits committed (D8); 12 empty 2012 stays dropped (D10) | Luca |
| Hospital comparison; train on hospital 1 (D11); thinned test sets (D14) | Luca |
| Classical baselines and SOFA check on both tasks, before the cut | Luca |
| GRU, training and evaluation scripts; 1-epoch runs on both tasks | repo setup, Luca |
| Hour-59 finding, caveats, this plan, work ledger | Matteo |

Not started: GRU-D, ODE-RNN, tuning, all training runs.

## 4. One protocol for both tasks

| | Rule |
| --- | --- |
| Code | One training script and one evaluation script for both tasks. No task-specific branches except the label format. |
| Label format (the only difference) | Sepsis: one label per hour, loss on every hour. Mortality: one label per stay, loss on the last hour. |
| Models | GRU, GRU-D, ODE-RNN, each under 1 M parameters. Build order: GRU (done), GRU-D, ODE-RNN. |
| Input variants | `values`, `pattern`, `all`, defined once (table below). |
| Preprocessing | Hourly grid, forward fill, z-score from the training set, mask, log(1 + gap). Same functions for both. |
| Splits | The committed 70/15/15 patient splits. The test split is not used until tuning is frozen. |
| Tuning | 8 configurations per model and task, seed 0, `all` inputs, chosen by validation AUROC. The chosen configuration is used for all three input variants. Check: repeat the sweep for GRU `values` on each task; if its best configuration is clearly better, tune per variant. |
| Training | Adam, batch 64, up to 30 epochs, early stopping on validation AUROC (patience 5), no class weights. Same for both tasks. |
| Seeds | 0, 1, 2 for every cell. |
| Metrics | AUROC first, AUPRC next to it. Sepsis: pre-onset hours are the headline, all hours next to it. |
| Statistics | Patient bootstrap. For a difference between two models: the same resampled patients for both. |
| Comparison between tasks | Only differences inside a task (M1–M3 below). Never raw scores. |

**Input variants per model** (proposal for P7):

| Model | `values` | `pattern` | `all` |
| --- | --- | --- | --- |
| GRU | forward-filled x | m, log gap | x, m, log gap |
| GRU-D | plain GRU on x (say so wherever reported) | decay driven by the gaps, values set to a constant | the original model |
| ODE-RNN | update every hour with forward-filled x, no mask | update with m only | update with x and m (the original model) |

**Primary measures, fixed before any run:**

| # | Measure | Answers |
| --- | --- | --- |
| M1 | Δmask = AUROC(`all`) − AUROC(`values`), per model and task, with S next to it | RQ1: what the pattern adds |
| M2 | Loss under state-dependent removal minus loss under matched random removal, for the `all` model, minus the same for the `values` model, at p = 0.10 | RQ2: sensitivity to a change in the pattern |
| M3 | Planted pattern: gain with the rule on, and loss when the rule is switched off, as a function of its strength | RQ1 and RQ2 with a known truth |
| M4 | Loss from hospital 1 to hospital 2, and under thinning, for `all` minus `values` | RQ3 (exploratory) |

## 5. Order of work

Each phase starts only when the one before has passed its gate.

| Phase | What | Task | Runs | Gate to pass |
| --- | --- | --- | --- | --- |
| 0 | Accept this plan. Implement the cut (P1), paired bootstrap (P3), matched random removal (P6); add the hour-59 check to `scripts/check_data.py`; save the git commit in every run's config; tag the repo `protocol-v1`. | both | – | tests green; decisions recorded |
| 1 | Repeat the classical baselines after the cut. | both | CPU | the hour-only AUROC for sepsis has dropped; if it has not, stop and look |
| 2 | GRU on mortality: tuning, then the three input variants. | mortality | 8 + 9 | test AUROC of `all` in the range of published results for this dataset (about 0.82 to 0.86; to check against Che et al. 2018) |
| 3 | GRU-D and ODE-RNN: implement, test on synthetic data, then mortality: tuning and input variants. Removal evaluation for all mortality models. | mortality | 16 + 18 | same check as phase 2; M1 and M2 computed for mortality |
| 4 | Sepsis, trained on hospital 1: tuning, input variants, removal evaluation. Every run is scored on both hospitals. | sepsis | 24 + 27 | M1 and M2 computed for sepsis |
| 5 | Shift experiments (section 6): E5 planted pattern, thinning, transfer. | see section 6 | 36 + 18 | M3 and M4 computed |
| 6 | Only if time is left: sepsis trained on hospital 2 (27 runs: RQ1 at hospital 2 and the reverse transfer); more seeds where a primary measure is unclear. | sepsis | up to 27 | – |

Phases 2–3 (mortality) and the implementation work of phase 3 can overlap. E5 can start as soon
as phase 3 is done, in parallel with phase 4.

**Why mortality first:** its runs take about a third of the time, published numbers exist to
check our setup, and it has neither caveat. Errors in the pipeline show up there cheaply.

**Compute:** about 90 mortality runs (about 10 GPU-hours) and about 75 sepsis runs (about 22
GPU-hours), plus up to 9 for phase 6. Inside the 35 GPU-hours planned for the core if phase 6
is skipped.

## 6. The three shift experiments

| | E5 Planted pattern (new) | Thinning (E4) | Transfer (E4) |
| --- | --- | --- | --- |
| What | On real records, delete measurements of one densely measured variable (chosen and recorded before the first run): keep with probability q1 for patients with the outcome, q0 for the others. Strength = q1 / q0 in {1, 1.5, 2, 4}; 1 is the control. Removal only, same patients (invariants 2 and 3). | Delete measurements of hospital-1 test patients until each variable is as frequent as at hospital 2 (D6, D14). | Score the hospital-1 models on the hospital-2 test set. No retraining, no fine-tuning. |
| Training | Each model trained once per strength | none: evaluation of phase-4 models | none: read from phase-4 runs |
| Test conditions | rule on; rule off (same deletion rate for all); same patients | recorded vs thinned, same patients | hospital 1 vs hospital 2 |
| Task | mortality, all three models. Sepsis with GRU only if time is left. | sepsis | sepsis |
| Direction | – | hospital 1 → hospital 2 rates | primary: train hospital 1, test hospital 2. Reverse only in phase 6. |
| Variables | all | (a) all 34 as built (D14); (b) the 25 comparable ones | (a) all 34; (b) the 25 comparable ones |
| Runs | 3 models × 4 strengths × 3 seeds = 36 | 0 | 18 for (b): 3 models × `values`, `all` × 3 seeds, trained on the 25 variables |
| When | after phase 3 | after phase 4 | after phase 4 |
| Always shown next to it | the `values` model under the same rule | the `values` model under the same thinning | the `values` model |

**The 25 comparable variables** (P5): rate ratio between the hospitals within 1/3 and 3. This
leaves out BaseExcess, HCO3, Chloride, EtCO2, TroponinI, FiO2, pH, PaCO2 and PTT. (b) needs
models trained on these 25, because a model trained on 34 still depends on the other 9.

**If only one of E5 and thinning fits:** keep E5. It is the only experiment whose true answer
we know. Thinning costs no training and is already built, so it stays in either case.

## 7. Rules for the runs

1. **Freeze first.** No run counts before the tag `protocol-v1`. A later change to the
   protocol is a decision in `docs/decisions.md`, and the affected runs are repeated.
2. **Validation for choices, test for results.** Tuning and early stopping use the validation
   split only.
3. **Every run is reproducible:** resolved config and git commit saved with it; run name from
   `train.py`.
4. **One table of results.** A script collects every `metrics.json` into one file in
   `results/`. Figures and tables are made from that file only.
5. **Primary measures are fixed** (M1–M4). Anything else in the report is marked as exploratory.
6. **One owner per phase,** one line in `docs/ledger.md` when it starts and when it ends.

## 8. Steps

1. Team: accept or change P1, P4, P5, P6, P8 and the order in section 5. Record them in
   `docs/decisions.md`; update `docs/experiments.md` (add E5, the 25-variable contrast, the
   order) and `docs/work_plan.md`.
2. Phase 0 code, each with a test on synthetic data: cut at ICU hour 59 in the data config;
   paired bootstrap in `eval/`; per-variable matched random removal in `perturb/`; hour-59 check
   in `scripts/check_data.py`; git commit in the saved config; results collector.
3. Tag `protocol-v1`.
4. Phases 1 to 5 in order, each closed with a short note in the weekly handoff.
5. Phase 6 only if time is left.

## 9. Not in this plan

Kidney failure (D12), eICU, Neural CDE, Transformer, raw 2012 time stamps, fully synthetic
data. The correlation across models for RQ3 is replaced by M4.
