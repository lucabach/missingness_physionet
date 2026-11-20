# Week 2 handoff (owners: Luca, Matteo; dates: 16–22 Nov)

Written 17 Nov by Luca. This is the document to read before taking over the repository in
week 3. It says what was inspected, what we found, what is decided, what is still open, the
trade-offs we see, and the questions for Sarah. All work is on branch `luca_branchmann`
(PR #1), which also contains `main` and Matteo's `matteo/study-design` (PR #4).

## 1. In five lines

1. The data pipeline, splits, classical baselines and all evaluation tools are done and tested
   (29 tests on synthetic data, CI green).
2. Sepsis records are now **cut at ICU hour 59** (D15). Before the cut, the hour of the stay
   gave away the label; most of our earlier "the pattern beats the values" result was that.
3. After the cut the measurement pattern still predicts sepsis well (pattern-only AUROC 0.70,
   values-only 0.72, both 0.74 at hospital 1), but it no longer beats the values.
4. The removal experiment is fixed: random removal is matched per variable (D16), p ∈ {0.05,
   0.10, 0.15} with 0.10 as the primary level (D17), and differences use a paired bootstrap (D19).
5. The week-3 owner can start the deep models. Before that, the team must confirm four proposed
   items (section 6) and tag `protocol-v1`.

## 2. Where the repository stands

| Item | State |
| --- | --- |
| `main` | setup, work ledger, `docs/study_design.md`, study plan (PRs #2, #3) |
| PR #1 `luca_branchmann` | everything in this document; contains `main` and PR #4. **Merge this first**, then PR #4 has nothing left to add |
| PR #4 `matteo/study-design` | study-design caveats, CLAUDE.md section on plans, plan handoff (already merged into PR #1) |
| Active plan | `docs/plans/2026-11-13/17-03-StudyPlanTwoTasks/` (`plan.md`, `status.md`, `handoff.md`) |
| Decisions | D1–D19 in `docs/decisions.md`; open questions Q5, Q7–Q9 |
| Data | not in git. `bash scripts/download_data.sh` then `python scripts/prepare_data.py physionet2019 physionet2012` (about 30 min) |

## 3. What was inspected, and what we found

### 3.1 The data (13 Nov, `scripts/check_data.py`)

- PhysioNet 2019 (Reyna et al. 2020): 40,336 patients, hospital 1 20,336 (8.8% septic),
  hospital 2 20,000 (5.7%). Numbers match the published ones.
- PhysioNet 2012 (Silva et al. 2012): 11,988 stays after dropping 12 without any measurement
  (D10), 14.2% deaths.
- The two hospitals measure differently: hospital 1 measures 23 of 34 variables more often
  (BaseExcess 45x, HCO3 42x, Chloride 13x); hospital 2 measures EtCO2 (never at hospital 1),
  TroponinI and Glucose more (`results/hospital_rates.csv`). We train on hospital 1 (D11).

### 3.2 The hour-59 artefact (found by Matteo, checked and fixed 17 Nov)

Almost no record of a patient without sepsis goes past ICU hour 59. This comes from how the
challenge data was assembled, not from medicine.

| | hospital 1 | hospital 2 |
| --- | --- | --- |
| Records past hour 59: septic / not septic | 644 / 8 | 387 / 224 |
| Septic patients with a positive hour up to hour 59 | 1,222 of 1,790 | 797 of 1,142 |
| Positive share of hours after the cut | 1.51% | 1.01% |
| Hours kept | 94.8% | 94.6% |
| Septic test patients with a positive hour (before → after) | 269 → 175 | 172 → 121 |

`python scripts/check_data.py physionet2019` prints this table; it matches Matteo's private
check exactly. Three hospital-1 records start after hour 59 and drop out whole.

**Gate of plan phase 1 passed:** the time-only baseline fell from 0.712 to 0.582 (gradient
boosting) and from 0.668 to 0.508 (logistic regression) at hospital 1. It does not fall to 0.5
because the hour still carries a little real signal (sepsis risk changes over the stay).

### 3.3 Classical baselines before and after the cut (`results/baselines.csv`)

Test AUROC, sepsis trained on hospital 1, one seed, default settings. Before the cut in
brackets (`results/baselines_before_cut.csv`). Mortality is not affected by the cut.

| Features | GB hospital 1 | GB hospital 2 | LR hospital 1 | LR hospital 2 | Mortality GB | Mortality LR |
| --- | --- | --- | --- | --- | --- | --- |
| pattern only | 0.696 (0.793) | 0.669 (0.744) | 0.692 (0.729) | 0.654 (0.490) | 0.770 | 0.758 |
| values only | 0.722 (0.750) | 0.602 (0.666) | 0.680 (0.703) | 0.637 (0.671) | 0.853 | 0.851 |
| time only | 0.582 (0.712) | 0.586 (0.696) | 0.508 (0.668) | 0.482 (0.624) | – | – |
| all | 0.744 (0.814) | 0.622 (0.753) | 0.743 (0.773) | 0.689 (0.665) | 0.866 | 0.860 |

95% intervals are about ±0.03 at hospital 1 and ±0.05 at hospital 2. What this means:

- **The pattern still matters for sepsis, but less.** Gradient boosting, hospital 1:
  S ≈ 0.80 (was 0.93), Δmask ≈ +0.02 (was +0.06). The logistic regression gives Δmask ≈ +0.06.
- **Mortality is unchanged:** values dominate (S ≈ 0.74, Δmask ≈ +0.01).
- **The transfer collapse was the clock.** Before the cut the pattern-only logistic regression
  fell to 0.49 at hospital 2. Hospital 2 keeps many non-septic patients past hour 59, so "late
  hour = sepsis" was wrong there. After the cut it goes from 0.69 to 0.65, a normal loss.
- **Odd result to watch:** at hospital 2 gradient boosting on all features (0.622) is worse than
  on the pattern alone (0.669). It seems to overfit hospital-1 value features that transfer
  badly. One seed, untuned: do not read more into it yet.
- Pre-onset hours give almost the same numbers. Without the 426 patients who are positive from
  their first hour (D18) the numbers drop by about 0.01–0.03; the time-only logistic regression
  jumps around there (0.44 to 0.63 at hospital 2), a small-sample effect.

### 3.4 Is the sepsis pattern just the label's own lab tests? (`results/sofa_check.csv`)

The sepsis label comes from SOFA, a score built from lab values. We removed the pattern
features of the five SOFA variables and compared with removing five random other variables.

| Gradient boosting, all features | hospital 1 | hospital 2 |
| --- | --- | --- |
| Nothing removed | 0.744 | 0.622 |
| SOFA pattern removed | 0.727 | 0.654 |
| 5 random variables removed (5 draws) | 0.731 to 0.749 | 0.620 to 0.661 |

At hospital 1 the SOFA pattern matters a little more than random variables (−0.017 vs 0 to
−0.013): part of the pattern signal is linked to the label by construction, most is not. At
hospital 2 the draws vary too much to tell. This caveat stays in the report; mortality is the
clean contrast.

### 3.5 Removal: how much can we remove? (`results/removal_shares.csv`)

State-dependent removal (D5) deletes the extra measurements in hours when a patient is
measured more than their usual (median) number per hour, i.e. the extra tests ordered when
they got worse. Only some values qualify:

| Test set | Share of values that qualify | Patients with enough for p = 0.10 | for p = 0.15 | Achieved at p = 0.15 |
| --- | --- | --- | --- | --- |
| Sepsis hospital 1 | 23.9% | 99% | 92% | 0.149 |
| Sepsis hospital 2 | 17.2% | 88% | 56% | 0.133 |
| Mortality | 20.5% | 98% | 86% | 0.146 |

p = 0.3 (the proposal's value) reaches only 17% of hospital-1 patients, so it is out. The
qualifying values are mostly labs: vital signs are 45% of them at hospital 1 against 75% of all
values. Uniform random removal would therefore remove different *variables*, not only different
*hours*. The matched random removal (D16) takes the same count per variable; the table shows
identical variable mixes for both methods.

### 3.6 Thinning (`results/thinning_rates.csv`)

Rebuilt after the cut: hospital-1 test patients thinned to hospital-2 rates, 3,051 patients,
15.5% of values removed, 23 of 34 variables within 0.002 of the target rate, seeds 0–2. The
other 11 variables are measured less at hospital 1 and cannot be raised by removal (D11).

### 3.7 Pipeline check on real data (T10, partial)

Not a result (before `protocol-v1`, untuned, one seed); it only shows that training, the
removal evaluation and the collector work on real data. GRU on mortality, default settings,
early stopping after 13 epochs, about 4 s per epoch on 4 CPUs:

| GRU, mortality | clean | random removal p = 0.10 | state-dependent p = 0.10 |
| --- | --- | --- | --- |
| `all` | 0.865 [0.84, 0.89] | 0.860 | 0.861 |
| `values` | 0.858 [0.84, 0.88] | 0.851 | 0.851 |

The `all` AUROC lands at the top of the expected 0.82–0.86 range. Achieved removal fraction
0.0997. Mortality GRU shows almost no sensitivity to either removal, which fits its small
Δmask (+0.007). Commands: `python -m missingness.train configs/experiment/mortality_gru.yaml
run_name=pipeline_check_mortality_gru_all`, then `python -m missingness.evaluate
outputs/pipeline_check_mortality_gru_all --perturb state` (and `random`).

## 4. What was built (17 Nov), all with tests on synthetic data

| Task | What | Where |
| --- | --- | --- |
| T1 | Cut at hour 59, `max_hour` in the data config, applied when loading | `data/records.py`, `configs/data/physionet2019.yaml` |
| T2 | Hour-59 table | `scripts/check_data.py` |
| T3 | Baselines, SOFA check and thinned sets redone after the cut | `results/` |
| T4 | Paired bootstrap for differences between models | `eval.bootstrap_diff`, `eval.bootstrap_contrast` |
| T5 | Random removal matched per variable; `evaluate.py` default p = 0.10 | `perturb/removal.py` |
| T6 | Git commit saved in every run's `config.yaml`; results collector | `train.py`, `scripts/collect_results.py` → `results/runs.csv` |
| T7 | Planted pattern (E5): rule on in train/val/test via a `planted:` config block; rule off via `evaluate --perturb planted_off` | `perturb/planted.py`, `experiment.py`, `evaluate.py` |
| T8 | Pre-onset numbers without the patients positive from their first hour | `experiment.score_predictions` |
| item 7 | 25 comparable variables as a data config + three experiment configs; tuning sweep script (grid proposed) | `configs/data/physionet2019_comparable.yaml`, `slurm/submit_tuning.sh` |
| Q4 evidence | Removal shares per p | `scripts/removal_shares.py` |

## 5. Decisions taken this week

- **D15** cut at ICU hour 59. **D16** matched random removal. **D17** p ∈ {0.05, 0.10, 0.15},
  primary 0.10. **D18** pre-onset with and without the early-positive patients. **D19** paired
  bootstrap for differences. Details and numbers in `docs/decisions.md`.

## 6. Still open

### Needs the team (before the tag `protocol-v1`)

| # | Item | Proposal | Who |
| --- | --- | --- | --- |
| Q7 | Tuning grid (8 configurations) | hidden {64, 128} × lr {1e-3, 3e-4} × weight decay {0, 1e-5}, as in `slurm/submit_tuning.sh` | team |
| Q8 | Planted pattern (P4, E5): run it at all? which variable and q1? | yes; HR on mortality; q1 = 1.0, q0 = 1/strength | team, ask Sarah |
| Q9 | 25-variable transfer contrast (P5): 18 extra sepsis runs | yes, if the cluster budget allows | team |
| Q5 | Which GRU-D and ODE-RNN implementations to port | week-3 owner proposes | week-3 owner |
| – | Week-3 and week-4 owners | fill in `docs/work_plan.md` | team |
| – | Cluster login tested, `docs/cluster.md` filled | – | week-3 owner |

### Handover list (`docs/work_plan.md`) status

| # | Item | State |
| --- | --- | --- |
| 1 | PR #1 and #4 merged; fixes recorded; `docs/experiments.md` updated | decisions and docs done; **merge PR #1 after one review** |
| 2 | Cut at hour 59 + check | done |
| 3 | Baselines repeated, thinned sets rebuilt | done, gate passed |
| 4 | Paired bootstrap, matched removal, pre-onset with/without | done |
| 5 | Git commit per run, results collector | done |
| 6 | Planted-pattern code | done (no runs) |
| 7 | Tuning configurations and 25 variables in `configs/` | done as proposals (Q7, Q9) |
| 8 | GRU on mortality end to end (T10) | partly: one `all` and one `values` run, see 3.7; the 8-config sweep needs Q7 |
| 9 | Tag `protocol-v1` | **not done**: after PR #1 is merged and Q7–Q9 are answered |

## 7. Things to consider, and the trade-offs

1. **Cut at 59 vs a fixed 48-hour window.** We cut at 59 because that is where the artefact
   is. Cutting sepsis at 48 h would make both tasks use the same window (mortality uses the
   first 48 h), at the price of more hours and more septic patients lost. 59 keeps
   more data; 48 is easier to explain. We chose 59; worth asking Sarah.
2. **The cut costs positives.** 35% of septic test patients at hospital 1 (30% at hospital 2)
   have all their positive hours after hour 59 and now count as negatives up to hour 59. Their
   labels are still correct hour by hour, but the intervals are wider. This is why D19 matters.
3. **Small removal fractions.** p ≤ 0.15 gives small perturbations, so the M2 effects may be
   small and need the paired bootstrap and three seeds to show. A larger p would need a new
   state-dependent rule (e.g. removing whole busy hours), which is a different experiment.
4. **Matched random removal is stricter.** It isolates timing from variable choice, but if a
   model is sensitive mainly because it needs labs, M2 will now show less. That is the honest
   reading, and the report should say so.
5. **The time signal is not zero after the cut** (0.58 time-only). Pattern features such as
   "count so far" grow with time, so some of the pattern signal is still the clock. Report the
   time-only baseline next to Δmask.
6. **Planted pattern (E5) costs 36 runs** but is the only experiment whose answer we know. The
   plan says: if only one of E5 and thinning fits, keep E5 (thinning costs no training anyway).
7. **Hospital 2 numbers are noisy** (about 120 septic test patients). Treat transfer results as
   exploratory (CLAUDE.md), with paired intervals.
8. **Tuning once on `all`** and reusing it for `values` and `pattern` is cheaper but could favour
   `all`. The plan checks this with one extra sweep for GRU `values`.

## 8. What the week-3 owner does, in order

1. `git fetch && git checkout main && git pull` after PR #1 is merged; `bash scripts/ledger.sh`;
   read this file and the plan's `status.md`.
2. Get the data (section 2) and run `pytest` and `python scripts/check_data.py physionet2019`
   (the hour-59 table must match section 3.2).
3. Once Q7–Q9 are answered: record them in `docs/decisions.md`, tag `protocol-v1`.
4. Phase 2: `bash slurm/submit_tuning.sh mortality gru`, pick the best by validation AUROC,
   then the 9 GRU runs (`bash slurm/submit_ablations.sh mortality` covers all three models).
   Check: `all` test AUROC about 0.82–0.86.
5. Phase 3: implement GRU-D and ODE-RNN (Q5) with the input variants of plan section 4, smoke
   test, then the same on mortality and the removal evaluation at p = 0.10.
6. Phase 4: sepsis on hospital 1. Every run is scored on both hospitals automatically.
7. After each batch: `python scripts/collect_results.py`, one line in `docs/ledger.md`.

## 9. Questions for Sarah

1. **Hour-59 artefact.** In PhysioNet 2019 almost no non-septic record goes past ICU hour 59,
   so the hour of the stay predicts the label (AUROC 0.67 from the hour alone). We cut every
   record at hour 59. Is that an acceptable fix, or would you prefer a fixed 48-hour window for
   both tasks?
2. **What is left of the pattern effect.** After the cut, pattern-only features reach 0.70 and
   values-only 0.72 for sepsis (all 0.74); for mortality 0.77 vs 0.85. Is this still a strong
   enough starting point for the project, given that the headline "pattern beats values" no
   longer holds?
3. **Size of the removal.** Our state-dependent rule can only remove about 17–24% of values, so
   we use p ∈ {0.05, 0.10, 0.15} instead of 0.3. Is a small perturbation fine for RQ2, or should
   we design a rule that allows larger changes?
4. **Fair random control.** We now make random removal delete the same number of values per
   variable as state-dependent removal, so the two differ only in timing. Do you agree this is
   the right control?
5. **Planted pattern.** We plan an experiment where we delete one variable's measurements more
   often for patients who survive, so that we know the true pattern effect, and check that our
   sensitivity measure finds it (36 short mortality runs). Is it worth the compute, or should
   we spend it elsewhere?
6. **Label link.** Removing the pattern of the five SOFA variables costs 0.017 AUROC at hospital
   1, slightly more than random variables. Is reporting this check together with the mortality
   contrast enough to address the label-construction concern?
7. **Scope.** With the cut, the planted pattern and the 25-variable transfer contrast, the core
   is about 160 runs (with tuning). If we run short, which should go first: ODE-RNN, the 25-variable
   contrast, or the planted pattern?

## 10. How to rerun everything

```bash
python scripts/check_data.py physionet2019 physionet2012   # data table + hour-59 table
python scripts/run_baselines.py                             # baselines + SOFA check, ~30 min on 4 CPUs
python scripts/build_thinned_sets.py --seeds 0 1 2          # thinned hospital-1 test sets, ~2 min
python scripts/removal_shares.py                            # removal fractions, ~2 min
python scripts/collect_results.py                           # all runs in outputs/ -> results/runs.csv
pytest && ruff check . && ruff format --check .
```
