# Handoff: start of phase 0 (written 13 Nov 2026, evening)

For an agent with no memory of the earlier sessions. Read `plan.md` in this folder first; this
file says where things stand and what to do next, in order.

## Goal

Carry out the study plan: make the pipeline safe against the problems of section 2 of the plan
(phase 0), repeat the baselines after the cut (phase 1), then start the mortality runs (phase 2).

## Where things stand

| Item | State |
| --- | --- |
| `main` | has the work ledger, `docs/study_design.md` and this plan (PRs #2 and #3, merged by Matteo) |
| PR #4 (`matteo/study-design`) | open, CI green: caveat distinction in `study_design.md`, CLAUDE.md section on plans and ledgers, this handoff |
| PR #1 (`luca_branchmann`) | open, mergeable: splits, data checks, baselines, thinned test sets, D8–D14. **Everything below needs it.** |
| Local branch `matteo/phase0` (not pushed) | `main` + `luca_branchmann` + `matteo/study-design`. Start here. |
| Fixes P1, P4, P5, P6, P8 | Matteo: yes. The rest of the team has not confirmed yet. No D-numbers yet (next free number after PR #1 is D15). |

Verified on `matteo/phase0` on Matteo's machine (13 Nov): `pytest` 21 passed, `ruff` clean;
`scripts/prepare_data.py` and `scripts/check_data.py` give 20,336 + 20,000 patients (8.80 % and
5.71 % septic) and 11,988 stays (14.24 % deaths); a 2-epoch GRU run on mortality reaches test
AUROC 0.86 at 5 s per epoch on CPU.

## Setup on Matteo's machine

- Repo: `~/Documents/ETH/Msc/2 Semester/Deep Learning/missingness_physionet`. Python env: `.venv`
  (CPU-only torch; the machine has no GPU, 14 cores). Use `.venv/bin/python`, `.venv/bin/pytest`,
  `.venv/bin/ruff`.
- Data: `data/raw/physionet2019` and `data/raw/physionet2012` are symlinks into the private repo
  `../dataset_Probe/data/`. Processed tables are in `data/processed/`. Do not delete either.
- Sepsis deep runs belong on the cluster (`docs/cluster.md` is still empty). Mortality runs are
  fast enough locally (a 30-epoch GRU run takes about 3 minutes).

## What can be done now, in this order

Each task: code, a test on synthetic data, `ruff` and `pytest` green, one commit. Branch from
`matteo/phase0` (for example `matteo/phase0-cut`), small PRs.

| # | Task | Where | Done when |
| --- | --- | --- | --- |
| T1 | Cut at ICU hour 59 (P1). Add `max_hour: 59` to `configs/data/physionet2019.yaml`; apply it in `records.load_processed` (drop rows with `hour > max_hour`), so the processed table stays complete. | `data/records.py`, config, `tests/test_loaders.py` | no record has an hour above 59; a record that starts after hour 59 is dropped as a whole and this is counted |
| T2 | Hour-59 check. Print, per hospital, records that run past hour 59 by outcome, and the numbers after the cut. | `scripts/check_data.py` | matches the table below |
| T3 | Phase 1 gate: repeat the baselines after the cut. Rebuild the thinned test sets too (rates change). | `scripts/run_baselines.py`, `scripts/build_thinned_sets.py` (about 15 min CPU) | `results/baselines.csv` updated; the hour-only AUROC (was 0.712 / 0.696) is reported next to the old one. If it has not dropped, stop and look. |
| T4 | Paired bootstrap for the difference between two models on the same resampled patients (P3). | `eval/metrics.py`, `tests/test_metrics.py` | interval of a difference is narrower than the two single intervals suggest on a toy case; identical scores give exactly 0 |
| T5 | Random removal matched per variable (P6): same number of values removed per variable as the state-dependent rule, different hours. | `perturb/removal.py`, `tests/test_perturb.py` | per-variable counts equal; `check_only_removed` passes |
| T6 | Save the git commit in each run's `config.yaml`; a script that collects every `metrics.json` into one table in `results/`. | `train.py` or `config.py`, `scripts/collect_results.py` | one row per run, test set and metric |
| T7 | E5 planted pattern, code only: a removal function with keep probabilities q1 (outcome) and q0 (no outcome) for one variable, a config key to apply it to train, validation and test, and a "rule off" test condition with the same overall rate. | `perturb/`, `experiment.py`, tests | removal only; same patients; with q1 = q0 the mask of that variable is unrelated to the label |
| T8 | Pre-onset numbers with and without the 426 patients who are positive from their first hour (P8). | `experiment.score_predictions` | both reported for sepsis |
| T9 | GRU-D (open question Q5: which implementation to port), with the input variants of the plan, section 4. | `models/grud.py`, smoke test | passes the smoke test for `values`, `pattern`, `all` |
| T10 | Mortality GRU as a pipeline check: tuning sweep and the three input variants (17 runs, under an hour locally). | configs, `outputs/` | test AUROC of `all` between 0.82 and 0.86 or an explanation. **Not a result:** runs count only after the tag `protocol-v1`. |

**Numbers T2 must reproduce** (from the private check, `hour` = `ICULOS`):

| | hospital 1 | hospital 2 |
| --- | --- | --- |
| Records running past hour 59: septic / not septic | 644 / 8 | 387 / 224 |
| Septic patients with a positive hour up to hour 59 | 1,222 of 1,790 | 797 of 1,142 |
| Positive share of hours after the cut | 1.51 % | 1.01 % |
| Share of hours kept | 94.8 % | 94.6 % |

## Needs a human

1. The other three confirm or object to P1, P4, P5, P6, P8 and the order of work (PR #3's list).
2. Merge PR #1 and PR #4. Then record the accepted fixes as D15 and following in
   `docs/decisions.md`, update `docs/experiments.md` (E5, the 25-variable contrast, the order)
   and `docs/work_plan.md` (plan, step 1), and tag `protocol-v1` once phase 0 is merged.
3. The 8 tuning configurations are not defined anywhere. Proposal: hidden size {64, 128} ×
   learning rate {1e-3, 3e-4} × weight decay {0, 1e-5}.
4. The variable for the planted pattern (E5) must be chosen before its first run. Proposal: HR.
5. Cluster access for the sepsis runs.

## Pitfalls already found

- Luca pushes to `luca_branchmann` often. `git fetch` and merge it again before starting, and
  check `docs/decisions.md` there before using a D-number.
- `main` is not protected on GitHub, but the rule is: branch, PR, one review. Matteo merged #2
  and #3 directly; an agent does not merge without being told to.
- Commits under the teammate's name, no AI trailers. One ledger line per piece of work before
  pushing, append only (`docs/ledger.md`).
- Invariants 2 and 3 (removal only, same patients) apply to T5 and T7.
- The pattern features of the baselines (count so far, hours since last) grow with time. After
  the cut the hour of the stay still carries some signal; it will not fall to 0.5.
- The hour-59 counts and the lab-rate tables in `docs/study_design.md` are not reproducible in
  this repo until T2 is done.
- Shell on Matteo's machine is zsh: quote strings such as `"====="` in `echo`.

## When this handoff is finished

Delete this file, record the outcome in `status.md`, and keep `docs/plans/ledger.md` as it is
(the plan stays active until phase 5).
