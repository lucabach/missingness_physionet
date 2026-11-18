# Week 2 close: protocol, all GRU runs on CPU, first conclusions, handoff to Giulio

Created 18 Nov 2026 by Matteo. Builds on `luca_branchmann` (PR #1) and the study plan
(`docs/plans/2026-11-13/17-03-StudyPlanTwoTasks/plan.md`), which stays the method.

## Goal

By Sunday 22 Nov:

1. The protocol is frozen and tagged `protocol-v1`. Every later run, ours or Giulio's, goes
   through the same scripts, so results are comparable.
2. Every GRU run of the study is done on CPU, for both tasks, and evaluated.
3. The first conclusions and the questions for Sarah are written down.
4. Giulio (week 3) can start from one clean handoff and only has to implement GRU-D and
   ODE-RNN and run the same commands with another model name.

## Decisions taken for this (Matteo, 18 Nov)

- The team's answers are not awaited. Q7 (tuning grid), Q8 (planted pattern) and Q9
  (25-variable contrast) are decided as Luca proposed them (D20–D22).
- Sepsis GRU moves from week 3 to week 2. After the cut a sepsis GRU epoch takes about 11 s on
  a laptop CPU, so the cluster is not needed for it. The cluster was planned when records ran
  up to 336 h.
- The GRU part of the planted pattern (E5) runs in week 2, not week 4.
- Giulio does GRU-D and ODE-RNN on both tasks (weeks 3 and 4 in `docs/work_plan.md`).

## Steps

### A. Protocol (branch `matteo/protocol-v1`, on top of `luca_branchmann`)

Fixes from the review of PR #1 that matter for the runs, plus what comparable runs need.

| # | What | Why |
| --- | --- | --- |
| A1 | Fast paired bootstrap: weighted AUROC/AUPRC on one sort, same resampled patients, same numbers as before (tested against the old code) | Scoring took about 2 min per test set, longer than training |
| A2 | Run names and collector: planted runs get their own name; `results/runs.csv` gets data tag, planted rule, hyperparameters, validation AUROC | A planted run overwrote the plain run; 25-variable runs and tuning runs could not be told apart |
| A3 | Tuning runs do not score the test sets; `scripts/pick_config.py` writes the chosen configuration into `configs/experiment/<task>_<model>.yaml` | Plan rule 2 (test unused until tuning is frozen); config, not code |
| A4 | `evaluate.py`: several perturbations per data load; perturbed predictions saved; planted test sets checked with `check_only_removed` | M2 and M4 need paired intervals from predictions; sepsis loads take 36 s |
| A5 | `scripts/evaluate_runs.py` (the standard evaluation of every run) and `scripts/measures.py` (M1–M4 from saved predictions) | Same evaluation and measures for every model |
| A6 | `LOCAL=1` and `MODELS=` in `slurm/submit_ablations.sh` | Runs on a laptop and on the cluster from the same script |
| A7 | Docs: D20 and following, statuses in the study plan, `docs/experiments.md`, `docs/work_plan.md`, comments in configs and baselines | One agreed state |
| A8 | Baselines: gradient boosting early-stops on validation patients, not random hours; rerun (30 min CPU) | Hours of one patient were in both parts; the gate and S/Δmask numbers come from these baselines |
| A9 | Tag `protocol-v1` on the last commit of A | Plan rule 1 |

### B. Runs (CPU, Thursday to Saturday)

| # | Runs | Command |
| --- | --- | --- |
| B1 | Mortality GRU tuning: 8 on `all`, 8 on `values` (plan check) | `LOCAL=1 bash slurm/submit_tuning.sh mortality gru` |
| B2 | Mortality GRU: 3 inputs × 3 seeds, then the standard evaluation | `LOCAL=1 MODELS=gru bash slurm/submit_ablations.sh mortality` |
| B3 | E5 on mortality, GRU: 4 strengths × 3 seeds, rule off | see `docs/experiments.md` E5 |
| B4 | Sepsis GRU tuning: 8 + 8 | as B1 |
| B5 | Sepsis GRU: 9 runs; removal, thinning, hospital 2 | as B2 |
| B6 | Sepsis GRU on the 25 comparable variables: `values`, `all` × 3 seeds | P5 |
| B7 | `scripts/collect_results.py`, `scripts/measures.py` | results tables |

### C. Writing (Saturday and Sunday)

| # | What |
| --- | --- |
| C1 | `docs/conclusions.md`: first conclusions from the baselines and the GRU (M1–M4), and where the results differ from what we expected |
| C2 | Questions for Sarah: Luca's list plus the new ones from the runs |
| C3 | `docs/handoffs/week-02.md` rewritten as one clean handoff for Giulio |
| C4 | Work ledger lines, this plan's status, PR description with run commands |

## Not in this plan

GRU-D and ODE-RNN (Giulio), sepsis trained on hospital 2 (phase 6), the planted pattern on
sepsis.
