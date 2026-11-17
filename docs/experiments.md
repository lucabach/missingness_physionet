# Experiment matrix

All runs: same preprocessing, splits (`splits/*_seed0.json`), tuning budget and evaluation.
Seeds 0, 1, 2. Run names: `<dataset>_<model>_<inputs>_seed<k>` (set by `train.py`; the
25-variable runs use the data `tag` instead of the dataset name). Sepsis records are cut at
ICU hour 59 (D15). The order of the runs and the protocol are in the study plan
(`docs/plans/2026-11-13/17-03-StudyPlanTwoTasks/plan.md`, sections 4 and 5): mortality first.
No run counts as a result before the tag `protocol-v1`. Collect results with
`python scripts/collect_results.py` → `results/runs.csv`.

## E1 Week-1 check (CPU)

| What | Output |
| --- | --- |
| per-variable measurement rate and inter-measurement time, hospital 1 vs 2 | `results/hospital_rates.csv` + figure (done 13 Nov) |
| LR and gradient boosting on pattern-only / values-only / time-only features, both tasks | `results/baselines.csv` (redone after the cut, 17 Nov; before the cut: `results/baselines_before_cut.csv`) |
| SOFA-variable check: drop pattern features of SOFA variables vs same number of others | `results/sofa_check.csv` (17 Nov, after the cut; before: `results/sofa_check_before_cut.csv`) |
| How much state-dependent removal can remove, per p; variable mix of each removal method | `results/removal_shares.csv` (17 Nov, `scripts/removal_shares.py`) |

## E2 Information ablation (RQ1)

| Task | Models | Inputs | Seeds | Hospitals trained | Runs |
| --- | --- | --- | --- | --- | --- |
| mortality (first) | gru, grud, odernn | values, pattern, all | 3 | – | 27 |
| sepsis | gru, grud, odernn | values, pattern, all | 3 | hospital 1 (D11) | 27 |

Plus tuning: 3 models × 8 configs × 2 tasks = 48 runs (`slurm/submit_tuning.sh`, grid still
to confirm). Report S and Δmask per model (`eval/scores.py`), Δmask with a paired bootstrap
(`eval.bootstrap_diff`). GRU-D "values" = plain GRU.

## E3 Removal (RQ1 sensitivity, RQ2)

On every trained `all` model and the matching `values` model, for p in {0.05, 0.10, 0.15}
(D17; p = 0.10 is the primary level, measure M2):
`python -m missingness.evaluate <run> --perturb random --p <p>` and `--perturb state`.
Random removal takes the same number of values per variable as state-dependent removal (D16).
Sensitivity (M2) = (loss under state-dependent − loss under random) for `all`, minus the same
for `values`, with a paired bootstrap (`eval.bootstrap_contrast`, weights [1, −1, −1, 1]).

## E4 Thinning and transfer (sepsis, RQ2–RQ3)

Train on the hospital that measures more (`data.train_source`, decided in week 1). Evaluate on:
(a) same hospital as recorded, (b) same patients thinned to the other hospital's rates
(`--perturb thin`), (c) the other hospital (`test_hospital2` in `metrics.json`).
Relate (a)−(b) and (a)−(c) losses to sensitivity scores, as an exploratory association (M4).

Contrast on the 25 variables both hospitals measure at a comparable rate (plan P5, proposed):
3 models × `values`, `all` × 3 seeds = 18 runs with `configs/experiment/sepsis_comparable_<model>.yaml`.

## E5 Planted pattern (plan P4, proposed; code ready, no runs)

On real mortality records, keep the measurements of one variable with probability q1 for
patients who died and q0 for the others (`perturb/planted.py`), in train, validation and test.
Strength q1/q0 ∈ {1, 1.5, 2, 4}; 3 models × 4 strengths × 3 seeds = 36 runs. Set it with a
`planted:` block in the experiment config (`variable`, `q1`, `q0`); test the rule switched off
with `python -m missingness.evaluate <run> --perturb planted_off`. Still to choose: the
variable (proposal HR) and q1 (proposal 1.0, so q0 = 1/strength).

## Compute budget (proposal table)

~15 min per GRU-type run and ~25 min per ODE-RNN run on PhysioNet 2019, about a third on
2012; core ≈ 35 GPU-hours. Evaluations reuse trained models.
