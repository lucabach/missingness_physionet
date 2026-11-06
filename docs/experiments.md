# Experiment matrix

All runs: same preprocessing, splits (`splits/*_seed0.json`), tuning budget and evaluation.
Seeds 0, 1, 2. Run names: `<dataset>_<model>_<inputs>_seed<k>` (set by `train.py`).

## E1 Week-1 check (CPU)

| What | Output |
| --- | --- |
| per-variable measurement rate and inter-measurement time, hospital 1 vs 2 | `results/hospital_rates.csv` + figure |
| LR and gradient boosting on pattern-only / values-only / time-only features, both tasks | `results/baselines.csv` |
| SOFA-variable check: drop pattern features of SOFA variables vs same number of others | `results/sofa_check.csv` |

## E2 Information ablation (RQ1)

| Task | Models | Inputs | Seeds | Hospitals trained | Runs |
| --- | --- | --- | --- | --- | --- |
| sepsis | gru, grud, odernn | values, pattern, all | 3 | 2 | 54 |
| mortality | gru, grud, odernn | values, pattern, all | 3 | – | 27 |

Plus tuning: 3 models × 8 configs × 2 tasks = 48 runs. Report S and Δmask per model
(`eval/scores.py`). GRU-D "values" = plain GRU.

## E3 Removal (RQ1 sensitivity, RQ2)

On every trained `all` model and the matching `values` model, for p in {0.1, 0.3, 0.5}:
`python -m missingness.evaluate <run> --perturb random --p <p>` and `--perturb state`.
Sensitivity = extra loss under state-dependent vs random, compared with the values-only model.

## E4 Thinning and transfer (sepsis, RQ2–RQ3)

Train on the hospital that measures more (`data.train_source`, decided in week 1). Evaluate on:
(a) same hospital as recorded, (b) same patients thinned to the other hospital's rates
(`--perturb thin`), (c) the other hospital (`test_hospital2` in `metrics.json`).
Relate (a)−(b) and (a)−(c) losses to sensitivity scores, as an exploratory association.

## Compute budget (proposal table)

~15 min per GRU-type run and ~25 min per ODE-RNN run on PhysioNet 2019, about a third on
2012; core ≈ 35 GPU-hours. Evaluations reuse trained models.
