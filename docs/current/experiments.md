# Experiment matrix

*Kind: living (rewrite in place when a run type or timing changes). Map of all docs: [docs/README.md](../README.md).*

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

Plus tuning (D20): 3 models × 8 configs × 2 tasks = 48 runs, and 8 more per task for the GRU
`values` check. `slurm/submit_tuning.sh <task> <model>`, then `python scripts/pick_config.py
<task> <model> --write` freezes the choice in the experiment config. Then
`slurm/submit_ablations.sh <task>` (`MODELS=gru LOCAL=1` for one model on a laptop).
Report S and Δmask per model (M1, `scripts/measures.py`). GRU-D "values" = plain GRU.

## E3 Removal (RQ1 sensitivity, RQ2)

On every trained `all` model and the matching `values` model, for p in {0.05, 0.10, 0.15}
(D17; p = 0.10 is the primary level, measure M2), removal seed 0 for every model (D23):
`python scripts/evaluate_runs.py` runs these for every finished run.
Random removal takes the same number of values per variable as state-dependent removal (D16).
Sensitivity (M2) = (loss under state-dependent − loss under random) for `all`, minus the same
for `values`, with a paired bootstrap (`eval.bootstrap_contrast`, weights [1, −1, −1, 1]).

## E4 Thinning and transfer (sepsis, RQ2–RQ3)

Train on the hospital that measures more (`data.train_source`, decided in week 1). Evaluate on:
(a) same hospital as recorded, (b) same patients thinned to the other hospital's rates
(`--perturb thin`), (c) the other hospital (`test_hospital2` in `metrics.json`).
Relate (a)−(b) and (a)−(c) losses to sensitivity scores, as an exploratory association (M4).

Thinning seeds 0, 1, 2 for every sepsis `all` and `values` run (`scripts/evaluate_runs.py`).

Contrast on the 25 variables both hospitals measure at a comparable rate (D22): 3 models ×
`values`, `all` × 3 seeds = 18 runs, `INPUTS="values all" bash slurm/submit_ablations.sh
sepsis_comparable` (uses the sepsis tuned configuration).

## E5 Planted pattern (plan P4, D21, D24)

On real mortality records, keep the HR measurements with probability q1 = 0.8 for patients
who died and q0 = 0.8/strength for the others (`perturb/planted.py`), in train, validation
and test. Strength {1, 1.05, 1.1, 1.2}; `all` and `values` × 4 strengths × 3 seeds = 24 runs
per model (`slurm/submit_planted.sh`, `MODELS=gru LOCAL=1` for one model). Run names carry
the rule (`physionet2012_gru_all_plantedHR-q0.8-s1.1_seed0`). Rule off (`planted_off`) is
evaluated by `scripts/evaluate_runs.py`.

The first design (D21: q1 = 1.0, strength {1, 1.5, 2, 4}) saturates: the GRU finds the rule
perfectly at every strength (test AUROC 0.999), so it has no dose-response. Its 24 GRU runs
stay as the saturated end (`Q1=1.0 STRENGTHS="1 1.5 2 4"`); the other models skip it (D24).

## E6 Pattern-blind reference for RQ2 (D29, proposed)

Forward fill lets the `values` model rebuild the mask from value changes, so it is not a clean
reference for M2. Plan: `docs/plans/2026-11-28/15-50-RQ2PatternFreeReference/`.

```bash
python scripts/leak_audit.py --coverage       # bin coverage per width, fix the snapshot block first
python scripts/leak_audit.py                  # -> results/leak_audit/leak_audit.csv (no training)
python scripts/evaluate_runs.py --channel     # channel intervention on the forward-fill `all` runs
python scripts/leak_audit.py --print-commands # 18 decoded-pattern + 36 snapshot training runs
python scripts/evaluate_runs.py --runs '*_snapshot_*'
```

Run names carry the view: `physionet2012_gru_values_snapshot_seed0`,
`physionet2012_gru_pattern_decoded_seed0`. Measures: `M1_decoded_gap`, `M1_bound_snapshot`,
`M2_ref_snapshot`, `M4_thinning_ref_snapshot`, `M2_channel_pattern|values`,
`M4_thinning_channel_pattern|values`; the snapshot check is M1 with `view = snapshot`.

## Standard evaluation and measures (D23)

```bash
python scripts/evaluate_runs.py      # removal, thinning, rule off for every finished run
python scripts/collect_results.py    # every score -> results/runs.csv
python scripts/measures.py           # M1-M4 -> results/measures.csv
```

Each measure is a mean over seeds with a paired patient bootstrap (definitions in the
docstring of `scripts/measures.py` and the study plan, section 4).

## Compute budget

Proposal: ~15 min per GRU-type run and ~25 min per ODE-RNN run on PhysioNet 2019, about a
third on 2012; core ≈ 35 GPU-hours. Measured 18 Nov on a laptop CPU (14 cores) after the cut:
sepsis GRU 11 s per epoch, mortality GRU about 4 s; loading the sepsis data 36 s; scoring one
sepsis test set 13 s. Every GRU run fits on a laptop: the 104 GRU runs of week 2 and their
standard evaluation take roughly 7 hours of one chain (two chains in parallel slow each
other down 2–3x); `scripts/measures.py` takes about 20 minutes per model.

## Predictive performance across seeds

After the Week 3 runs, `python scripts/select_predictor.py mortality` (or `sepsis`) compares
the six full-input checkpoints, the two within-model seed averages, and mixtures that give
GRU-D 25%, 50%, or 75% of the weight. Each member predicts a probability. Selection maximizes
validation AUROC over stays for mortality and over pre-onset hours at hospital 1 for sepsis.
The selected mixture is fixed for all test hospitals. Its report in
`results/predictor_<task>.json` includes a paired patient-bootstrap interval for the test AUROC
gain over the individual checkpoint selected on the same validation set.

To predict on additional records with the saved checkpoints, pass the report to
`missingness.models.predictor.load_predictor`. It returns a model and normalizer for the
usual `missingness.experiment.predict` interface. Move the model to the prediction device first.

The first mortality sweep configuration (hidden size 64, batch size 64) took about 2 seconds
per epoch for GRU-D and 2–3 seconds for ODE-RNN on the Week 3 laptop CPU, including validation.
