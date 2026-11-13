# Values or Habits? (missingness_physionet)

ETH Deep Learning HS26 group project: **how sensitive are clinical time-series models to
changes in how patients are measured?**

In intensive-care data most values are missing, and *what* gets measured reflects clinical
decisions: sicker patients are tested more often. A model can therefore predict an outcome
partly from **whether and when** something was measured instead of from the value itself.
We measure how much predictive information that measurement pattern carries, how sensitive
trained GRU, GRU-D and ODE-RNN models are to changes in it, and how they behave when
measurements are removed from the *same* patients or when the model moves to another
hospital.

Team: Parthiv Kangsabanik, Matteo Pelossi, Giulio Rebuffo, Luca Bachmann.
Advisor TA: Sarah Martinson.

New here? Read [docs/project_overview.md](docs/project_overview.md) (plain-language summary
and glossary), then [CONTRIBUTING.md](CONTRIBUTING.md) (how we work together).
Using an AI coding agent? It should read [CLAUDE.md](CLAUDE.md) first.

## Quick start

```bash
git clone https://github.com/lucabach/missingness_physionet.git
cd missingness_physionet

# Python 3.10+; with uv (fast) ...
uv venv && source .venv/bin/activate && uv pip install -e ".[dev]"
# ... or with plain pip
python -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]"

pre-commit install          # lint + strip notebook outputs on every commit
pytest                      # ~10 s, synthetic data only, no download needed
python -m missingness.train configs/experiment/smoke.yaml   # 2-epoch end-to-end run
```

## Get the data (not in git)

Both datasets are open access on PhysioNet (no credentialing needed):

```bash
bash scripts/download_data.sh                          # -> data/raw/
python scripts/prepare_data.py physionet2019 physionet2012   # -> data/processed/
```

Details, sizes and the processed format: [data/README.md](data/README.md).
Patient splits are fixed and committed in [splits/](splits/).

## Run experiments

```bash
# train one model on one input variant (values | pattern | all) and seed
python -m missingness.train configs/experiment/sepsis_gru.yaml inputs=pattern seed=1

# re-evaluate a trained run with measurements removed (no retraining)
python -m missingness.evaluate outputs/physionet2019_gru_all_seed0 --perturb random --p 0.3
python -m missingness.evaluate outputs/physionet2019_gru_all_seed0 --perturb state  --p 0.3
python -m missingness.evaluate outputs/physionet2019_gru_all_seed0 --perturb thin

# on the cluster
sbatch slurm/train.sbatch configs/experiment/sepsis_gru.yaml inputs=all seed=0
bash slurm/submit_ablations.sh sepsis
```

Any config value can be overridden on the command line (`train.lr=3e-4`). Each run writes
its resolved config, checkpoint, metrics and predictions to `outputs/<run_name>/`.
The full experiment matrix is in [docs/experiments.md](docs/experiments.md).

## Where everything is

| Path | What |
| --- | --- |
| `src/missingness/data/` | PhysioNet loaders, `PatientRecord` format, preprocessing (x, mask, delta), splits, torch dataset |
| `src/missingness/perturb/` | removing observations: random, state-dependent, rate-matched thinning, invariant checks |
| `src/missingness/models/` | GRU (done), GRU-D and ODE-RNN (week 3), input variants for the ablation |
| `src/missingness/baselines/` | logistic regression / gradient boosting checks (week 2) |
| `src/missingness/eval/` | AUROC/AUPRC, patient bootstrap CIs, pre-onset hours, information scores S and Δmask |
| `src/missingness/train.py`, `evaluate.py` | entry points (`python -m missingness.train ...`) |
| `configs/` | YAML configs: `data/`, `model/`, `experiment/` |
| `scripts/` | data download and preparation |
| `slurm/` | cluster job templates |
| `splits/` | committed patient splits (same patients for everyone) |
| `tests/` | pytest suite on synthetic data, runs in CI |
| `notebooks/` | exploration only, outputs stripped |
| `results/` | small final tables and figures for the report (raw runs stay in `outputs/`) |
| `docs/` | overview, study design (why two tasks), work plan and plans, experiments, decision log, work ledger, TA feedback, weekly handoffs, cluster notes |
| `CLAUDE.md` / `AGENTS.md` | instructions for AI coding agents |

## Status

Week-by-week plan and owners: [docs/work_plan.md](docs/work_plan.md). Latest handoff note:
[docs/handoffs/](docs/handoffs/).

| Component | State |
| --- | --- |
| Data format, preprocessing, splits | written and unit-tested on synthetic files; **not yet run on the real data** (week 1) |
| Evaluation (AUROC/AUPRC, bootstrap, pre-onset) | done, tested |
| Removal and thinning | first version, tested for invariants; definitions to review in week 2 |
| GRU + training/evaluation scripts | done, smoke-tested |
| Classical baselines | stub (week 2) |
| GRU-D, ODE-RNN | stubs (week 3) |
| Neural CDE, Transformer | extensions (week 5), not started |
