# CLAUDE.md

Instructions for AI coding agents (Claude Code, Codex, Cursor, ...) and a compact reference
for humans. Read this fully before changing anything. `AGENTS.md` points here.

## The project in five lines

ETH Deep Learning HS26 group project "Values or Habits?". In ICU data, *whether and when* a
variable is measured carries information (sicker patients are tested more). We measure
(1) how much predictive information the measurement pattern carries, (2) how sensitive
trained GRU, GRU-D and ODE-RNN models are to changes in it, and (3) whether that sensitivity
goes with larger losses at another hospital. Full proposal summary: `docs/project_overview.md`.

- **Tasks:** causal hourly sepsis prediction (PhysioNet 2019, two hospital systems) and
  in-hospital mortality from the first 48 h (PhysioNet 2012).
- **Core models:** GRU (forward-filled inputs), GRU-D, ODE-RNN. **Extensions only:** Neural
  CDE, Transformer, eICU. Do not grow the core scope without a decision in `docs/decisions.md`.
- **Experiments:** information ablation (values / pattern / all inputs), test-time removal
  (random vs state-dependent), rate-matched thinning of the same patients, cross-hospital
  transfer. Matrix: `docs/experiments.md`.

## Commands

```bash
source .venv/bin/activate                     # setup: see README "Quick start"
pytest                                        # all tests, synthetic data, ~10 s
ruff check . --fix && ruff format .           # lint + format (CI enforces both)
python -m missingness.train configs/experiment/smoke.yaml        # end-to-end smoke run
python -m missingness.train configs/experiment/<task>_<model>.yaml inputs=<values|pattern|all> seed=<0|1|2>
python -m missingness.evaluate outputs/<run> --perturb <random|state|thin> [--p 0.3]
python scripts/prepare_data.py physionet2019 physionet2012       # raw -> processed
```

Before you say a change is done: `ruff check . && ruff format --check . && pytest` must pass.

## Code map

```
src/missingness/
  paths.py            all paths; override with MISSINGNESS_DATA_DIR / MISSINGNESS_OUTPUT_DIR
  config.py           YAML loading: data cfg + model cfg + experiment cfg + CLI overrides
  data/records.py     PatientRecord (the one data format) + processed parquet I/O
  data/physionet2019.py, physionet2012.py   raw-file parsers
  data/preprocessing.py   x (forward fill), m (mask), delta (time gap), Normalizer
  data/splits.py      patient-level stratified splits (committed in splits/)
  data/dataset.py     TimeSeriesDataset + collate -> padded batch dict
  data/synthetic.py   fake data for tests only
  perturb/            random_removal, state_dependent_removal, paired_removal,
                      thin_to_rates, check_only_removed
  models/             build_model registry; inputs.py defines the input variants
  baselines/          classical baselines (week 2, stub)
  eval/               metrics (AUROC, AUPRC, patient bootstrap, pre_onset_mask), scores (S, Δmask)
  experiment.py       load_data / predict / score_predictions shared by train and evaluate
  train.py, evaluate.py   CLI entry points
configs/{data,model,experiment}/   all hyperparameters live here
```

## Core invariants (do not break these)

1. **One data format.** Everything goes through `PatientRecord`: `values` (T, D) with NaN for
   "not measured". Mask and time gaps are always *derived* from the NaNs by
   `preprocessing.featurize`, never stored or edited by hand. A perturbation only sets
   values to NaN, then the inputs are recomputed.
2. **Removal only.** Perturbations may remove existing observations, never add, move or
   change one. Every perturbed test set must pass `perturb.check_only_removed(original,
   perturbed)`: same patients in the same order, same hours, same labels.
3. **Same patients.** Recorded and thinned test sets contain exactly the same patients
   (Sarah's rule). Splits are patient-level, fixed in `splits/`, never regenerated silently.
4. **Causal predictions.** The prediction at hour t may only use data up to t. Models output
   one logit per step `(B, T)`; mortality reads the last observed step via `label_mask`.
5. **Model interface.** A model is `__init__(n_vars, n_static, inputs, **cfg)` +
   `forward(batch) -> logits (B, T)`, registered in `models/__init__.py::MODELS`, with a
   `configs/model/<name>.yaml`. It must support the three input variants of `models/inputs.py`.
6. **Same evaluation for everyone.** Use `eval.bootstrap_ci` (resamples patients, not
   hours) and `experiment.score_predictions`; for sepsis always report pre-onset hours too.
7. **Config, not code.** Hyperparameters and paths go in `configs/` or env vars. A run must
   be reproducible from its saved `outputs/<run>/config.yaml`.

## Terminology and labels (from TA feedback, binding for code, docs, plots and the report)

- Say **"sensitivity to measurement changes"**. Do **not** say "reliance" (too strong).
- The PhysioNet 2019 hospital systems are **hospital 1** (challenge set A) and **hospital 2**
  (set B); in code `hospital1` / `hospital2`. Never call them A/B.
- Never reuse a letter or label for two different things (the draft proposal used "B" twice).
  Keep names unique: RQ1–RQ3, D1… for decisions, hospital 1/2, input variants
  `values` / `pattern` / `all`.
- Removal methods are **random** and **state-dependent**; the rate-matched one is **thinning**.
- Information scores: **S** (pattern share) and **Δmask** (gain from adding mask and gaps).

## Claims and wording (scientific rules from the TA)

- Thinning removes physiological information *and* changes the measurement pattern. Never
  attribute a thinning loss to the pattern alone; compare with the values-only model under
  the same thinning, and state the limitation.
- Δmask is a lower bound (forward-filled values leak timing). For GRU-D, the values-only
  variant is a plain GRU: say so wherever it is reported.
- The sepsis label is computed from lab values (SOFA), so measurement and label are partly
  linked by construction; mortality is the cleaner contrast.
- The transfer analysis (RQ3) is an exploratory association, not a predictive claim.
- Always cite the paper when mentioning SHAP, PhysioNet or MIMIC (and the datasets:
  Reyna et al. 2020 for 2019, Silva et al. 2012 for 2012, Goldberger et al. 2000 for PhysioNet).
  BibTeX keys are in the proposal's `references.bib`.
- Full TA feedback with context: `docs/ta_feedback.md`.

## Data rules

- Never commit data, processed tables, checkpoints, predictions or anything patient-level.
  `data/`, `outputs/` and `*.pt` are git-ignored; do not force-add them.
- Committed exceptions: `splits/*.json` (public record IDs only) and small aggregate
  tables/figures in `results/`.
- Raw data lives in `data/raw/`, processed in `data/processed/<dataset>/timeseries.parquet`
  (format documented in `data/records.py` and `data/README.md`).

## Working conventions

- **Branches and PRs.** Never push to `main` directly; branch as `<name>/<short-topic>`
  (e.g. `matteo/grud`), open a PR, get one review from a teammate. CI must be green.
- **Small PRs** with a run command in the description. Code without a run command is not done.
- **Tests.** New logic gets a test in `tests/` on synthetic data (`data/synthetic.py`); tests
  must not need the real datasets or a GPU.
- **Commits** are made under the human teammate's own name. Do not add AI co-author
  trailers or "Generated with ..." lines to commits or PR descriptions.
- **Decisions.** Any choice that changes results (preprocessing, label, split, metric,
  removal definition, model scope) gets an entry in `docs/decisions.md`.
- **Work ledger.** Before starting, run `bash scripts/ledger.sh` to see what the others
  are working on, on every branch. Before pushing, add your line at the bottom of
  `docs/ledger.md` (one line per piece of work, append only; rules at the top of the file).
- **Weekly relay.** One owner per week; on Sunday the owner writes
  `docs/handoffs/week-NN.md` from the template. Read the latest handoff before starting work.
- **Notebooks** are for exploration; move reusable code into `src/missingness/`. Outputs are
  stripped by the pre-commit hook. Name them `NN_initials_topic.ipynb`.
- Style: ruff (line length 100), type hints on public functions, short docstrings that say
  *what* and *why*. Keep code comments simple.

## Open items an agent should not decide alone

See `docs/decisions.md` → "Open questions" (e.g. final label choice, which hospital trains,
state-dependent removal definition). Ask the human instead of picking silently.
