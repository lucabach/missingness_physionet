Status: active

## Done

- Review of PR #1 (18 Nov): no blocker; tests and lint pass. Findings that matter for the runs
  became steps A1–A8.
- A1–A8 (18–19 Nov): fast bootstrap (same numbers, 10x faster); run names and results table;
  tuning without test sets and `scripts/pick_config.py`; evaluation with saved predictions;
  `scripts/evaluate_runs.py` and `scripts/measures.py`; laptop runners; D20–D23; baselines
  rerun with early stopping on validation patients (gradient boosting at hospital 2: 0.622 →
  0.759; the earlier "transfer loss" was overfitting).
- A9: tag `protocol-v1` (local). Moved on 19 Nov, before it was ever pushed, to `a2f1770` so
  that it includes D24.
- B1–B7 (18–19 Nov): all 104 GRU runs from clean commits in a second worktree, standard
  evaluation, `results/runs.csv`, `results/measures.csv`. Tuned: mortality hidden 128, lr 3e-4,
  weight decay 1e-5; sepsis hidden 64, lr 1e-3, no weight decay; the `values` check stayed
  below 0.01 on both tasks.
- B3b: the planted pattern of D21 saturated (AUROC 0.999); redesigned as D24 and rerun.
- For week 3: `tests/test_models.py` (interface, causality, input variants) and `--models` in
  the result scripts.
- C1–C3: `docs/current/conclusions.md` (results, surprises, six questions for Sarah) and the
  handoff `docs/log/handoff_week_02.md`.
- Docs reorganized into `reference/`, `current/`, `log/` with a map in `docs/README.md`
  (Matteo's request, 19 Nov).

## Next

- Matteo: review and merge PR #1; push `matteo/protocol-v1` and the tag
  (`git push -u origin matteo/protocol-v1 && git push origin protocol-v1`); open its PR.
- Matteo: send the questions of `docs/current/conclusions.md`, section 7, to Sarah.
- Sunday 22 Nov: handoff call with Giulio; then this plan is done.

## Blockers

- None.

Last update: 2026-11-19
