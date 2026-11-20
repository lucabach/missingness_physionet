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
- 20 Nov: PR #1 merged into `main` (merge commit `21113c2`); GitHub shows PR #4 as merged too,
  its commits were part of PR #1. `matteo/protocol-v1` and the tag `protocol-v1` pushed; PR #5
  opened against `main`. The questions for Sarah were merged with Luca's draft email (same
  numbering as `conclusions.md`, section 7; his baseline numbers updated to after the
  early-stopping fix).
- 20 Nov: PR #5 merged into `main` by Matteo, without a second review, so that `main` is
  ready for Giulio's handoff.
- 20 Nov: the handoff shows the four GRU findings with one figure each (section 2, linked at
  the top); `scripts/plot_findings.py` draws them, with `--model grud|odernn` for week 3.
  Branch `matteo/handoff-figures`, not pushed yet.

## Next

- Luca sends the merged email to Sarah; her answers go to `docs/log/ta_feedback.md`.
- Sunday 22 Nov: handoff call with Giulio; then this plan is done.

## Blockers

- None.

Last update: 2026-11-20
