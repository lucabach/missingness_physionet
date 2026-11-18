Status: active

## Done
- Review of PR #1 (18 Nov): no blocker; tests and lint pass. Findings that matter for the runs
  became steps A1–A8.
- A1–A8 done (18–19 Nov): fast bootstrap (same numbers, 10x faster); run names and results
  table; tuning without test sets and `scripts/pick_config.py`; evaluation with saved
  predictions; `scripts/evaluate_runs.py` and `scripts/measures.py`; laptop runners;
  D20–D23; baselines rerun with early stopping on validation patients.
- A9: tag `protocol-v1` on the commit that adds this status (local; Matteo pushes it).
- Baseline rerun changed the picture at hospital 2: gradient boosting `all` 0.622 → 0.759.
  The earlier "transfer loss" of the baselines was overfitting (`results/baselines.csv`).
- Timing on Matteo's laptop: sepsis GRU 11 s per epoch; sepsis data load 36 s; scoring one
  sepsis test set 13 s; mortality GRU about 4 s per epoch.

## Next
- B1–B7: runs from a separate worktree at the tag (`../missingness_physionet_runs`, detached),
  so every run records a clean commit while the docs keep changing here. Outputs go to this
  checkout's `outputs/`.
- C1–C4 once the GRU numbers are in.

## Blockers
- None. PR #1 should be merged (Matteo reviews) before this branch's PR.

Last update: 2026-11-19
