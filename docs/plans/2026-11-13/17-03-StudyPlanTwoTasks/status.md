Status: active

## Where the phases stand (section 5 of the plan)

| Phase | What | State |
| --- | --- | --- |
| 0 | Fixes P1, P3, P6, P8; protocol; tag `protocol-v1` | **done**: code by Luca 17 Nov (D15–D19); protocol and tag by Matteo 18 Nov (D20–D23) |
| 1 | Classical baselines after the cut | **done**, gate passed (time-only 0.712 → 0.583). Rerun 18 Nov after the early-stopping fix |
| 2 | GRU on mortality: tuning, 9 runs | **done** 19 Nov. Gate: `all` test AUROC 0.870, slightly above the expected 0.82–0.86 |
| 3 | GRU-D and ODE-RNN: implement, mortality | not started: Giulio, week 3 (`docs/log/handoff_week_02.md`) |
| 4 | Sepsis, trained on hospital 1 | GRU: done or running (plan `2026-11-18/23-11-Week2GruRunsAndHandoff`); GRU-D, ODE-RNN: week 3 |
| 5 | E5 planted pattern, thinning, transfer | GRU: E5 run twice (D21 saturated, redesigned as D24), thinning and transfer evaluated with the sepsis runs; other models: weeks 3–4 |
| 6 | Optional: sepsis trained on hospital 2, more seeds | not started |

## Changes to the plan since it was written

- P4 (planted pattern) redesigned: D21 → D24, because the first rule gave the label away.
- The handoff of 13 Nov is closed and deleted, as the plan rules say (in git:
  `git show 22c2229:docs/plans/2026-11-13/17-03-StudyPlanTwoTasks/handoff.md`): T1–T8 done by
  Luca, T10 (tuning sweep) done for the GRU, T9 (GRU-D) moved to Giulio's week 3.

## Next

- Phases 3–5 for GRU-D and ODE-RNN (Giulio, week 3), then measures for all three models
  (week 4).

## Blockers

- None. PR #1 and the `matteo/protocol-v1` PR should be merged before week 3 starts.

Last update: 2026-11-19
