Status: active

## Done
- Review of PR #1 (18 Nov): no blocker; tests (29) and lint pass. Findings that matter for the
  runs are steps A1–A8 of the plan.
- Timing on Matteo's laptop (14 cores, CPU torch): sepsis GRU 11 s per epoch after the cut;
  loading the sepsis data 36 s; scoring one sepsis test set with 1,000 bootstrap draws about
  2 min (the reason for A1).
- The 25 comparable variables are the same before and after the cut (checked 18 Nov).

## Next
- Steps A1–A9, then the runs B1–B7.

## Blockers
- None. PR #1 should be merged by its review (Matteo) so that this branch can follow.

Last update: 2026-11-18
