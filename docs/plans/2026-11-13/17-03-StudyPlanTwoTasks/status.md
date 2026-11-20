Status: active

## Done
- Plan written as a proposal (13 Nov), with `docs/study_design.md` as the argument for two tasks.
- Matteo accepts all proposed fixes (P1, P4, P5, P6, P8) and the order of work (13 Nov).
- Environment and data prepared on Matteo's machine; pipeline checked on real data (see handoff).

- Phase 0 code done on `luca_branchmann` (17 Nov, Luca): T1–T8 with tests; P1, P3, P6, P8
  recorded as D15–D19 (Luca confirmed P1 and P6; the matched removal and p levels as D16–D17).
- Phase 1 gate passed (17 Nov): time-only AUROC 0.712 → 0.582 (gradient boosting, hospital 1).
  Baselines, SOFA check and thinned sets redone after the cut. Details:
  `docs/handoffs/week-02.md`.

## Next
- Review and merge PR #1 (`luca_branchmann`, contains `main` and PR #4).
- Team answers Q7 (tuning grid), Q8 (planted pattern, P4) and Q9 (25 variables, P5) in
  `docs/decisions.md`; then tag `protocol-v1`.
- T9 (GRU-D) and the T10 tuning sweep: week-3 owner.

## Blockers
- No training run counts before the team has accepted the protocol and the tag `protocol-v1`
  exists.
- PR #1 not merged yet (needs one review).

Last update: 2026-11-17
