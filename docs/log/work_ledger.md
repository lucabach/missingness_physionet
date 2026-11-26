# Work ledger

*Kind: log (append one line per piece of work). Map of all docs: [docs/README.md](../README.md).*

One line per piece of work: who did it, on which branch, and what came out of it. It lets us
see what the four of us (and our agents) are working on without reading every branch. It is an
index, not a report: details go in the PR, `docs/decisions.md`, `docs/handoffs/` or `results/`.

## Rules

1. **Look before you start.** Run `bash scripts/ledger.sh` (it lists the lines of all
   branches, not only yours) and check whether someone is already on your topic. If so, talk
   to them before you start.
2. **One line per piece of work**, added at the bottom before you push. A piece of work is
   one topic on one branch, however many commits or agent sessions it took. Never one line
   per commit.
3. **`start`, `done`, `dropped`.** Work that is finished at its first push gets a single
   `done` line. Longer work gets a `start` line at its first push and a `done` or `dropped`
   line when it ends.
4. **Write the outcome.** A `done` line says what was found or built, not what was tried:
   "non-septic records end at ICU hour 59", not "looked at record lengths". At most 25 words,
   plus one pointer in `where` (PR, decision number or file).
5. **Append only.** Never edit, reorder or delete a pushed line, yours or anyone else's. To
   correct a line, add a new one.
6. **Merging.** `.gitattributes` sets `merge=union` for this file, so a merge keeps the lines
   of both branches without a conflict. If GitHub still reports a conflict here, run
   `git merge origin/main` on your branch (this resolves it) and push. When two branches
   worked on the same topic, whoever unites them adds one `merged` line with the joint
   conclusion and which version was kept.
7. **Size.** If the table has more than 60 lines at a Sunday handoff, the week's owner moves
   the lines older than two weeks to `docs/ledger_archive.md`. Nobody else removes lines.

Columns: `who` is the teammate's first name, with `+agent` if an AI agent did most of the
work. `area` is one of `data`, `perturb`, `models`, `baselines`, `eval`, `runs`, `analysis`,
`docs`, `infra`.

## Ledger

| date | who | branch | area | status | what | where |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-11-13 | matteo+agent | matteo/ledger | docs | done | Work ledger with union merge, and `scripts/ledger.sh` to list the lines of all branches | this file |
| 2026-11-13 | matteo+agent | (private probe repo) | analysis | start | Check of the proposal against the 2019 and 2012 data: non-septic records end at ICU hour 59, at most 10 positive hours per patient; report follows | – |
| 2026-11-13 | matteo+agent | matteo/study-design | docs | done | Why two tasks (label and cohort caveats of the sepsis data) and a proposed study plan: 10 problems with fixes, one protocol, run order | `docs/study_design.md`, `docs/plans/` |
| 2026-11-17 | luca+agent | luca_branchmann | data | done | Phase 0 + gate: cut at hour 59 kills the clock (time-only 0.71→0.58); after it pattern-only 0.70 vs values 0.72 (sepsis, hospital 1) | D15–D19, `docs/handoffs/week-02.md` |
| 2026-11-18 | matteo+agent | matteo/protocol-v1 | runs | start | Protocol frozen as `protocol-v1` (D20–D23); all GRU runs on CPU. Baseline fix: the hospital-2 drop of gradient boosting was early-stopping leakage | D20–D23, `docs/plans/2026-11-18/` |
| 2026-11-19 | matteo+agent | matteo/protocol-v1 | runs | done | All 104 GRU runs on CPU: pattern mostly redundant, no removal sensitivity (M2 ≈ 0), planted pattern redesigned and found, more transfer loss with pattern | `docs/current/conclusions.md` |
| 2026-11-20 | matteo+agent | matteo/handoff-figures | docs | done | Week-2 handoff shows the four GRU findings with one figure each; `scripts/plot_findings.py --model <m>` draws the same for GRU-D and ODE-RNN | `docs/log/handoff_week_02.md` |
| 2026-11-20 | giulio+agent | giulio/week3 | models | start | Implemented GRU-D and ODE-RNN for the three input variants in the shared training pipeline | D25–D26 |
| 2026-11-20 | giulio+agent | giulio/week3 | runs | done | Both deep models completed eight configurations per task and the full 96-run post-tuning matrix | D25–D26 |
| 2026-11-24 | luca+agent | claude/project-thread-uxhuch | perturb | start | Whole-busy-hour removal (state vs matched random) at 20/25/30% for all three models; code and test in, runs pending | D27 (proposed) |
| 2026-11-24 | luca+agent | claude/project-thread-uxhuch | runs | done | Cloud retrain of the 36 core runs reproduces the week-3 measures; whole-busy-hour removal at 20/25/30% gives M2 ≈ 0 (2 of 18 sepsis checks significant, no trend) | D27 (proposed), `results/cloud_retrain_2026-11-24/` |
| 2026-11-26 | luca+agent | claude/project-thread-td20j4 | runs | done | SOFA check for GRU, GRU-D, ODE-RNN (63 runs): hiding the SOFA pattern costs +0.009 to +0.015 before onset at hospital 1; beyond random draws only for GRU-D (both hospitals) and GRU (hospital 2) | D28 (proposed), `results/sofa_check_neural/` |
