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
