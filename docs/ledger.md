# Work ledger

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
