# Contributing

How the four of us work in this repo. The short version: branch, small PR, one review,
green CI, write it down.

## Setup (once)

See README "Quick start". Then `pre-commit install`, so lint and notebook-output stripping
run on every commit.

## Weekly relay

We split the core work into one-week packages (plan and owners: `docs/work_plan.md`).

- **One owner per week, Monday to Sunday.** The owner does the main work and decides small
  questions alone. Bigger ones go in `docs/decisions.md`.
- **Definition of done** is the deliverable list for that week. Anything not done by Sunday
  is written down as an open item, not passed on silently.
- **Sunday handoff.** The owner merges their work, writes `docs/handoffs/week-NN.md` from
  `docs/handoffs/TEMPLATE.md`, and holds a 20-minute call with the next owner.
- **Off-week jobs:** review the owner's PRs within a day, read 1–2 benchmark papers, finish
  the PhysioNet credentialing (CITI training) so eICU stays possible as a fallback.

## Git workflow

1. `git switch main && git pull`, then `bash scripts/ledger.sh` to see what the others are
   working on (all branches).
2. `git switch -c <name>/<topic>` (e.g. `giulio/thinning`)
3. Commit small, meaningful steps. Message: imperative, one line, e.g. `Add rate-matched thinning`.
4. `ruff check . && ruff format --check . && pytest` locally.
5. Add your line to `docs/ledger.md` (rules at the top of that file).
6. Push and open a PR into `main`; fill in the template, including how to run it.
7. One teammate reviews; CI must be green; the author merges (squash merge).

Never push to `main` directly, never commit data or checkpoints, never force-push a shared
branch.

## Adding things

- **A model:** class in `src/missingness/models/<name>.py` following the interface in
  CLAUDE.md, register it in `MODELS`, add `configs/model/<name>.yaml` and
  `configs/experiment/<task>_<name>.yaml`, extend the smoke test.
- **An experiment:** a config in `configs/experiment/`, a row in `docs/experiments.md`.
- **A perturbation:** a function in `src/missingness/perturb/` returning new records; a test
  that it passes `check_only_removed`.
- **A decision:** a numbered entry in `docs/decisions.md` (date, owner, what, why).

## Results

Raw runs go to `outputs/` (ignored). When a number or figure is final enough for the
report, save the aggregate table/figure in `results/` with the script or notebook that made
it, and note the run names it came from.

## Using AI agents

Agents read `CLAUDE.md`. Review agent-written code like any other PR. If an agent breaks an
invariant or a terminology rule, fix CLAUDE.md so the next one does not.
