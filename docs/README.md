# Docs map

The one place that says how the documents work. `CLAUDE.md` (read by every agent; `AGENTS.md`
points to it) repeats the rules in short.

## Three folders, three rules

| Folder | Kind | Rule |
| --- | --- | --- |
| `reference/` | **static** | Background and records that do not change: the proposal, the study design, the handoff template, the bibliography. Do not rewrite them. If something in one turns out wrong, add a dated note at the top or at the section that says what changed and where the current version is. |
| `current/` | **living** | Always shows the current state. Rewrite in place, keep it short, update its date line. Old versions are in git. |
| `log/` | **log** | Append only. Add a row, a section or a new file at the end; never edit, reorder or delete an old entry. Correct an entry with a new one. |
| `plans/` | mixed | The plans, in the layout every agent uses (one folder per plan, see below). |

When a file moves, fix the paths in `reference/`, `current/`, `plans/` and the code. Entries
in `log/` keep the paths they were written with; the table at the end maps old paths to new
ones. (Editing a line of `log/work_ledger.md` would also make the next merge keep both
versions of it.)

## Where to look for what

| You want | Read |
| --- | --- |
| **Where things stand: results, surprises, questions for Sarah** | `current/conclusions.md` |
| **What the next person does** | the newest `log/handoff_week_NN.md` |
| What the project is, in plain language | `reference/project_overview.md` |
| Why two tasks, and text for the report | `reference/study_design.md` |
| How the runs are done: method, protocol, measures M1–M4, phases | the study plan, `plans/2026-11-13/17-03-StudyPlanTwoTasks/plan.md` |
| Who does what in which week | `current/work_plan.md` |
| Which runs exist, run counts, commands, compute | `current/experiments.md` |
| Why something is done the way it is (D1, D2, ...) | `log/decisions.md` |
| Who did what, on which branch | `log/work_ledger.md` (all branches: `bash scripts/ledger.sh`) |

## Every file

| File | What it is | Who updates it, when |
| --- | --- | --- |
| `reference/project_overview.md` | the revised proposal of 6 Nov: story, glossary, RQ1–RQ3 | only after a scope decision with a D-number |
| `reference/study_design.md` | why two tasks, the caveats, a draft paragraph for the report | dated notes only; its section 8 has the numbers from before the cut |
| `reference/handoff_template.md` | the skeleton of a weekly handoff | – |
| `reference/references.bib` | BibTeX keys for the report | add an entry when a new source is cited |
| `current/conclusions.md` | results so far, what differs from what we expected, questions for Sarah | whoever adds results (week 4: all three models) |
| `current/work_plan.md` | owner and goal of each week | when an owner or a week's scope changes |
| `current/experiments.md` | run matrix, run counts, commands, compute budget | when a run type is added or a timing is measured |
| `current/cluster.md` | how to run on the ETH cluster | whoever first uses the cluster |
| `log/decisions.md` | every choice that changes results, numbered D1, D2, ... (rows of one table; the newest at the bottom) | with the change; supersede an entry with a new one, never renumber |
| `log/work_ledger.md` | one line per piece of work: who, branch, outcome | before every push (rules at its top) |
| `log/ta_feedback.md` | what Sarah said, one section per meeting | after every meeting with Sarah |
| `log/handoff_week_NN.md` | what a week did and what the next owner does | the week's owner, on Sunday: copy `reference/handoff_template.md`, fill it in, link it in the 20-minute handoff call. Frozen after the call |
| `log/handoff_week_02_phase0_luca.md` | Luca's phase-0 handoff of 17 Nov | kept as written |

## Plans

```text
plans/
  ledger.md                        index of plans: Active, Noteworthy finished (living)
  YYYY-MM-DD/HH-MM-ShortName/      day and time the plan was created
    plan.md                        goal, approach, steps (living while active, then static)
    status.md                      first line "Status: active | done | dropped" (living)
    handoff.md                     only while a session hands the plan to another; deleted when done
```

There are three planning documents and each has one job:

- `current/work_plan.md` says **who and when**: one owner per week.
- The **study plan** (`plans/2026-11-13/17-03-StudyPlanTwoTasks/`) says **how**: the fixes after
  the data checks, one protocol for both tasks, the measures M1–M4, the phases with their
  gates. It stays active until every phase is done.
- A **plan for one stretch of work** (for example `plans/2026-11-18/23-11-Week2GruRunsAndHandoff/`)
  lists the steps of that stretch and is closed when they are done.

## Moved on 19 Nov

Older documents, ledger lines and commit messages use the old paths.

| Old path | New path |
| --- | --- |
| `docs/project_overview.md`, `study_design.md`, `references.bib` | `docs/reference/` (same names) |
| `docs/handoffs/TEMPLATE.md` | `docs/reference/handoff_template.md` |
| `docs/work_plan.md`, `experiments.md`, `cluster.md`, `conclusions.md` | `docs/current/` (same names) |
| `docs/decisions.md`, `ta_feedback.md` | `docs/log/` (same names) |
| `docs/ledger.md` (the work ledger) | `docs/log/work_ledger.md` |
| `docs/handoffs/week-NN.md` | `docs/log/handoff_week_NN.md` |
| `docs/handoffs/week-02.md` of 17 Nov (Luca) | `docs/log/handoff_week_02_phase0_luca.md` |
| `docs/handoffs/README.md` | merged into this file |
| `docs/plans/2026-11-13/17-03-StudyPlanTwoTasks/handoff.md` | deleted when closed (19 Nov), as the plan rules say; in git: `git show 22c2229:docs/plans/2026-11-13/17-03-StudyPlanTwoTasks/handoff.md` |
