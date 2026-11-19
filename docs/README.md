# Docs map

Every document here is one of three kinds. Check the kind before you edit a file.

| Kind | Rule |
| --- | --- |
| **static** | A record or a reference. Do not rewrite it. If something in it turns out wrong, add a dated note at the top that says what changed and where the current version is. |
| **living** | Always shows the current state. Rewrite it in place, keep it short, and update its date line. Old versions are in git. |
| **log** | Append only. Add at the bottom, or add a new file; never edit, reorder or delete an old entry. Correct an entry with a new one. |

## Where to look for what

| You want | Read |
| --- | --- |
| What the project is, in plain language | `project_overview.md` |
| Why two tasks, and text for the report | `study_design.md` |
| **Where things stand: results, surprises, questions for Sarah** | `conclusions.md` |
| **What the next person does** | the latest `handoffs/week-NN.md` |
| How the runs are done (method, protocol, measures, phases) | the study plan, `plans/2026-11-13/17-03-StudyPlanTwoTasks/plan.md` |
| Who does what in which week | `work_plan.md` |
| Which runs exist and the commands | `experiments.md` |
| Why something is done the way it is | `decisions.md` |
| Who did what, on which branch | `ledger.md` |

## Every file

| File | What it is | Kind | Who updates it, when |
| --- | --- | --- | --- |
| `project_overview.md` | the revised proposal of 6 Nov: story, glossary, RQ1–RQ3 | static | only after a scope decision with a D-number |
| `study_design.md` | the argument for two tasks, the caveats, a draft paragraph for the report | static | its numbers in section 8 are from before the cut; current numbers are in `conclusions.md` |
| `ta_feedback.md` | what Sarah said, one section per meeting | log | after every meeting with Sarah |
| `references.bib` | BibTeX keys for the report | log | when a new source is cited |
| `decisions.md` | D1, D2, ...: every choice that changes results | log | with the change; supersede, never renumber |
| `ledger.md` | the work ledger: one line per piece of work | log | before every push (rules at its top) |
| `handoffs/week-NN.md` | what a week did and what the next owner does | log | one new file per week, written on Sunday, frozen after the handoff call |
| `handoffs/week-02-phase0-luca.md` | Luca's phase-0 handoff of 17 Nov | static | kept as written |
| `handoffs/TEMPLATE.md`, `handoffs/README.md` | how to write a handoff | static | – |
| `conclusions.md` | results so far, what differs from what we expected, questions for Sarah | living | whoever adds results (week 4: all three models) |
| `work_plan.md` | owners and goals of each week | living | when an owner or a week's scope changes |
| `experiments.md` | run matrix, run counts, commands, compute budget | living | when a run type is added or a timing is measured |
| `cluster.md` | how to run on the ETH cluster | living | whoever first uses the cluster |
| `plans/ledger.md` | index of plans: Active, Noteworthy finished | living | when a plan starts, ends or is dropped |
| `plans/<day>/<time-name>/plan.md` | goal, approach, steps of one plan | living while active, static when done | the plan's owner |
| `plans/<day>/<time-name>/status.md` | done, next, blockers of that plan | living | as the work moves |
| `plans/<day>/<time-name>/handoff.md` | notes for the next session of that plan | living until closed, then static (kept, marked closed) | the session that stops |

## Three plan documents, three jobs

- `work_plan.md` says **who and when** (one owner per week). It is older than the plans folder.
- The **study plan** (`plans/2026-11-13/17-03-StudyPlanTwoTasks/`) says **how**: the fixes
  after the data checks, one protocol for both tasks, the measures M1–M4, the phases with
  their gates. It stays active until every phase is done.
- A **work plan for one stretch of work** (for example
  `plans/2026-11-18/23-11-Week2GruRunsAndHandoff/`) lists the steps of that stretch and is
  closed when they are done.

The `plans/` folder (one folder per plan, with `plan.md` and `status.md`, indexed in
`plans/ledger.md`) is the convention that `CLAUDE.md` gives every agent.
