# Work plan: one week each

*Kind: living (rewrite in place when owners or scope change). Map of all docs: [docs/README.md](../README.md).*

Updated 19 Nov 2026: every GRU run moves to week 2 (they run on a laptop CPU after the cut at
hour 59); Giulio owns week 3 and does GRU-D and ODE-RNN with the same scripts. The method is
the study plan (`docs/plans/2026-11-13/17-03-StudyPlanTwoTasks/plan.md`); this week's plan is
`docs/plans/2026-11-18/23-11-Week2GruRunsAndHandoff/`. The team's
[work-split doc](https://claude.ai/artifact/Tk79134ft9RDc12DqzFJkR) still shows the first
version; its owner should update it. Fill in the owner of week 4.

By the end of week 4 all core experiments should be done (about 75% of the project); the
report, final figures and extensions remain.

| Week | Dates | Owner | Focus | Done when | Code |
| --- | --- | --- | --- | --- | --- |
| 1 | 9–15 Nov | Luca, Matteo | Data pipeline and hospital comparison | **Done 13 Nov** (PR #1): both datasets load; splits committed (D8); outcomes settled (D12); hospitals compared, train on hospital 1 (D11). Data checks that changed the plan: `docs/reference/study_design.md`. | `data/`, `scripts/`, `splits/`, `eval/` |
| 2 | 16–22 Nov | Luca, Matteo | Baselines, removal and thinning, making the pipeline safe (plan phases 0 and 1), **and every GRU run (phases 2, 4 and 5 for GRU)** | 17 Nov (Luca, PR #1): phase 0, gate passed, D15–D19. 18–22 Nov (Matteo): protocol frozen (D20–D23, tag `protocol-v1`); GRU on both tasks: tuning, 9 runs per task, planted pattern (24 runs), 25 variables (6 runs), removal and thinning evaluation, measures M1–M4; first conclusions and questions for Sarah (`docs/current/conclusions.md`); handoff `docs/log/handoff_week_02.md` | `eval/`, `scripts/`, `slurm/`, `configs/` |
| 3 | 23–29 Nov | Giulio | GRU-D and ODE-RNN (plan phases 3 and 4 for these two models) | Both models implemented with the input variants of the plan (section 4), tested on synthetic data (Q5: which implementation); for each model, with the same scripts as the GRU: mortality tuning (8), 9 runs, planted pattern (24); sepsis tuning (8), 9 runs, 25 variables (6); `scripts/evaluate_runs.py` on every run; everything submitted before handing over | `models/`, `slurm/` |
| 4 | 30 Nov–6 Dec | _tbd_ | All core results (plan phase 5) | `scripts/measures.py` for all three models; results table and figures from `results/runs.csv` and `results/measures.csv`; the GRU conclusions in `docs/current/conclusions.md` extended to GRU-D and ODE-RNN; a two-page results note | `scripts/`, `results/`, `docs/` |
| 5 | | | Only if time is left (plan phase 6): sepsis trained on hospital 2 (27 runs), more seeds where a primary measure is unclear; reruns | | |
| 6 | | | Final analysis, figures, tables | | |
| 7 | | | Full report draft, starting from `docs/reference/study_design.md` | | |
| 8 | | | Report revision, ideally after showing results to Sarah | | |
| 9–10 | | everyone | Review, fixes, submission | | |

## Handover to week 3

The list of what had to be ready before week 3 (protocol, cut, baselines, evaluation,
traceability, planted-pattern code, tuning configs, a GRU check, the tag `protocol-v1`) is
done; what Giulio starts from and does is in `docs/log/handoff_week_02.md`.

## Changes against the first version (6 Nov)

GRU runs moved from week 3 to week 2; sepsis is trained on hospital 1 only (27 runs, not 54);
the planted pattern (E5) and the 25-variable contrast are new; the extension of week 5 is
dropped. The first version: `git show 5718369:docs/work_plan.md`.

## Risks

- **The handover list slips → week 3 slips.** Items 2 to 5 come first; item 6 can move to
  week 4 if needed.
- **Cluster queue.** Week 3 submits all its runs before handing over; whoever takes week 3
  tests their cluster login during week 2.
- **Knowledge lost at handoffs.** Sunday note + call are mandatory; no run command, not done.
- **A run made before the tag `protocol-v1`** is a pipeline check and does not count as a result.
- **Week 4 overruns.** Move the planted pattern on sepsis and the 25-variable contrast to week 5.
