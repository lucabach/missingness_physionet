# Work plan: one week each

*Kind: living (rewrite in place when owners or scope change). Map of all docs: [docs/README.md](README.md).*

Updated 18 Nov 2026: every GRU run moves to week 2 (they run on a laptop CPU after the cut at
hour 59); Giulio owns week 3 and does GRU-D and ODE-RNN with the same scripts. The method is
the study plan (`docs/plans/2026-11-13/17-03-StudyPlanTwoTasks/plan.md`); this week's plan is
`docs/plans/2026-11-18/23-11-Week2GruRunsAndHandoff/`. The team's
[work-split doc](https://claude.ai/artifact/Tk79134ft9RDc12DqzFJkR) still shows the first
version; its owner should update it. Fill in the owner of week 4.

By the end of week 4 all core experiments should be done (about 75% of the project); the
report, final figures and extensions remain.

| Week | Dates | Owner | Focus | Done when | Code |
| --- | --- | --- | --- | --- | --- |
| 1 | 9–15 Nov | Luca, Matteo | Data pipeline and hospital comparison | **Done 13 Nov** (PR #1): both datasets load; splits committed (D8); outcomes settled (D12); hospitals compared, train on hospital 1 (D11). Data checks that changed the plan: `docs/study_design.md`. | `data/`, `scripts/`, `splits/`, `eval/` |
| 2 | 16–22 Nov | Luca, Matteo | Baselines, removal and thinning, making the pipeline safe (plan phases 0 and 1), **and every GRU run (phases 2, 4 and 5 for GRU)** | 17 Nov (Luca, PR #1): phase 0, gate passed, D15–D19. 18–22 Nov (Matteo): protocol frozen (D20–D23, tag `protocol-v1`); GRU on both tasks: tuning, 9 runs per task, planted pattern (24 runs), 25 variables (6 runs), removal and thinning evaluation, measures M1–M4; first conclusions and questions for Sarah (`docs/conclusions.md`); handoff `docs/handoffs/week-02.md` | `eval/`, `scripts/`, `slurm/`, `configs/` |
| 3 | 23–29 Nov | Giulio | GRU-D and ODE-RNN (plan phases 3 and 4 for these two models) | Both models implemented with the input variants of the plan (section 4), tested on synthetic data (Q5: which implementation); for each model, with the same scripts as the GRU: mortality tuning (8), 9 runs, planted pattern (24); sepsis tuning (8), 9 runs, 25 variables (6); `scripts/evaluate_runs.py` on every run; everything submitted before handing over | `models/`, `slurm/` |
| 4 | 30 Nov–6 Dec | _tbd_ | All core results (plan phase 5) | `scripts/measures.py` for all three models; results table and figures from `results/runs.csv` and `results/measures.csv`; the GRU conclusions in `docs/conclusions.md` extended to GRU-D and ODE-RNN; a two-page results note | `scripts/`, `results/`, `docs/` |
| 5 | | | Only if time is left (plan phase 6): sepsis trained on hospital 2 (27 runs), more seeds where a primary measure is unclear; reruns | | |
| 6 | | | Final analysis, figures, tables | | |
| 7 | | | Full report draft, starting from `docs/study_design.md` | | |
| 8 | | | Report revision, ideally after showing results to Sarah | | |
| 9–10 | | everyone | Review, fixes, submission | | |

## Handover list: ready before week 3 starts (owners of weeks 1 and 2)

The week-3 owner should only have to build models and submit runs. Details and acceptance
checks for each item: `handoff.md` in the plan folder (tasks T1 to T10).

| # | Item | Why the week-3 owner needs it |
| --- | --- | --- |
| 1 | PR #1 and PR #4 merged; the accepted fixes recorded as decisions (D15 and following); `docs/experiments.md` updated | one agreed state of `main` |
| 2 | Sepsis records cut at ICU hour 59, with the hour-59 check in `scripts/check_data.py` (T1, T2) | otherwise every sepsis run carries the artefact and must be repeated |
| 3 | Baselines repeated after the cut and thinned test sets rebuilt (T3) | the gate of phase 1: confirms the cut works |
| 4 | Paired bootstrap, matched random removal, pre-onset numbers with and without the 426 patients (T4, T5, T8) | evaluation is fixed before any deep model is scored |
| 5 | Git commit saved with each run; one script that collects all results (T6) | every run is traceable from the first one |
| 6 | Planted-pattern code, tested, no runs yet (T7) | week 4 only has to run it |
| 7 | The 8 tuning configurations and the 25 comparable variables written into `configs/` | the sweep can be submitted on day one |
| 8 | GRU on mortality run end to end as a pipeline check (T10) | shows the training script and the numbers are sound |
| 9 | Tag `protocol-v1` | from here on runs count as results |

Also before week 3: the week-3 owner tests their cluster login and fills in `docs/cluster.md`.

## What changed against the first version

- Weeks 1 and 2 also carry the fixes from the data checks (plan, section 2). Most of the
  original week-1 and week-2 work was finished on 13 Nov.
- Week 3: mortality before sepsis; sepsis trained on hospital 1 only (27 runs, not 54); 18
  runs on the 25 comparable variables are new.
- Week 4: the planted pattern (E5) is new; the measures are M1 to M4 of the plan.
- Week 5: no Neural CDE, Transformer or eICU extension; the reverse hospital direction if time
  is left.

## Risks

- **The handover list slips → week 3 slips.** Items 2 to 5 come first; item 6 can move to
  week 4 if needed.
- **Cluster queue.** Week 3 submits all its runs before handing over; whoever takes week 3
  tests their cluster login during week 2.
- **Knowledge lost at handoffs.** Sunday note + call are mandatory; no run command, not done.
- **A run made before the tag `protocol-v1`** is a pipeline check and does not count as a result.
- **Week 4 overruns.** Move the planted pattern on sepsis and the 25-variable contrast to week 5.
