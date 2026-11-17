# Work plan: one week each

Updated 13 Nov 2026 to follow the study plan
(`docs/plans/2026-11-13/17-03-StudyPlanTwoTasks/plan.md`): what each week does, in which order,
and what must be ready before the next owner starts. The team's
[work-split doc](https://claude.ai/artifact/Tk79134ft9RDc12DqzFJkR) still shows the first version; its owner
should update it. Fill in the owners of weeks 3 and 4.

By the end of week 4 all core experiments should be done (about 75% of the project); the
report, final figures and extensions remain.

| Week | Dates | Owner | Focus | Done when | Code |
| --- | --- | --- | --- | --- | --- |
| 1 | 9–15 Nov | Luca, Matteo | Data pipeline and hospital comparison | **Done 13 Nov** (PR #1): both datasets load; splits committed (D8); outcomes settled (D12); hospitals compared, train on hospital 1 (D11). Data checks that changed the plan: `docs/study_design.md`. | `data/`, `scripts/`, `splits/`, `eval/` |
| 2 | 16–22 Nov | Luca, Matteo | Baselines, removal and thinning, **and making the pipeline safe (plan phases 0 and 1)** | Done 13 Nov (PR #1): baselines, SOFA check, removal review, thinned test sets. **Still to do before 22 Nov:** the handover list below. | `baselines/`, `perturb/`, `eval/`, `data/` |
| 3 | 23–29 Nov | _tbd_ | Deep models, tuning, training runs (plan phases 2 to 4) | GRU-D and ODE-RNN in the one training script, with the input variants of the plan (section 4); tuning sweep on the validation split (8 configs per model and task); **mortality first:** 27 runs, `all` AUROC checked against published numbers; **then sepsis trained on hospital 1:** 27 runs, plus 18 runs on the 25 comparable variables; everything submitted before handing over | `models/`, `slurm/` |
| 4 | 30 Nov–6 Dec | _tbd_ | All core results (plan phase 5) | Results collected into one table; measures M1 to M4 with paired intervals; removal evaluation for both tasks; planted pattern on mortality (36 short runs, then rule on / rule off); thinned and hospital-2 evaluation for sepsis; result figures and a two-page results note | `evaluate.py`, `perturb/`, `results/` |
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
