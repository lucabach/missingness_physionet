# Work plan: one week each

Mirrors the team's work-split doc (https://claude.ai/artifact/Tk79134ft9RDc12DqzFJkR). If the
two disagree, update both. Fill in the owners.

By the end of week 4 all core experiments should be done (about 75% of the project); the
report, final figures and extensions remain.

| Week | Dates | Owner | Focus | Done when | Code |
| --- | --- | --- | --- | --- | --- |
| 1 | 9–15 Nov | _tbd_ | Data pipeline and hospital comparison | Both datasets load into `PatientRecord` (values, mask, time gap); patient splits committed; label choice settled (sepsis / mortality / kidney failure); evaluation code checked; hospital 1 vs 2 measurement rates compared, with a note on which measures more | `data/`, `scripts/`, `splits/`, `eval/` |
| 2 | 16–22 Nov | _tbd_ | Cheap baselines, removal and thinning | LR and gradient boosting on pattern-only, values-only, time-only for both tasks; SOFA-variable check; removal (random, state-dependent) and thinning reviewed, tested; thinned sepsis test sets built | `baselines/`, `perturb/` |
| 3 | 23–29 Nov | _tbd_ | Deep models, tuning, all training runs | GRU, GRU-D, ODE-RNN in one training script on the cluster; tuning sweep (8 configs per model) done; all 81 ablation runs submitted (54 sepsis, 27 mortality) | `models/`, `slurm/` |
| 4 | 30 Nov–6 Dec | _tbd_ | All core results | Ablations collected, S and Δmask; removal experiments and sensitivity for both tasks; thinned and cross-hospital evaluation for sepsis; result figures and a two-page results note | `evaluate.py`, `results/` |
| 5 | | | One extension (Neural CDE, Transformer or eICU) and reruns | | |
| 6 | | | Final analysis, figures, tables | | |
| 7 | | | Full report draft | | |
| 8 | | | Report revision, ideally after showing results to Sarah | | |
| 9–10 | | everyone | Review, fixes, submission | | |

## Risks

- **Week 1 slips → everything slips.** Week 1 ships a minimal working loader by Wednesday.
- **Cluster queue.** Week 3 submits *all* ablation runs before handing over; whoever takes
  week 3 tests their cluster login during week 2.
- **Knowledge lost at handoffs.** Sunday note + call are mandatory; no run command, not done.
- **Hospitals measure alike.** If week 1 finds little difference, tell Sarah at once and
  decide between thinning only and applying for eICU access.
- **Week 4 overruns.** Move the mortality sensitivity analysis to week 5.
