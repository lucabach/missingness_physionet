# Study design: why the same analysis runs on sepsis and on mortality

Reference for the report (sections on data, method and limitations). It gives the argument, the
numbers behind it and a draft paragraph. Decided: two tasks (D2). Marked *proposed*: not yet
agreed, see `docs/plans/2026-11-13/17-03-StudyPlanTwoTasks/plan.md`.

## 1. What we want to show

Clinicians measure sicker patients more often. A model can therefore predict an outcome partly
from the **measurement pattern** (which variable was measured when) and not only from the
values. We ask how much information the pattern carries, how sensitive models are to changes
in it, and whether that sensitivity goes with a loss at another hospital (RQ1–RQ3).

The mechanism we mean is: **patient gets worse → clinician measures more → pattern carries
information about the patient.** Every claim in the report is about this mechanism.

## 2. Why we start with sepsis

Early sepsis prediction on PhysioNet 2019 (Reyna et al. 2020) is the natural first task:

- hourly records and an hourly label, so prediction is causal (hour t uses data up to t);
- two hospital systems, the only open dataset where we can test a real change of hospital;
- earlier work reports that the mask alone predicts sepsis almost as well as the full input
  (Nie et al. 2026; number still to be checked in the paper), which is the effect we want to
  examine.

## 3. Two caveats of the sepsis data

In this dataset the pattern can predict the label for two reasons that are **not** the
mechanism of section 1. A good pattern-only score cannot tell the three apart.

| | Label caveat | Cohort caveat |
| --- | --- | --- |
| In one sentence | The label is built from clinicians' actions. | Patients without sepsis who stay long are missing from the dataset. |
| Why | Sepsis-3 onset (Singer et al. 2016) is the earlier of (a) the time cultures or IV antibiotics were ordered and (b) the time the SOFA score rose by 2 points. Four of the six SOFA parts need a lab result. | Almost no record of a patient without sepsis continues past ICU hour 59. |
| Effect on the pattern | The hour the label turns on is, by definition, an hour around which tests were ordered. "Tests ordered" predicts the label even if tests were ordered at random. | (i) The hour of the stay alone predicts the label. (ii) "Not septic" means "left the ICU early", so heavy monitoring looks more specific to sepsis than it is. |
| Evidence | Around onset the lab rate rises to 1.7–1.8 times that of comparable patients without sepsis (hospital 2: in the 6 h before onset; hospital 1: after onset). | Records past ICU hour 59: hospital 1, 644 septic and 8 not; hospital 2, 387 and 224. A model that only knows the hour of the stay reaches AUROC 0.71 (hospital 1, gradient boosting, `results/baselines.csv`). |
| Can we fix it? | No. The order times are not released, so its size cannot be measured. | Part (i): yes, cut every record at ICU hour 59 (*proposed*). Part (ii): no. |

**Same illness, different label time** (label caveat): two identical patients get worse at
10:00. For one, cultures and labs are ordered at 11:00, for the other at 17:00. The first is
labelled septic from 11:00, the second from 17:00. The label follows the clinician's action.

## 4. What we do inside the sepsis data

| Step | Removes | Status |
| --- | --- | --- |
| Cut every record at ICU hour 59 (training, validation, test, both hospitals) | cohort caveat, part (i) | *proposed* |
| Report a model that sees only the hour of the stay | shows what remains of part (i) | done before the cut, repeat after |
| Report pre-onset hours next to all hours | the heavy testing after onset | done (invariant 6) |
| Check the SOFA variables' pattern against other variables | bounds one part of the label caveat | done: dropping it costs 0.011 AUROC at hospital 1 (`results/sofa_check.csv`) |

What remains after these steps: the label caveat as a whole, and part (ii) of the cohort
caveat. Neither can be removed with PhysioNet 2019 alone.

## 5. The second task: in-hospital mortality

We run the same analysis on PhysioNet 2012 (Silva et al. 2012): in-hospital death, predicted
from the first 48 h in the ICU.

| | Sepsis (PhysioNet 2019) | Mortality (PhysioNet 2012) |
| --- | --- | --- |
| Outcome | hourly label, 1 from 6 h before Sepsis-3 onset | died in hospital, one label per stay |
| Outcome built from measurements or orders | yes | no |
| Cohort depends on the outcome | yes (section 3) | no: every stay is at least 48 h and we use the same 48 h for all |
| Patients | 40,336 in two hospital systems | 11,988 in one hospital (D10) |
| Patients with the outcome | 2,932 (7.3 %); 2,019 have a positive hour up to ICU hour 59 | 1,707 (14.2 %) |
| Share of positive labels | 1.8 % of hours (1.3 % after the cut) | 14.2 % of stays |
| Variables | 34, on an hourly grid | 37, put on an hourly grid (D3) |
| Change of hospital possible | yes | no |

The two shares of positive labels cannot be compared: one counts hours, the other stays. The
number of patients with the outcome is similar in both tasks.

**What mortality adds.** Death is not defined by a measurement or an order, and survivors and
non-survivors were selected in the same way. If the pattern predicts death, the label caveat
and the cohort caveat cannot be the reason.

**What sepsis adds.** Two hospitals. Only there can we study a change of hospital (RQ3).

## 6. Same method on both tasks

Same code, models, input variants (`values`, `pattern`, `all`), preprocessing, tuning budget,
seeds, removal experiments and statistics. Only the label format differs: one label per hour
for sepsis, one per stay for mortality. We never compare raw scores between the tasks. We
compare **differences inside each task**: what the pattern adds to the values (Δmask), and how
much a model loses when measurements are removed. Details: the plan, section 4.

## 7. How to read the results

| What we see | What we may say |
| --- | --- |
| The pattern matters in both tasks | The effect is not explained by label construction or cohort selection alone. Strongest result. |
| It matters for sepsis, little for mortality | Much of the sepsis effect may come from how the label and the dataset are built. This would also put the large mask-only results of earlier work in a different light. |
| It matters in neither | No clear effect under causal prediction. A valid outcome. |

With sepsis alone the first two rows cannot be told apart.

## 8. First evidence

Classical baselines on causal hand-made features (`scripts/run_baselines.py`, test AUROC,
gradient boosting, sepsis trained and tested on hospital 1; before the cut at hour 59):

| Features | Sepsis | Mortality |
| --- | --- | --- |
| values only | 0.750 | 0.853 |
| pattern only | 0.793 | 0.770 |
| hour of the stay only | 0.712 | – |
| all | 0.814 | 0.866 |
| **gain from the pattern (all − values)** | **+0.064** | **+0.013** |

- The pattern is informative in both tasks, and adds much more for sepsis.
- For sepsis the hour of the stay alone is almost as good as the values. The pattern features
  (count so far, hours since last measurement) also grow with time, so part of the pattern's
  gain may be the clock. This is why the baselines must be repeated after the cut.
- One seed, default settings: a first indication, not a result for the report.

## 9. Limits of this design

- **Mortality is another dataset.** Other patients, other years, one hospital. A difference
  between the tasks points to the two caveats but does not prove them.
- **Different label format.** One label per hour against one per stay. This is why we compare
  differences inside each task.
- **Part (ii) of the cohort caveat stays** in every sepsis result.
- **Hourly grid.** Both datasets are on an hourly grid, so we study the measurement pattern,
  not irregular sampling times.

## 10. Draft paragraph for the report

> We first study early sepsis prediction on PhysioNet 2019 (Reyna et al. 2020), which provides
> hourly records from two hospital systems. Two properties of this dataset limit what a result
> on it can show. First, the label follows the Sepsis-3 definition (Singer et al. 2016): onset
> is set by the time clinicians ordered cultures or antibiotics and by a rise in the SOFA
> score, most of whose components require laboratory results. The label is therefore partly
> built from the same clinical actions that form the measurement pattern. Second, the released
> cohort is selected: almost no patient without sepsis has a record beyond ICU hour 59 (8 of
> 652 such records in hospital 1), so the comparison group consists of short stays. Either
> property could make the measurement pattern predictive of the label without the mechanism we
> are interested in, namely that clinicians measure sicker patients more often. We remove the
> part we can by cutting all records at ICU hour 59, and we repeat the identical analysis on
> in-hospital mortality in PhysioNet 2012 (Silva et al. 2012). Death is not defined by
> measurements, and all stays cover the same 48 hours. An effect that appears in both tasks
> cannot be attributed to label construction or cohort selection alone; an effect that appears
> only for sepsis suggests that it can.

Check before use: the Sepsis-3 wording against Reyna et al. 2020 and Singer et al. 2016, and
every number against the table below.

## 11. Where the numbers come from

| Number | Source |
| --- | --- |
| Patients, outcome rates, positive hours | `scripts/check_data.py` (branch `luca_branchmann`) |
| Baseline AUROCs, SOFA check | `results/baselines.csv`, `results/sofa_check.csv` (branch `luca_branchmann`) |
| Records past ICU hour 59; 2,019 patients and 1.3 % after the cut; lab rate around onset | Matteo's check on the raw files, 13 Nov. Not yet reproducible in this repo: step 1 of the plan adds it to `scripts/check_data.py`. |
