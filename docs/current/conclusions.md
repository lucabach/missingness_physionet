# Conclusions so far

*Kind: living (rewrite in place as results come in). Map of all docs: [docs/README.md](../README.md).*

State of 19 Nov 2026 (Matteo): classical baselines and **every GRU run** (protocol `protocol-v1`,
D15–D24). GRU-D and ODE-RNN follow in week 3 with the same scripts; week 4 extends this file to
all three models. Every number comes from `results/` (`baselines.csv`, `sofa_check.csv`,
`runs.csv`, `measures.csv`); rerun commands are in `docs/log/handoff_week_02.md`, section 8.

How to read the numbers: test AUROC, mean over seeds 0–2. A difference between models comes
with a 95% interval from the paired patient bootstrap (D19, D23): `+0.009 [0.002, 0.017]`. For
sepsis the headline is pre-onset hours (plan section 4); all hours next to it.

## 1. In short

1. **The measurement pattern carries a lot, but mostly the same information as the values.**
   A model on the pattern alone gets 73–91% of the way from chance to the full model (S), but
   adding the pattern to the values adds little: Δmask +0.009 for the mortality GRU, +0.029
   for the sepsis GRU (hospital 1, pre-onset), 0 to +0.01 for gradient boosting (mortality,
   hospital 1).
2. **Neither GRU is sensitive to a state-dependent change of the pattern.** Removing the extra
   measurements of busy hours costs no more than removing the same number of the same
   variables at random (M2 at p = 0.10: mortality −0.002 [−0.006, 0.002]; sepsis, hospital 1,
   pre-onset +0.003 [−0.004, 0.009]). For sepsis it even costs *less*: the extra tests of busy
   hours mostly repeat what was just measured.
3. **The planted pattern shows that the measures work.** When we plant a measurement habit
   ourselves, the `all` GRU picks it up in proportion to its strength and loses more than it
   gained when the habit is switched off; the `values` GRU never picks it up (section 4.2).
4. **Transfer to hospital 2 (exploratory):** the sepsis GRU with the pattern loses more than the
   values-only GRU, at hospital 2 (M4 +0.022 [−0.013, 0.056], pre-onset) and when the same
   patients are thinned to hospital-2 rates (+0.012 [−0.002, 0.024]); the same on the 25
   variables both hospitals measure alike. The classical baselines lose nothing at hospital 2.
5. Three earlier statements no longer hold (section 6): "the pattern beats the values"
   (the hour-59 artefact), "the baselines collapse at hospital 2" (an early-stopping bug), and
   "the SOFA variables carry more of the pattern signal" (not significant with intervals).

## 2. Facts about the data that shape every result

- **Cohorts.** PhysioNet 2019 (Reyna et al. 2020): hospital 1 20,336 patients (8.8% septic),
  hospital 2 20,000 (5.7%); sepsis models train on hospital 1 (D11). PhysioNet 2012 (Silva et
  al. 2012): 11,988 stays (D10), 14.2% deaths.
- **The hour-59 artefact (D15).** Almost no record of a patient without sepsis goes past ICU
  hour 59 (past it at hospital 1: 644 septic, 8 not), so the hour alone predicted the label
  (AUROC 0.67). All sepsis records are cut at hour 59; the hour alone then gives 0.51 (as a
  score) or 0.58 (time-only gradient boosting). Cost: septic test patients with a positive
  hour fall from 269 to 175 (hospital 1) and from 172 to 121 (hospital 2), so intervals are wide.
- **The hospitals measure differently.** Hospital 1 measures 23 of 34 variables more often
  (BaseExcess 45×, HCO3 42×); hospital 2 measures EtCO2 (never at hospital 1), TroponinI and
  Glucose more (`results/hospital_rates.csv`).
- **State-dependent removal can only remove a little.** It deletes the extra measurements of
  hours in which a patient is measured more than usual. Only 24% (hospital 1), 17%
  (hospital 2) and 20% (mortality) of values qualify, so p ≤ 0.15 (D17); they are mostly
  labs, which is why random removal is matched per variable (D16).
- **Thinning** hospital-1 test patients to hospital-2 rates removes 15.5% of values; 23 of 34
  variables reach the target rate, the other 11 are measured less at hospital 1 (D14).

## 3. RQ1: how much does the pattern carry?

### 3.1 Classical baselines (after the cut and the early-stopping fix, one seed)

| Task, test set | Model | all | values | pattern | time | Δmask | S |
| --- | --- | --- | --- | --- | --- | --- | --- |
| mortality | gradient boosting | 0.867 | 0.867 | 0.773 | – | +0.000 | 0.74 |
| mortality | logistic regression | 0.860 | 0.851 | 0.758 | – | +0.009 | 0.72 |
| sepsis, hospital 1, pre-onset | gradient boosting | 0.749 | 0.739 | 0.716 | 0.588 | +0.010 | 0.87 |
| sepsis, hospital 1, pre-onset | logistic regression | 0.737 | 0.677 | 0.687 | 0.471 | +0.060 | 0.79 |
| sepsis, hospital 2, pre-onset | gradient boosting | 0.757 | 0.708 | 0.690 | 0.561 | +0.049 | 0.74 |

Intervals for single AUROCs are about ±0.02 (mortality), ±0.03 (hospital 1) and ±0.04
(hospital 2). The values features of the baselines still leak some timing (docstring of
`baselines/features.py`), so their Δmask is a lower bound.

### 3.2 GRU, M1 (`results/measures.csv`)

| Task, test set | all | values | pattern | Δmask (all − values) | S |
| --- | --- | --- | --- | --- | --- |
| mortality | 0.870 | 0.861 | 0.770 | **+0.009 [0.002, 0.017]** | 0.73 |
| sepsis, hospital 1, pre-onset | 0.758 | 0.729 | 0.713 | **+0.029 [0.005, 0.055]** | 0.83 |
| sepsis, hospital 1, all hours | 0.765 | 0.730 | 0.727 | +0.035 [0.012, 0.061] | 0.86 |
| sepsis, hospital 2, pre-onset | 0.698 | 0.691 | 0.674 | +0.007 [−0.018, 0.031] | 0.88 |

- Mortality `all` 0.870 is slightly above the expected 0.82–0.86 (plan phase 2 gate). We use
  all 12k stays (8.4k for training), more than the 4k of the early papers; nothing in the
  pipeline points to leakage (every stay covers 48 h).
- The sepsis GRU gains more from the pattern than gradient boosting does (+0.03 against
  +0.01). Its values-only variant is weaker (0.730 against 0.740), its pattern-only variant
  equal (0.727 against 0.725).
- Seeds agree closely: the standard deviation over the three seeds is at most 0.015 AUROC per
  cell, well below the patient intervals.

## 4. RQ2: how sensitive are the models to a change in the pattern?

### 4.1 Removal, M2

M2 = the extra loss under state-dependent removal compared with matched random removal, for
`all` minus the same for `values`. Positive means the pattern inputs make the model more
sensitive to the state-dependent change.

| Task, test set | p = 0.05 | p = 0.10 (primary) | p = 0.15 |
| --- | --- | --- | --- |
| mortality | +0.000 [−0.003, 0.003] | −0.002 [−0.006, 0.002] | −0.000 [−0.006, 0.005] |
| sepsis, hospital 1, pre-onset | −0.003 [−0.010, 0.003] | +0.003 [−0.004, 0.009] | +0.006 [−0.003, 0.014] |
| sepsis, hospital 2, pre-onset | −0.002 [−0.009, 0.005] | +0.002 [−0.007, 0.009] | +0.005 [−0.005, 0.014] |
| sepsis, 25 variables, hospital 1, pre-onset | +0.002 [−0.003, 0.006] | −0.005 [−0.011, 0.002] | +0.001 [−0.006, 0.008] |

Mortality: removal itself costs little (p = 0.15: `all` 0.870 → 0.861 random, 0.858
state-dependent; `values` 0.861 → 0.855 and 0.852), and the two methods cost the same.
Sepsis: the same null result, at both hospitals and on the 25 variables; at hospital 1 M2
grows a little with p but stays inside its interval. State-dependent removal costs **less**
than random removal for both variants (hospital 1, p = 0.10, pre-onset: `all` −0.007 against
−0.010, `values` −0.001 against −0.007). Our rule deletes the extra measurements of busy
hours, and these mostly repeat what was just measured; random removal (same number per
variable) also hits the isolated measurements that carry new information. So M2 ≈ 0 means:
the GRU does not depend on the busy-hour extras more than on any other measurement.

### 4.2 Planted pattern, M3 (mortality, HR, D21 and D24)

| Strength q1/q0 (D24: q1 = 0.8) | 1 | 1.05 | 1.1 | 1.2 |
| --- | --- | --- | --- | --- |
| `all`, rule on | 0.870 | 0.875 | 0.885 | 0.939 |
| `all`, rule off | 0.870 | 0.871 | 0.854 | 0.797 |
| `values`, rule on / off | 0.862 / 0.861 | 0.862 / 0.861 | 0.862 / 0.861 | 0.862 / 0.861 |
| gain of `all` (on − strength 1) | – | +0.005 [0.003, 0.007] | +0.015 [0.007, 0.023] | +0.069 [0.054, 0.084] |
| loss when off, `all` − `values` | – | +0.003 [0.001, 0.005] | +0.030 [0.023, 0.038] | +0.141 [0.120, 0.164] |

- The measures find what we planted, and grow with its strength. The `values` GRU gains and
  loses nothing (forward-filled HR leaks almost no timing here).
- **Switching the habit off costs more than the habit gained:** at strength 1.2 the `all` model
  falls to 0.797, below the `values` model (0.862) and below its own control (0.870). A model
  that learned a measurement habit is worse than one that never saw the pattern once the habit
  changes. This is the risk RQ2 and RQ3 are about, shown where we know the truth.
- The first design (D21: q1 = 1, strength 1.5, 2, 4) saturated: `all` reached 0.999 at every
  strength, because a patient who died never lost an HR value. Its runs stay as the saturated
  end (gain +0.13, loss when off +0.26 to +0.28).

## 5. RQ3: transfer to hospital 2 (sepsis, exploratory)

Sepsis models train on hospital 1. Thinning removes hospital-1 measurements of the same
patients down to hospital-2 rates; transfer scores the model on hospital 2's own patients.
M4 is the extra loss of `all` over `values` (pre-onset hours; the two hospitals are
bootstrapped independently, D23).

| Pre-onset AUROC | recorded (hospital 1) | thinned (hospital 1, seeds 0–2) | hospital 2 |
| --- | --- | --- | --- |
| `all`, 34 variables | 0.758 | 0.745 | 0.698 |
| `values`, 34 variables | 0.729 | 0.728 | 0.691 |
| `all`, 25 variables | 0.749 | 0.738 | 0.695 |
| `values`, 25 variables | 0.724 | 0.721 | 0.693 |

| M4, `all` − `values` | 34 variables | 25 variables |
| --- | --- | --- |
| thinning (recorded − thinned) | +0.012 [−0.002, 0.024] | +0.008 [0.001, 0.015] |
| transfer (hospital 1 − hospital 2) | +0.022 [−0.013, 0.056] | +0.023 [−0.009, 0.058] |

- **The model with the pattern loses more, under thinning and at hospital 2.** The
  direction is the one RQ3 asks about; the transfer intervals include 0 (about 120 septic
  test patients at hospital 2). All hours give the same picture (thinning +0.015 [0.002,
  0.027], transfer +0.028 [−0.007, 0.062]); without the patients positive from their first
  hour the transfer difference is +0.053 [0.011, 0.095].
- **The pattern's gain does not travel:** Δmask is +0.029 at hospital 1 and +0.007
  [−0.018, 0.031] at hospital 2.
- **Not caused by the 9 variables whose rates differ more than threefold** between the
  hospitals: on the other 25 the extra transfer loss is the same (+0.023).
- Thinning the same patients produces about half of the extra transfer loss (0.012 of 0.022). Thinning also
  removes physiological information, so this is a comparison with the `values` model under
  the same thinning, not a loss caused by the pattern alone.
- **The classical baselines behave differently:** gradient boosting on all features scores
  0.749 at hospital 1 and 0.757 at hospital 2 (pre-onset), no loss at all.

## 6. What differs from what we expected

1. **The pattern no longer beats the values** (Luca, 17 Nov). Before the cut the pattern-only
   baseline beat the values (0.793 against 0.750); most of that was the hour of the stay. The
   project's premise holds in a weaker form: the pattern is informative, but redundant.
2. **The baselines' "transfer loss" was a bug.** Gradient boosting early-stopped on random
   hours, so hours of one patient were on both sides. Fixed 18 Nov: at hospital 2, `all` went
   from 0.622 to 0.759. Luca's "odd result" (pattern better than all at hospital 2) is gone.
3. **The SOFA check is inconclusive.** Removing the pattern of the five SOFA variables costs
   0.014 [−0.006, 0.037] at hospital 1 and 0.000 [−0.022, 0.023] at hospital 2; five random
   variables cost −0.002 to +0.004. Earlier, without intervals, this read as "the SOFA
   variables matter more".
4. **The planted pattern of D21 gave the label away** (section 4.2); redesigned as D24.
5. **Mortality shows no RQ2 effect** at the small removal fractions we can reach (p ≤ 0.15).
6. **The sepsis `pattern` GRU stops at the 30-epoch cap** (28–30 epochs, other variants 16–19),
   with validation AUROC still rising slowly: its S may be slightly underestimated. The cap is
   part of the protocol for every model; changing it needs a decision and reruns.
7. **State-dependent removal hurts less than random removal** (section 4.1). We expected the
   opposite: the extra tests of busy hours should be the informative ones. They are mostly
   repeats.
8. **The sepsis GRU gains more from the pattern than gradient boosting, and loses that gain at
   hospital 2** (Δmask +0.029 → +0.007), while gradient boosting transfers without loss.

## 7. Questions for Sarah

1. **Redundant pattern.** After the cut, the pattern alone reaches 73–91% of the full model's
   gain over chance, but adds only +0.01 (mortality) to +0.03 (sepsis GRU) on top of the
   values. Is "the pattern is informative but mostly redundant with the values, and how much
   a model gains from it depends on the model" an acceptable answer to RQ1?
2. **A null RQ2 with our removal rule.** State-dependent removal deletes the extra measurements
   of busy hours. Only 17–24% of values qualify, so p ≤ 0.15, and these extras mostly repeat
   what was just measured: both GRUs lose *less* from them than from matched random removal,
   and M2 is about 0 on both tasks. Is this null result informative for RQ2, or should we
   define the state-dependent change differently (for example, delete the measurements taken
   right after a patient worsens, or a whole busy hour)?
3. **The planted pattern.** Our first design gave the label away (AUROC 0.999); the revised one
   (q1 = 0.8, strength 1 to 1.2) gives a clear dose-response, and switching the habit off then
   costs more than it gained. Is this an acceptable "known truth" test of our measures, and is
   it worth running it on sepsis as well?
4. **Transfer.** The sepsis GRU with the pattern loses more at hospital 2 than the values-only
   GRU (M4 +0.022 [−0.013, 0.056], pre-onset; the interval includes 0), and about half of that
   also appears when the same patients are thinned to hospital-2 rates. The classical
   baselines lose nothing at hospital 2. With three models, RQ3 is an association over three
   points: is that enough for an exploratory RQ3, or should the thinning of the same patients
   carry the shift part of the report?
5. **Cut at hour 59 or a 48-hour window.** We cut sepsis records at hour 59, where the
   artefact is. A 48-hour window would match mortality but drop more septic patients. Which do
   you prefer?
6. **Label link.** The SOFA check is not significant either way (0.014 [−0.006, 0.037]). Is
   reporting it, with mortality as the clean contrast, enough for the label-construction caveat?

## 8. Wording for the report

- Say "sensitivity to measurement changes", never "reliance"; hospital 1 and 2, never A/B.
- Δmask is a lower bound (forward-filled values leak timing). GRU-D `values` is a plain GRU.
- A thinning loss is never attributed to the pattern alone: compare with the `values` model
  under the same thinning (M4) and state the limitation.
- RQ3 is an exploratory association.
- Cite Reyna et al. 2020 (PhysioNet 2019), Silva et al. 2012 (PhysioNet 2012), Goldberger et
  al. 2000 (PhysioNet).
