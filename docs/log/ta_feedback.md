# TA feedback (Sarah Martinson)

*Kind: log (add one section per meeting with Sarah). Map of all docs: [docs/README.md](../README.md).*

## Proposal meeting, 6 Nov 2026

1. **Wording:** say "sensitivity to measurement changes", not "reliance" (too strong).
2. **Task:** sepsis is a good idea, but PhysioNet may have too few cases; also consider
   mortality or kidney failure as targets. → D2, open question Q1.
3. **Benchmarks:** consult the papers that benchmark models on PhysioNet, for task choice
   and which baselines they use.
4. **Same patients:** unthinned and thinned samples always contain the same set of patients.
   → invariant enforced by `perturb.check_only_removed`.
5. **Thinning removes information too:** it removes the measurement pattern *and*
   physiological information. Keep this in mind in every claim and analysis.
6. **Citations:** always cite the paper when mentioning SHAP, PhysioNet, MIMIC.
7. **Labels:** don't overload letters (A/B/C); "B" meant different things in the proposal.
   Keep labels consistent and unique. → hospital 1/2, RQ1–3, D1…, Q1….
8. **Simpler models:** add ODE-type models; check which models benchmark papers use. → ODE-RNN.
9. **Scope:** large; consider cutting something; the CDE model may be too much. → D1.

Add later meetings below with the date.

## Results meeting, 26 Nov 2026

From Luca's raw notes, checked on 26 Nov against the repo (`main` after PR #8), `results/`, the
decisions log and the meeting slides. Where the notes were imprecise the checked version is
given and marked **Correction**. Slide numbers refer to the 10-slide meeting deck.

### What was asked, and the checked answer

1. **"Pattern = observation pattern?"** Yes. In our code the pattern is the mask m (measured
   this hour or not) plus the time since the last measurement delta (`preprocessing.featurize`).
2. **"Do we need to change the task?"** Sarah: first look at the results. Later in the meeting:
   we may adapt the research questions to what we found (see item 15).
3. **"Why two datasets?"** Mortality comes from PhysioNet 2012 (Silva et al. 2012), sepsis from
   PhysioNet 2019 (Reyna et al. 2020); neither dataset has both labels. Mortality is also the
   contrast whose label is not built from measurements (D12).
4. **"Do you know the two hospitals?"** The challenge page only says "three separate hospital
   systems", two of them public. Reyna et al. 2020 name them: set A (our hospital 1) Beth Israel
   Deaconess Medical Center, Boston; set B (our hospital 2) Emory University Hospital, Atlanta.
   **To confirm in the paper before it goes in the report** (from memory, not checked here). In
   code, plots and the report they stay hospital 1 / hospital 2.
5. **The artefact. Correction:** we do not remove the last 59 hours. We cut every PhysioNet 2019
   record **at ICU hour 59**: hours up to 59 are kept, everything after is dropped (D15), in
   training, validation and test, both hospitals. Reason: almost no non-septic record goes past
   hour 59 (hospital 1: 644 septic vs 8 non-septic), so the length of the stay gave the label
   away. Two numbers are in circulation: the hour alone as a score, AUROC 0.67 → 0.51 (D15), and
   the time-only model, 0.71 → 0.58 (slide 3 notes, work ledger 17 Nov). Use one of them, named,
   in the report.
6. **"Difference between thinning and removal?"**
   - *Removal* (RQ2, test time only): take p of each patient's values away, either from busy
     hours (state-dependent, D5 extras; D27 whole busy hours, proposed) or the same number of
     each variable at random hours (random, D16). p = 0.05–0.15 (D17), 0.20–0.30 (D27).
     Measure M2 = extra loss of the `all` model over the `values` model, state-dependent minus random.
   - *Thinning* (RQ3): per variable, keep each value with probability target rate / source rate,
     so the same hospital-1 test patients get hospital 2's measurement rates (D6, D14; about
     15% of values removed).
7. **"What is transfer?"** Train on hospital 1, test on the hospital-2 test split; measure M4 =
   how much more the `all` model loses than the `values` model.
8. **RQ1, "does pattern + values = all? Values also contain the pattern."** Sarah is right, and
   we already say it in part: forward-filled values leak timing (a value that changes was just
   measured), so the `values` model is not pattern-free and Δmask is a lower bound (CLAUDE.md;
   GRU-D `values` is a plain GRU). The scores are not additive (mortality GRU: pattern 0.77,
   values 0.86, all 0.87). This is the reason for Luca's RQ2 redesign (open item L1).
9. **RQ2, "how do you quantify no effect?"** Today we say "the 95% interval includes 0", which
   is absence of evidence, not evidence of absence. Needed: a margin fixed in advance (for
   example ±0.01 AUROC) and the statement that the whole interval lies inside it, or the
   largest effect the interval rules out. The planted habit gives the detection power: the GRU
   with the pattern picks up a planted effect of +0.005 / +0.015 / +0.069 (slide 5 notes). Open
   item L3.
10. **Busy-hour removal above 30%. Sarah:** removing up to 30% keeps the information that an
    hour was busy. Checked: at 20–30% the D27 rule removes whole busy hours one at a time until
    p is reached, so most patients keep some busy hours, and the D5 extras rule stops at the
    patient's median count but only 17–24% of values qualify (slide 5 notes). We have not yet
    computed what share of each patient's values sits in busy hours, i.e. what p "remove every
    busy hour" means. Open item L2.
11. **Slides. Sarah:** "I know Claude made the slides." Clean the slides for the final
    presentation (open item T1).
12. **RQ3, "what is the point of the thinned version?"** Luca: the middle ground between
    hospital 1 and 2. **Precision:** thinning keeps the hospital-1 patients and changes only the
    measurement rates, so the thinned-vs-values comparison isolates a rate change without a
    change of patients.
13. **"Is the thinning done to match hospital 2?"** Luca: yes. **Precision:** it matches
    hospital 2's *per-variable measurement rates* (D14), and only for the 23 of 34 variables that
    hospital 1 measures more often; the 11 it measures less (for example EtCO2, TroponinI)
    cannot be matched by removal (D11). It does not match when in the stay tests happen, how
    tests cluster in time, or the patient mix. This gap is why Sarah suggests distribution
    matching methods (item 18).

### Feedback

14. **Reframe RQ1:** "the pattern is *usable*", not "the pattern is redundant". Slides 4 and 7
    said "informative but mostly redundant". The checked numbers support "usable": the pattern
    alone gets 65–86% of the way from chance to the full model (slide 4 notes), and adding it
    to the values gains +0.004 to +0.012 (mortality) and +0.026 to +0.043 (sepsis, before onset).
15. **We may adapt the research questions** to the findings; we do not have to stick to the
    proposal.
16. **Report positive and negative findings** (lists below).
17. **Focus on the deep learning part:** say, per model, why we expect it to be more or less
    sensitive *before* showing the result (open item T2).
18. **Thinning (marked important by Sarah):** explain why we use thinned data; consider
    *training* on thinned data; do more on the thinning question with more and better plots (a
    line plot may not be the best); look at the literature on unsupervised distribution
    matching. Open items P2–P4.
19. **Data access (marked important):** check that the data is public for everyone. Checked on
    26 Nov on the PhysioNet pages: PhysioNet 2019 (challenge-2019 v1.0.0) is Open Access under
    CC BY 4.0, PhysioNet 2012 (challenge-2012 v1.0.0) is Open Access under the Open Data
    Commons Attribution License v1.0. Anyone can download both without credentialing (as
    `README.md` says). **Note:** eICU (an extension in CLAUDE.md) and MIMIC need credentialed
    access and training, so they would not meet this rule; keep them out or say so.

### Findings to report (both kinds)

Positive:
- RQ1: the pattern alone predicts both outcomes well (mortality 0.74–0.77, sepsis 0.71 AUROC,
  three models); it is usable.
- The method works: all three models learn the planted habit, and switching it off drops them
  below the values-only model (GRU 0.94 → 0.80 vs 0.86).
- Thinning: the `all` model loses more than the `values` model, significant for GRU-D
  (+0.021 [0.009, 0.034]) and ODE-RNN (+0.014 [0.003, 0.025]); GRU +0.012 [−0.002, 0.024].
- SOFA check for the neural models (D28 proposed): hiding the SOFA variables' pattern hurts
  GRU-D more than five random variables at both hospitals; the effect is small next to Δmask,
  so most of the pattern signal is not label construction.

Negative:
- Adding the pattern to the values gains little (above), and that gain is a lower bound we
  cannot yet separate cleanly (item 8).
- Busy-hour removal has no effect beyond random removal at 5–30% (mortality 9/9 and sepsis
  16/18 intervals include 0, no trend with the rate).
- Transfer to hospital 2: the extra loss with the pattern (+0.018 to +0.027) has intervals that
  include 0 (about 120 septic hospital-2 test patients); gradient boosting loses nothing.
- RQ3 as proposed (link sensitivity to transfer across models) has no spread to work with:
  the three models are similar.

### Fixes to the meeting slides

- **Slide 10 notes (SOFA):** "0.014 [−0.006, 0.037] at hospital 1, against −0.002 to +0.004 for
  five random variables" is the **gradient-boosting all-hours** number (`results/sofa_check.csv`),
  although the slides otherwise use hours before onset; there it is +0.021 [−0.000, 0.043]
  against +0.001 to +0.003. Label it as gradient boosting, all hours. The neural check
  (`results/sofa_check_neural/README.md`) now adds: GRU-D significant at hospital 1 (+0.015
  [0.002, 0.027] before onset) and beyond the random draws at both hospitals; GRU beyond the
  random draws at hospital 2 only; ODE-RNN no SOFA-specific effect.
- Slides 4 and 7: "mostly redundant" → "usable" (item 14).
- Slide 3 vs D15: one artefact number, named (item 5).

### Still open (owners)

| id | owner | item |
| --- | --- | --- |
| P1 | Parthiv | RQ3 on the merged `main`: per-model thinning and transfer, reframed with Sarah's points |
| P2 | Parthiv | Train on thinned data (hospital 1 thinned to hospital-2 rates), test on hospital 1, thinned and hospital 2; does the transfer gap close? Needs a D-number |
| P3 | Parthiv | Better thinning plots than a line plot (for example per-variable rate shift vs loss, paired differences per model) |
| P4 | Parthiv | Literature on unsupervised distribution matching (domain adaptation between hospital measurement patterns); how thinning relates to it |
| L1 | Luca | RQ2 redesign: a values-only model that is fully pattern-free (forward fill leaks timing); keep the current null result as reported. Needs a D-number before runs |
| L2 | Luca | Removal above 30% per Sarah: first compute each patient's share of values in busy hours, then remove all busy hours or flatten every hour to the median count |
| L3 | Luca | Quantify "no effect": an equivalence margin or the largest effect ruled out, using the planted habit as the detection-power check |
| T1 | all | Clean the slides for the final presentation (fewer words, our own figures, slide-10 fix) |
| T2 | all | Deep learning focus: expectation per model, written before the results (GRU sees m and delta as plain inputs; GRU-D decays inputs and hidden state with the gaps, so it should depend most on delta; ODE-RNN evolves its state between observations, but with one Euler step per hour, D26, it is close to a GRU here) |
| T3 | all | Reframe RQ1 as "usable", adapt the RQs where the findings call for it (items 14–15) |
| T4 | Luca | Confirm the two hospital names in Reyna et al. 2020 (item 4) |
