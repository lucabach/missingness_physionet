# Project overview (plain language first)

Source: revised proposal after the meeting with Sarah on 6 Nov 2026
(`deep_learning/final_proposal/revised/proposal.tex` in the project files; BibTeX in
`docs/references.bib`).

## The story

In an intensive care unit (ICU), nobody measures every value every hour. Doctors order a
lactate test when they are worried, so the *fact* that lactate was measured already says
something about the patient. A model trained on ICU records can therefore learn two things:

- the patient's **physiology** (what the values are), and
- the hospital's **measuring habits** (whether and when values were taken).

Both help prediction in the hospital the model was trained in. Only the first should carry
over to a hospital with different monitoring habits. Our question: **how sensitive are
clinical time-series models to changes in how patients are measured?**

We use only real patients. We never invent data; we only *remove* existing measurements.

## Key ideas (glossary)

- **Measurement pattern:** which variables were measured at which hours. Encoded by the
  **mask** m (1 = measured) and the **time gap** δ (hours since the last measurement).
- **Values:** the measured numbers; between measurements we carry the last one forward
  (**forward fill**), giving x.
- **Causal prediction:** the prediction at hour t uses only data up to hour t.
- **Input variants:** a model trained on `values` (x), on `pattern` (m, δ) or on `all`.
- **Information scores:** S = how much of the full model's skill the pattern alone reaches;
  Δmask = how much adding the pattern to the values helps (a lower bound).
- **Removal:** at test time, delete a fraction of existing measurements, either **random**
  or **state-dependent** (the extra measurements in hours where a patient is measured more
  than usual). Same number removed both ways.
- **Sensitivity to measurement changes:** the *extra* loss under state-dependent over random
  removal, compared with a values-only model. (Not "reliance", see `ta_feedback.md`.)
- **Thinning:** remove measurements from hospital-1 test patients until each variable is
  measured as often as in hospital 2. Same patients, same remaining values.
- **Hospital 1 / hospital 2:** the two hospital systems of PhysioNet 2019 (challenge sets
  A and B).

## Research questions

- **RQ1** How much predictive information do measurement patterns carry beyond the values,
  and how sensitive is each trained model to changes in them?
- **RQ2** How does each model's performance change when only the measurement process
  changes, with the same patients?
- **RQ3** (exploratory) Is higher sensitivity associated with a larger loss at another
  hospital?

## Tasks and data

1. **Sepsis** (PhysioNet 2019; Reyna et al. 2020): at every hour predict the sepsis label
   (positive from 6 h before clinical onset). Two hospital systems, 40,336 patients. Caveat:
   the Sepsis-3 label is computed from lab values, so label and measurement are linked.
2. **Mortality** (PhysioNet 2012; Silva et al. 2012): predict in-hospital death from the
   first 48 h. 12,000 stays, ~14% deaths. Single source, so no transfer experiment.

Metrics: AUROC and AUPRC with 95% CIs from bootstrapping patients; for sepsis also
pre-onset hours only.

## Models

Core: **GRU** with forward-filled inputs, **GRU-D** (Che et al. 2018), **ODE-RNN**
(Rubanova et al. 2019), from open-source implementations, same preprocessing and tuning
budget. Extensions only: Neural CDE (Kidger et al. 2020), Transformer with time and mask
embeddings.

## Experiments (details in `experiments.md`)

1. **Week-1 check:** measurement rates of hospital 1 vs 2; logistic regression and gradient
   boosting on pattern-only, values-only and time-only features; SOFA-variable check.
2. **Information ablation:** each model × {values, pattern, all} → S and Δmask.
3. **Removal:** random vs state-dependent at several fractions p → sensitivity.
4. **Thinning and transfer (sepsis):** train on the hospital that measures more; test on
   (a) same hospital, (b) same patients thinned to the other hospital's rates, (c) the other
   hospital.

## Code ↔ concept

| Concept | Code |
| --- | --- |
| x, m, δ | `data/preprocessing.py::featurize` |
| input variants | `models/inputs.py` |
| S, Δmask | `eval/scores.py` |
| random / state-dependent removal | `perturb/removal.py::paired_removal` |
| thinning | `perturb/thinning.py::thin_to_rates` |
| same-patient guarantee | `perturb/invariants.py::check_only_removed` |
| bootstrap CIs, pre-onset hours | `eval/metrics.py` |
