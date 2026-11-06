# TA feedback (Sarah Martinson)

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
