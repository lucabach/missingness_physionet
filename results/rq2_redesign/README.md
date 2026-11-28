# RQ2 redesign, mortality results (D29, proposed)

28 Nov 2026. 72 runs on PhysioNet 2012 mortality (Silva et al. 2012; Goldberger et al. 2000):
GRU, GRU-D, ODE-RNN × {`all`, `values`, `pattern` with forward fill; `pattern` decoded;
`values` and `all` snapshot} × seeds 0–2, with the tuned configurations. These were trained
in the cloud container, not on the cluster, so the forward-fill rows here are a re-run of the
same protocol and are kept apart from `results/runs.csv` and `results/measures.csv`.

```bash
python scripts/collect_results.py --out results/rq2_redesign/runs.csv
python scripts/measures.py --out results/rq2_redesign/measures.csv
```

Values are the mean over three seeds of test AUROC (subset `all`). The 95% intervals come from
the paired patient bootstrap in `measures.csv`. For GRU-D the `values` variant is a plain GRU.

## Clean test AUROC

| Model | all | values | pattern | pattern decoded | values snapshot | all snapshot |
| --- | --- | --- | --- | --- | --- | --- |
| GRU | 0.870 | 0.861 | 0.770 | 0.760 | 0.777 | 0.770 |
| GRU-D | 0.870 | 0.860 | 0.758 | 0.758 | 0.776 | 0.776 |
| ODE-RNN | 0.863 | 0.859 | 0.736 | 0.712 | 0.774 | 0.774 |

## What it shows

1. **The pattern is almost fully decodable from forward-filled x.** A `pattern` model fed
   only m and delta rebuilt from x loses 0.011 (GRU), 0.000 (GRU-D) and 0.024 (ODE-RNN)
   AUROC against the true mask (`M1_decoded_gap`; every interval includes 0). So the
   forward-fill `values` model can see the pattern, as the leak audit predicted.
2. **The snapshot view removes the pattern's usefulness.** `all_snapshot − values_snapshot`
   (M1 in view `snapshot`) is −0.007 [−0.016, 0.001] (GRU), 0.000 [−0.010, 0.009] (GRU-D) and
   0.000 [−0.005, 0.006] (ODE-RNN). Adding m and delta does not help once every patient
   shares the schedule. The gate (within the L3 margin) passes for any margin of 0.01 or
   more; the margin itself is still open.
3. **The price is large.** The snapshot keeps 8 of 37 variables and costs 0.08–0.10 AUROC.
   So the upper end of the Δmask bound (`M1_bound_snapshot`, about 0.09 for all three models)
   is mostly information loss. The informative bound is therefore Δmask between 0.005–0.010
   (forward fill, lower end) and about 0.09. The snapshot does not narrow it in a useful
   way for mortality.
4. **No sensitivity to state-dependent removal through the pattern inputs.** Channel
   intervention on the forward-fill `all` models (pattern inputs from the removed data,
   values clean), `M2_channel_pattern`: between −0.001 and +0.004 for all models and rates.
   The one interval above 0 is GRU-D at p = 0.1 (+0.004 [0.001, 0.007]). The forward-fill M2
   is also about 0 everywhere (−0.002 to +0.003).
5. **`M2_ref_snapshot` should not be read as pattern sensitivity.** At p = 0.15 it is
   +0.010 to +0.016 (intervals exclude 0 for GRU-D and ODE-RNN). This comes from the
   reference, not from the `all` model: the snapshot models lose *less* under
   state-dependent removal than under random removal (random − state ≈ −0.008 to −0.012),
   whereas the forward-fill models lose about the same under both (+0.003 to +0.005). A
   likely reason is that state-dependent removal hits hours with many measurements, where a
   bin usually keeps another value, while random removal more often empties a sparse bin.
   The snapshot reference therefore reacts differently to *information* loss, and that
   difference enters M2. The channel intervention (point 4) has no such reference problem.

## Variant with 90% coverage (`snapshot-cov90`, 18 runs)

With `min_coverage` 0.90 instead of 0.95 the snapshot keeps 12 of 37 variables (adds
Glucose, K, Mg and Urine). It barely helps: the `values` snapshot reaches 0.777 (GRU), 0.778
(GRU-D) and 0.775 (ODE-RNN), against 0.776–0.777 with 95% coverage. The check still holds
(`all − values` in view `snapshot-cov90`: −0.000, +0.004, −0.002; all intervals include 0).
The bound (about 0.09) and `M2_ref_snapshot-cov90` (+0.013 to +0.015 at p = 0.15) are
unchanged. So the snapshot's information cost for mortality comes from the 4 h / 24 h bins,
not from the coverage cut. A narrower bin would be the lever, not more variables.

## Still to come

- Sepsis (both hospitals, including thinning and M4), 54 runs.

These will be added to this folder when they finish.
