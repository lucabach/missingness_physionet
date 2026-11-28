# RQ2 redesign: a reference model that cannot see the pattern (L1, D29 proposed)

*Created 28 Nov 2026 (Luca + agent). Open item L1 of the 26 Nov meeting with Sarah
(`docs/log/ta_feedback.md`). The literature review behind it is summarised in section 5.*

## 1. Problem

M2 compares the extra loss of the `all` model under state-dependent removal (against matched
random removal) with that of the `values` model. The `values` model is the reference that should
only lose physiological information. With forward fill it also sees the pattern:

- a new measurement is a change of x, so m can be rebuilt from x (`preprocessing.decode_mask`);
  it is missed only when a new value equals the carried one exactly;
- delta is the number of hours x has stayed flat, which a GRU can count;
- before the first value x is exactly 0 (the training mean).

So Δmask is a lower bound, and M2 is biased towards 0: when a value is removed, the `values`
model also notices it (x stops changing there), and that reaction is subtracted.

No imputation removes this completely. Physiology is only observed where something was
measured, so any input built from the observations is sharper near them; Le Morvan et al.
(2021) show that impute-then-regress is consistent for almost any imputation because the
learner can read the mask back (Josse et al. 2019 for mean imputation). Only a **common
schedule**, the same measurement times for every patient, is pattern-blind, and it costs
information (Sarah's point about thinning).

## 2. Approach (chosen by Luca on 28 Nov: audit + intervention + snapshot)

| Part | What | Training | Code |
| --- | --- | --- | --- |
| A. Leak audit | Share of true measurements visible as a change of forward-filled x, per variable; `pattern` model trained on m and delta rebuilt from x (`features=decoded`). Measure `M1_decoded_gap` = pattern(true) − pattern(decoded) | 18 short runs (3 models × 2 tasks × 3 seeds) | `scripts/leak_audit.py`, `eval/leak.py` |
| B. Channel intervention | The trained forward-fill `all` model gets x from the clean test set and m, delta from the perturbed one (`--channel pattern`), or the reverse (`--channel values`). No values lost in the pattern arm, so no reference model needed. `M2_channel_*`, `M4_thinning_channel_*` | none | `evaluate.py --channel`, `scripts/evaluate_runs.py --channel` |
| C. Snapshot view | Per variable, the last value of each fixed bin, shown at the bin end; variables with less than `min_coverage` of bins filled are hidden. Train `values` and `all` on it (`features=snapshot`) | 36 runs (3 models × 2 tasks × 2 variants × 3 seeds) | `preprocessing.snapshot_values`, `data/views.py`, `snapshot` block in the data configs |

### What C gives

- **Check that the pattern is gone:** `all_snapshot − values_snapshot` (M1 in the snapshot
  view) should lie within the L3 margin. The only remaining pattern is the share of empty
  bins, reported by the leak audit (`empty_bin_share`).
- **A bound on Δmask:** at least `all − values` with forward fill (M1, lower end), at most
  `all − values_snapshot` (`M1_bound_snapshot`, upper end: the snapshot also loses
  information).
- **M2 and M4 against a pattern-blind reference:** `M2_ref_snapshot`, `M4_thinning_ref_snapshot`.
  The snapshot model cannot react to removing the extra values of a bin as long as its last
  value survives; it only reacts when a bin becomes empty.

### How each part handles Sarah's thinning point

- A removes nothing.
- B keeps every value in the pattern arm, so the loss there is not information loss. It is a
  lower bound (x still carries the pattern) and the mixed inputs never occur in training
  (Hooker et al. 2019); report it as "sensitivity through the explicit pattern inputs".
- C removes information on purpose. The `all_snapshot` check uses the same thinned data, so
  the information loss cancels there; the snapshot gap is only ever reported as the upper end
  of a bound. Optional calibration: a bin-width curve (1, 2, 4, 8 h) on HR alone, which is
  measured almost every hour, shows the pure information cost of coarsening.

## 3. Steps and gates

1. `python scripts/leak_audit.py --coverage` on the training splits; fix the bin widths and
   `min_coverage` in the data configs **before any snapshot run** (no labels read). Record in D29.
2. `python scripts/leak_audit.py` → `results/leak_audit/leak_audit.csv`.
3. Channel evaluation of the existing `all` runs: `python scripts/evaluate_runs.py --channel`.
4. Training runs: `python scripts/leak_audit.py --print-commands` lists them (decoded pattern,
   snapshot `values` and `all`), then `scripts/evaluate_runs.py --runs '*_snapshot_*'`.
5. `python scripts/collect_results.py && python scripts/measures.py`.
6. Gate for the report: the snapshot check (`M1` in view `snapshot`) within the L3 margin. If
   not, report the snapshot as decorrelating, not removing, the pattern.

## 4. Open before runs

- Bin widths and `min_coverage` (step 1). Defaults in the configs: vitals 2 h; labs 12 h
  (mortality) and 24 h (sepsis); 95%.
- B edits model inputs at test time, as D28 did (invariant 1 holds for the records; the mixed
  batch is an input ablation). Confirm with the group.
- L3 margin, needed for the gate in step 6.

## 5. Literature (short)

Informative missingness: Che et al. 2018 (GRU-D), Lipton et al. 2016, Agniel et al. 2018 (BMJ),
Sharafoddini et al. 2019, Groenwold 2020. Imputation does not hide the mask: Josse et al. 2019,
Le Morvan et al. 2021 (NeurIPS), Van Ness et al. 2023. Prediction vs deployment: Sperrin et al.
2020, Sisk et al. 2023, Jeanselme et al. 2022. Alternatives that only weaken the leak and are
not run: GP adapters and multitask GPs (Li & Marlin 2016, Futoma et al. 2017, Moor et al. 2019),
interpolation networks (Shukla & Marlin 2019, 2021), stochastic imputation (GP-VAE, Fortuin et
al. 2020; CSDI, Tashiro et al. 2021; SAITS and BRITS via PyPOTS), adversarial removal (Ganin et
al. 2016; Zhang et al. 2018), which leaves the attribute decodable by a fresh probe (Elazar &
Goldberg 2018). Off-distribution inputs: Hooker et al. 2019 (ROAR). Full list with links:
the project write-up `results/rq2_pattern_free_values_model.md` in the shared project files.
