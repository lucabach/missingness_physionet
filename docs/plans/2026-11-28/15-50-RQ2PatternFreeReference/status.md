Status: active

## Done (28 Nov)

- Option chosen by Luca: leak audit + channel intervention + snapshot view (D29, proposed).
- Code with tests on synthetic data: `preprocessing.decode_mask`, `snapshot_values`,
  `featurize(view=...)`; `data/views.py` (config `features`, snapshot widths resolved from the
  training records and saved in the run config); `eval/leak.py` and `scripts/leak_audit.py`;
  `evaluate.py --channel`, `scripts/evaluate_runs.py --channel`; new measures in
  `scripts/measures.py`; `view` column in `results/runs.csv` and `results/measures.csv`.

- Real data (28 Nov): snapshot widths set from the coverage table (vitals 4 h, labs 24 h);
  leak audit in `results/leak_audit/`: 81% (mortality), 85% / 88% (sepsis, hospitals 1 and 2)
  of measurements are visible as a change of x. Preview: GRU mortality pattern model 0.770 on
  the true mask vs 0.757 on the mask rebuilt from x (seed 0).

- Mortality, 54 runs (3 models x 6 variants x 3 seeds), evaluated incl. channel intervention
  (28 Nov, cloud): `results/rq2_redesign/`. Decoded gap 0.000-0.024 (pattern almost fully
  decodable from x); snapshot check passes (all_snapshot - values_snapshot within +-0.01) but
  the snapshot costs 0.08-0.10 AUROC; channel M2 about 0; M2_ref_snapshot is driven by the
  reference, not pattern sensitivity (see README).

## Next

1. Group confirms D29 and the widths.
2. cov90 snapshot variant (18 mortality runs) and sepsis (54 runs): running in the cloud.
3. Decide how M2_ref_snapshot is reported (reference reacts to information loss).

## Blockers

- D29 needs the group's confirmation; the L3 margin is needed for the snapshot check.
