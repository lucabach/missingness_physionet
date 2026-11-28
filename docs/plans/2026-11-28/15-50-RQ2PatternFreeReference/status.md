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

## Next

1. Group confirms D29 and the widths.
2. Channel evaluation of the existing `all` runs (`scripts/evaluate_runs.py --channel`).
3. The 18 decoded-pattern and 36 snapshot runs (`scripts/leak_audit.py --print-commands`).

## Blockers

- D29 needs the group's confirmation; the L3 margin is needed for the snapshot check.
