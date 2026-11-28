Status: active

## Done (28 Nov)

- Option chosen by Luca: leak audit + channel intervention + snapshot view (D29, proposed).
- Code with tests on synthetic data: `preprocessing.decode_mask`, `snapshot_values`,
  `featurize(view=...)`; `data/views.py` (config `features`, snapshot widths resolved from the
  training records and saved in the run config); `eval/leak.py` and `scripts/leak_audit.py`;
  `evaluate.py --channel`, `scripts/evaluate_runs.py --channel`; new measures in
  `scripts/measures.py`; `view` column in `results/runs.csv` and `results/measures.csv`.

## Next

1. `python scripts/leak_audit.py --coverage` on the real data; fix bin widths and
   `min_coverage` in D29 before any snapshot run.
2. `python scripts/leak_audit.py`; channel evaluation of the existing `all` runs.
3. The 18 decoded-pattern and 36 snapshot runs (`scripts/leak_audit.py --print-commands`).

## Blockers

- D29 needs the group's confirmation; the L3 margin is needed for the snapshot check.
