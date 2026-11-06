# Data (not committed)

Everything in `data/` except this file is git-ignored. Put the data elsewhere (e.g. cluster
scratch) with `export MISSINGNESS_DATA_DIR=/path/to/data`.

## Datasets

| Dataset | Task | Size | Source |
| --- | --- | --- | --- |
| PhysioNet/CinC Challenge 2019 (Reyna et al. 2020) | hourly sepsis label, causal | 40,336 patients, 40 hourly variables, two hospital systems | https://physionet.org/content/challenge-2019/1.0.0/ |
| PhysioNet/CinC Challenge 2012 (Silva et al. 2012) | in-hospital death from first 48 h | 12,000 stays (sets a, b, c), 37 variables, ~14% deaths | https://physionet.org/content/challenge-2012/1.0.0/ |

Both are open access (cite Goldberger et al. 2000 for PhysioNet). In the 2019 data the
challenge's `training_setA` / `training_setB` are our **hospital 1** / **hospital 2**.

## Steps

```bash
bash scripts/download_data.sh                 # -> data/raw/physionet2019, data/raw/physionet2012
python scripts/prepare_data.py physionet2019 physionet2012
# only once, by the week-1 owner, then commit splits/*.json:
python scripts/prepare_data.py physionet2019 physionet2012 --make-splits
```

## Layout after preparation

```
data/
  raw/physionet2019/training/training_setA/*.psv   (hospital 1)
  raw/physionet2019/training/training_setB/*.psv   (hospital 2)
  raw/physionet2012/set-{a,b,c}/*.txt, Outcomes-{a,b,c}.txt
  processed/physionet2019/timeseries.parquet
  processed/physionet2012/timeseries.parquet
```

## Processed format

One row per patient-hour: `patient_id | source | hour | <static> | <variables> | label`.
Missing measurements are NaN. Variable and static column lists are in
`configs/data/<dataset>.yaml`. PhysioNet 2012 is put on an hourly grid (hours 0–47, mean
within an hour, decision D3). See `src/missingness/data/records.py`.

## Sanity numbers to check in week 1

- 2019: 40,336 patients in total; about 7% septic patients and about 2% positive hours.
- 2012: 12,000 stays; about 14% in-hospital deaths.
