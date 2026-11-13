# Week 1 handoff (owner: Luca + colleague, dates: 9–15 Nov) — draft after day 1 (13 Nov)

Luca's part, branch `luca_branchmann`. Merge with the colleague's branch before Sunday.

## Done
- Both datasets download and process: `scripts/download_data.sh` fixed (2019 has no ZIP;
  ~40k files fetched in parallel with retries, ~15 min).
- Loaders checked on the real files: all counts match (see table below), no parser bugs.
- Patient splits `splits/physionet2019_seed0.json` and `splits/physionet2012_seed0.json`
  committed (decision D8).
- Hospital 1 vs 2 comparison: `scripts/compare_hospitals.py` → `results/hospital_rates.csv`
  and `results/hospital_rates.pdf`.
- Facts for Q1 (labels), Q2 (training hospital), Q3 (SOFA list) written in `docs/decisions.md`.
- 1-epoch GRU runs on real data for both tasks, plus random/state-dependent removal and
  thinning evaluations: the whole pipeline runs end to end.

## Not done / open items
- Q1, Q2, Q3 need a team decision (Q6 decided: D10, the 12 empty 2012 stays are dropped) (facts in `docs/decisions.md`). After Q2, set
  `train_source` in `configs/data/physionet2019.yaml`; after Q3, fix `sofa_variables`.
- `prepare_data.py --make-splits` reprocesses the raw data (~9 min); could skip if the parquet exists.

## How to run what I built
```bash
bash scripts/download_data.sh
python scripts/prepare_data.py physionet2019 physionet2012        # ~9 min on 4 CPUs
python scripts/check_data.py physionet2019 physionet2012
python scripts/compare_hospitals.py                                # ~1.5 min
python -m missingness.train configs/experiment/sepsis_gru.yaml train.epochs=1 eval.n_boot=50
python -m missingness.evaluate outputs/<run> --perturb state --p 0.3
```

## Results and numbers worth knowing
| | hospital 1 | hospital 2 | PhysioNet 2012 |
| --- | --- | --- | --- |
| patients / stays | 20,336 | 20,000 | 11,988 (12 empty stays dropped) |
| positive patients | 1,790 (8.8%) | 1,142 (5.7%) | 1,707 deaths (14.2%) |
| positive hours | 2.2% | 1.4% | – |
| median stay (hours) | 39 | 38 | 48 (fixed) |
| observed values per patient-hour | 0.21 | 0.19 | – |

- Hospital 1 measures 23 of 34 variables more often, mostly blood gases and chemistry
  (BaseExcess, HCO3, Chloride, FiO2, pH, PaCO2, PTT). Hospital 2 measures 11 more often,
  including EtCO2, which hospital 1 never records, and TroponinI.
- 1-epoch GRU (all inputs, CPU, ~1 min/epoch for sepsis): sepsis test AUROC 0.75 (hospital 1)
  and 0.73 (hospital 2); mortality test AUROC 0.84. Not tuned, only a pipeline check.
- Removal with p = 0.3 achieves only 23% / 16%: too few state-dependent candidates (Q4).

## Watch out / next steps for the next owner
- EtCO2 is never measured in hospital 1, so a model trained there never sees it; it shows
  up only at test time in hospital 2 (normaliser falls back to mean 0, std 1).
- 426 septic patients are positive from their first recorded hour (no pre-onset negatives).
- Unit1/Unit2 are unknown for 39% of 2019 patients; Height for 48% of 2012 stays.
- Training on CPU is fine for checks; real runs belong on the cluster (`slurm/`).
