# Patient splits (committed)

`<dataset>_seed<k>.json` holds the patient IDs of the train / val / test split, stratified by
outcome and (for PhysioNet 2019) by hospital. Everyone uses `seed0` unless we decide otherwise
in `docs/decisions.md`. Created once by `python scripts/prepare_data.py <dataset> --make-splits`,
never overwritten. The files only contain the public PhysioNet record IDs, no patient data.
