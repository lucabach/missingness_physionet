#!/usr/bin/env bash
# E5 planted pattern on mortality (plan P4, D21): on real records, keep the HR measurements
# with probability q1 = 1.0 for patients who died and q0 = 1/strength for the others, in
# train, validation and test. Strength {1, 1.5, 2, 4}; 1 is the control.
#   bash slurm/submit_planted.sh                          # 3 models x 2 inputs x 4 x 3 seeds
#   MODELS=gru LOCAL=1 bash slurm/submit_planted.sh       # GRU only, on this machine
# Afterwards: python scripts/evaluate_runs.py (adds the rule-off evaluation).
set -euo pipefail
for model in ${MODELS:-gru grud odernn}; do
  cfg="configs/experiment/mortality_${model}.yaml"
  grep -q "^tuned:" "$cfg" || { echo "no tuned block in $cfg: run the sweep first"; exit 1; }
  for inputs in ${INPUTS:-all values}; do
    for strength in 1 1.5 2 4; do
      q0=$(python -c "print(1 / ${strength})")
      for seed in ${SEEDS:-0 1 2}; do
        args=("$cfg" "inputs=${inputs}" "seed=${seed}"
          "planted.variable=HR" "planted.q1=1.0" "planted.q0=${q0}")
        if [[ -n "${LOCAL:-}" ]]; then
          python -m missingness.train "${args[@]}" --skip-done
        else
          sbatch slurm/train.sbatch "${args[@]}" --skip-done
        fi
      done
    done
  done
done
