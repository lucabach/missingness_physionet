#!/usr/bin/env bash
# The information-ablation grid for one task: models x input variants x seeds.
#   bash slurm/submit_ablations.sh mortality                  # 3 models x 3 inputs x 3 seeds
#   MODELS=gru LOCAL=1 bash slurm/submit_ablations.sh sepsis  # GRU only, on this machine
#   MODELS=gru INPUTS="values all" bash slurm/submit_ablations.sh sepsis_comparable  # P5
# Uses the tuned configuration in configs/experiment/<task>_<model>.yaml (pick_config.py).
# Runs whose metrics.json exists are skipped. Afterwards: python scripts/evaluate_runs.py
# Sepsis trained on hospital 2 (phase 6 only): EXTRA="data.train_source=hospital2".
set -euo pipefail
task="${1:?usage: submit_ablations.sh sepsis|mortality|sepsis_comparable}"
for model in ${MODELS:-gru grud odernn}; do
  cfg="configs/experiment/${task}_${model}.yaml"
  grep -q "^tuned:" "$cfg" || { echo "no tuned block in $cfg: run the sweep first"; exit 1; }
  for inputs in ${INPUTS:-values pattern all}; do
    for seed in ${SEEDS:-0 1 2}; do
      args=("$cfg" "inputs=${inputs}" "seed=${seed}" ${EXTRA:-})
      if [[ -n "${LOCAL:-}" ]]; then
        python -m missingness.train "${args[@]}" --skip-done
      else
        sbatch slurm/train.sbatch "${args[@]}" --skip-done
      fi
    done
  done
done
