#!/usr/bin/env bash
# Submit the core information-ablation grid: models x input variants x seeds for one task.
#   bash slurm/submit_ablations.sh sepsis      # 3 models x 3 inputs x 3 seeds = 27 jobs
#   bash slurm/submit_ablations.sh mortality   # 27 jobs
# Sepsis is run once per training hospital (see docs/experiments.md): pass
#   EXTRA="data.train_source=hospital2" bash slurm/submit_ablations.sh sepsis
set -euo pipefail
task="${1:?usage: submit_ablations.sh sepsis|mortality}"
for model in gru grud odernn; do
  for inputs in values pattern all; do
    for seed in 0 1 2; do
      sbatch slurm/train.sbatch "configs/experiment/${task}_${model}.yaml" \
        "inputs=${inputs}" "seed=${seed}" ${EXTRA:-}
    done
  done
done
