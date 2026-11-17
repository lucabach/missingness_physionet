#!/usr/bin/env bash
# Tuning sweep (plan section 4): 8 configurations per model and task, seed 0, `all` inputs,
# chosen by validation AUROC. PROPOSED grid, to confirm with the team before protocol-v1:
#   hidden size {64, 128} x learning rate {1e-3, 3e-4} x weight decay {0, 1e-5}
#   bash slurm/submit_tuning.sh mortality gru        # 8 jobs
#   LOCAL=1 bash slurm/submit_tuning.sh mortality gru   # run one after another, no cluster
# Pick the best run with: python scripts/collect_results.py, then compare val_auroc in each
# outputs/tune_*/metrics.json. Also run once with INPUTS=values for GRU (the plan's check).
set -euo pipefail
task="${1:?usage: submit_tuning.sh sepsis|mortality gru|grud|odernn}"
model="${2:?usage: submit_tuning.sh sepsis|mortality gru|grud|odernn}"
inputs="${INPUTS:-all}"
for hidden in 64 128; do
  for lr in 1.0e-3 3.0e-4; do  # YAML needs the ".0" to read a float
    for wd in 0.0 1.0e-5; do
      args=("configs/experiment/${task}_${model}.yaml" "inputs=${inputs}" "seed=0"
        "model.hidden_size=${hidden}" "train.lr=${lr}" "train.weight_decay=${wd}"
        "run_name=tune_${task}_${model}_${inputs}_h${hidden}_lr${lr}_wd${wd}")
      if [[ -n "${LOCAL:-}" ]]; then
        python -m missingness.train "${args[@]}"
      else
        sbatch slurm/train.sbatch "${args[@]}"
      fi
    done
  done
done
