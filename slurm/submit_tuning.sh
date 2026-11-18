#!/usr/bin/env bash
# Tuning sweep (plan section 4, D20): 8 configurations per model and task, seed 0, `all`
# inputs, chosen by validation AUROC. The test sets are not scored (eval.test=false).
#   hidden size {64, 128} x learning rate {1e-3, 3e-4} x weight decay {0, 1e-5}
#   bash slurm/submit_tuning.sh mortality gru              # 8 cluster jobs
#   LOCAL=1 bash slurm/submit_tuning.sh mortality gru      # one after another, no cluster
#   INPUTS=values LOCAL=1 bash slurm/submit_tuning.sh mortality gru   # the plan's check (GRU)
# Then: python scripts/pick_config.py <task> <model> --write, and commit the changed configs.
set -euo pipefail
task="${1:?usage: submit_tuning.sh sepsis|mortality gru|grud|odernn}"
model="${2:?usage: submit_tuning.sh sepsis|mortality gru|grud|odernn}"
inputs="${INPUTS:-all}"
for hidden in 64 128; do
  for lr in 1.0e-3 3.0e-4; do  # YAML needs the ".0" to read a float
    for wd in 0.0 1.0e-5; do
      name="tune_${task}_${model}_${inputs}_h${hidden}_lr${lr}_wd${wd}"
      args=("configs/experiment/${task}_${model}.yaml" "inputs=${inputs}" "seed=0"
        "model.hidden_size=${hidden}" "train.lr=${lr}" "train.weight_decay=${wd}"
        "eval.test=false" "run_name=${name}" --skip-done)
      if [[ -n "${LOCAL:-}" ]]; then
        python -m missingness.train "${args[@]}"
      else
        sbatch slurm/train.sbatch "${args[@]}"
      fi
    done
  done
done
