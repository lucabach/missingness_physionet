#!/bin/bash
# One training run on the cluster. Usage:
#   sbatch slurm/train.sbatch configs/experiment/sepsis_gru.yaml inputs=pattern seed=1
#
# TODO (week 2/3 owner): check partition/account names and GPU syntax for the ETH student
# cluster in its documentation and fix the #SBATCH lines below; record them in docs/cluster.md.
#SBATCH --job-name=missingness
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs
source .venv/bin/activate
export MISSINGNESS_DATA_DIR="${MISSINGNESS_DATA_DIR:-$PWD/data}"
export MISSINGNESS_OUTPUT_DIR="${MISSINGNESS_OUTPUT_DIR:-$PWD/outputs}"

python -m missingness.train "$@"
