#!/bin/bash
#SBATCH --job-name=va-train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --array=0-4
#SBATCH --output=logs/%x_%A_%a.out
source .venv/bin/activate
# Ablation grid: frozen probe / lang-only / pop-only / both / both+speaker
ARGS=(
  "--adv-lang 0.0 --adv-pop 0.0 --adv-spk 0.0 --run-name probe"
  "--adv-lang 0.1 --adv-pop 0.0 --adv-spk 0.0 --run-name adv_lang"
  "--adv-lang 0.0 --adv-pop 0.1 --adv-spk 0.0 --run-name adv_pop"
  "--adv-lang 0.1 --adv-pop 0.1 --adv-spk 0.0 --run-name adv_both"
  "--adv-lang 0.1 --adv-pop 0.1 --adv-spk 0.05 --run-name adv_all"
)
python -m vabench.train --config configs/default.yaml --scheme iid ${ARGS[$SLURM_ARRAY_TASK_ID]}
