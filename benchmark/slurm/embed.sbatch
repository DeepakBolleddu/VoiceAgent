#!/bin/bash
#SBATCH --job-name=va-embed
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
module load cuda  # adjust to your HPC's module system
source .venv/bin/activate
python -m vabench.features.ssl_embed --config configs/default.yaml --layers all
