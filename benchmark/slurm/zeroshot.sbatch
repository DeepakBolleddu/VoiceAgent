#!/bin/bash
#SBATCH --job-name=va-zeroshot
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
source .venv/bin/activate
# RQ3: run best ablation config on every LOLO and LOPO fold.
python - <<'PY'
import json, subprocess
splits = json.load(open("artifacts/splits/splits.json"))
for scheme in ("lolo", "lopo"):
    for fold in splits[scheme]:
        subprocess.run(["python", "-m", "vabench.train",
                        "--scheme", scheme, "--fold", fold,
                        "--adv-lang", "0.1", "--adv-pop", "0.1"], check=True)
PY
