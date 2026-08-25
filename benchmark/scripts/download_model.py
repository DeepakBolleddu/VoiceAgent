"""
download_model.py — Pre-fetch the SSL backbone on a LOGIN node (has internet)
so the compute node (no internet) can load it offline.

Saves a full copy to a local dir (more robust than relying on the HF cache
path being identical across nodes) AND warms the HF cache.

Run on the LOGIN node:
  python scripts/download_model.py --config configs/hpc.yaml
Then the embed job loads it offline (pbs/02 sets HF_HUB_OFFLINE=1).
"""
import argparse
from pathlib import Path

import yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/hpc.yaml")
    ap.add_argument("--out", default="models", help="local dir to save into")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).expanduser().read_text())
    model = cfg["ssl"]["model"]
    short = model.split("/")[-1]
    dst = Path(args.out).expanduser() / short
    dst.mkdir(parents=True, exist_ok=True)

    from transformers import AutoFeatureExtractor, AutoModel
    print(f"[dl] fetching {model} ...")
    fe = AutoFeatureExtractor.from_pretrained(model)
    m = AutoModel.from_pretrained(model)
    fe.save_pretrained(dst)
    m.save_pretrained(dst)
    print(f"[dl] saved to {dst}")
    print(f"[dl] now set  ssl.model: {dst}  in your config (or leave the HF id "
          f"and rely on the warmed cache + HF_HUB_OFFLINE=1).")


if __name__ == "__main__":
    main()
