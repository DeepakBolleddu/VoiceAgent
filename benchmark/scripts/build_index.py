"""Step 1: scan corpora -> utterance/speaker index. Step 2: splits.
Usage: python scripts/build_index.py [--config configs/default.yaml]"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vabench.corpus_index import build_index, save_index  # noqa: E402
from vabench.splits import make_splits, save_splits  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/default.yaml")
args = ap.parse_args()
cfg = yaml.safe_load(Path(args.config).read_text())

utts, speakers = build_index(cfg)
save_index(cfg, utts, speakers)
splits = make_splits(speakers, cfg)
save_splits(cfg, splits)

# timestamp-coverage summary (decides forced-alignment need — review §B5)
cov = (utts.assign(has_ts=utts["start_ms"].notna())
       .groupby("corpus")["has_ts"].mean().round(3))
print("\nTimestamp coverage by corpus (low => run forced alignment first):")
print(cov.to_string())
