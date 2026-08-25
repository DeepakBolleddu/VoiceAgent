"""
diagnose_index.py — Per-(corpus, language) health check of the built index.

Answers the questions raised by the first real run:
  * Which languages actually carry disfluency/pause CHAT codes (so silver-PDI
    has variance) vs which are word-only transcripts (silver constant -> ρ=nan)?
  * Timestamp coverage per corpus (forced-alignment need).
  * Group-label sets after normalization (spot residual label noise).
  * Speaker counts (which languages/populations can support zero-shot folds).

Usage: python scripts/diagnose_index.py --config configs/hpc.yaml
"""
import argparse
from pathlib import Path

import pandas as pd
import yaml

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vabench.tables import read_utterances          # noqa: E402
from vabench.labels import add_silver_pdi            # noqa: E402
from vabench.tiers import is_participant             # noqa: E402

MARKERS = ["filled_pauses", "lexical_fillers", "mor_interjections",
           "fragments", "repetitions", "immediate_repetitions", "retracings",
           "reformulations", "untimed_pauses", "timed_pause_total_s",
           "trailing_off", "self_interruption"]

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/hpc.yaml")
args = ap.parse_args()
cfg = yaml.safe_load(Path(args.config).expanduser().read_text())
out_root = Path(cfg["out_root"]).expanduser()

df = read_utterances(out_root / "index" / "utterances.csv")
df = add_silver_pdi(df)
par = df[df["tier"].map(is_participant)].copy()
par["has_ts"] = par["start_ms"].notna()

rows = []
present = [m for m in MARKERS if m in par.columns]
for (corp, lang), g in par.groupby(["corpus", "language"]):
    marker_rate = g[present].astype(float).sum(axis=1).mean()
    rows.append({
        "corpus": corp, "language": lang,
        "speakers": g["speaker"].nunique(),
        "utts": len(g),
        "markers_per_utt": round(marker_rate, 3),
        "silver_std": round(float(g["silver_pdi"].std(skipna=True) or 0), 3),
        "ts_cov": round(float(g["has_ts"].mean()), 3),
        "groups": ",".join(sorted(x for x in g["group"].unique() if x)),
        "example_file": (g["rel_path"].iloc[0] if "rel_path" in g else ""),
    })
rep = pd.DataFrame(rows).sort_values(["corpus", "language"])
pd.set_option("display.max_columns", None, "display.width", 240)
print(rep.drop(columns=["example_file"]).to_string(index=False))
# Show an example file for any stray-language / degenerate cell so it can be inspected.
stray = rep[(rep["markers_per_utt"] < 0.05) | (rep["speakers"] < 3)]
if len(stray):
    print("\nExample files for degenerate/stray cells (cat one to verify before allowlisting):")
    for r in stray.itertuples():
        print(f"   {r.corpus}/{r.language}: {r.example_file}")

print("\n--- FLAGS ---")
dead = rep[rep["silver_std"] < 0.05]
if len(dead):
    print("silver-PDI ~constant (word-only transcripts; rely on audio+gold, "
          "exclude from silver-target eval):")
    print("   " + ", ".join(f"{r.corpus}/{r.language}" for r in dead.itertuples()))
low_ts = rep[rep["ts_cov"] < 0.5]
if len(low_ts):
    print("low timestamp coverage (forced-align before audio/windowed work):")
    print("   " + ", ".join(f"{r.corpus}/{r.language}={r.ts_cov}" for r in low_ts.itertuples()))
small = rep[rep["speakers"] < 8]
if len(small):
    print("too few speakers for a zero-shot fold (<8):")
    print("   " + ", ".join(f"{r.corpus}/{r.language}={r.speakers}" for r in small.itertuples()))
