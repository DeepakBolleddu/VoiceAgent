"""
ingest_annotations.py — Turn annotator CSV exports into gold labels + IAA report.

Reads all ratings_*.csv and events_*.csv dropped in an inbox, then:
  1. writes labels/{corpus}/{session}.ratings.csv and .events.csv in the exact
     shape vabench.labels.load_gold expects;
  2. computes inter-annotator agreement:
       - Tier 2 ratings: Krippendorff's alpha (ordinal) + audio-vs-transcript
         condition correlation (the independence check, review §B1);
       - Tier 1 events: alpha (nominal) on per-utterance event presence.

Usage:
  python scripts/ingest_annotations.py --inbox annotation/returned \
      --out labels
"""
import argparse
import itertools
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


def krippendorff_alpha(data: dict, level: str = "ordinal") -> float:
    """data: {unit_id: {rater: value}}. Simple, dependency-free implementation."""
    units = [u for u, r in data.items() if len(r) >= 2]
    if not units:
        return float("nan")
    vals = sorted({v for u in units for v in data[u].values()})
    vidx = {v: i for i, v in enumerate(vals)}

    def delta(a, b):
        if level == "nominal":
            return 0.0 if a == b else 1.0
        return (vidx[a] - vidx[b]) ** 2  # interval/ordinal proxy

    Do_num = Do_den = 0.0
    for u in units:
        r = list(data[u].values())
        m = len(r)
        for a, b in itertools.permutations(r, 2):
            Do_num += delta(a, b)
        Do_den += (m - 1)
    Do = Do_num / (2 * Do_den) if Do_den else 0.0

    allv = [v for u in units for v in data[u].values()]
    n = len(allv)
    De_num = sum(delta(a, b) for a, b in itertools.permutations(allv, 2))
    De = De_num / (n * (n - 1)) if n > 1 else 0.0
    return 1.0 - Do / De if De else float("nan")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--inbox", default="annotation/returned")
    ap.add_argument("--out", default="labels")
    args = ap.parse_args(argv)
    inbox, out = Path(args.inbox), Path(args.out)

    rfiles = sorted(inbox.glob("ratings_*.csv"))
    efiles = sorted(inbox.glob("events_*.csv"))
    if not rfiles:
        raise SystemExit(f"No ratings_*.csv in {inbox}")
    ratings = pd.concat([pd.read_csv(f) for f in rfiles], ignore_index=True)
    events = (pd.concat([pd.read_csv(f) for f in efiles], ignore_index=True)
              if efiles else pd.DataFrame(
                  columns=["corpus", "session", "start_utt", "end_utt",
                           "category", "resolution"]))
    print(f"[ingest] {len(ratings)} rating rows from {len(rfiles)} files, "
          f"{len(events)} event rows")

    # --- write gold in load_gold's format -----------------------------------
    for (corpus, session), g in ratings.groupby(["corpus", "session"]):
        d = out / corpus
        d.mkdir(parents=True, exist_ok=True)
        g[["utt_index", "rater", "condition", "score"]].to_csv(
            d / f"{session}.ratings.csv", index=False)
    for (corpus, session), g in events.groupby(["corpus", "session"]):
        d = out / corpus
        d.mkdir(parents=True, exist_ok=True)
        g[["start_utt", "end_utt", "category", "resolution"]].to_csv(
            d / f"{session}.events.csv", index=False)
    print(f"[ingest] wrote gold labels under {out}/")

    # --- IAA: Tier 2 ratings (ordinal), primary condition -------------------
    prim = ratings[ratings["condition"] == "audio+transcript"]
    r2 = defaultdict(dict)
    for row in prim.itertuples():
        r2[(row.corpus, row.session, row.utt_index)][row.rater] = int(row.score)
    a2 = krippendorff_alpha(r2, "ordinal")
    print(f"\n[IAA] Tier-2 rating alpha (ordinal, audio+transcript): {a2:.3f}")

    # condition independence: mean rating per unit, audio vs transcript-only
    piv = (ratings.groupby(["corpus", "session", "utt_index", "condition"])
           ["score"].mean().unstack("condition"))
    if {"audio+transcript", "transcript-only"} <= set(piv.columns):
        both = piv.dropna(subset=["audio+transcript", "transcript-only"])
        if len(both) >= 3:
            rho = both["audio+transcript"].corr(both["transcript-only"], "spearman")
            print(f"[IAA] audio-vs-transcript rating rho: {rho:.3f} "
                  f"(n={len(both)}) — high rho => acoustics dominate ratings "
                  f"(review B1): lean on repair events as primary criterion")

    # --- IAA: Tier 1 events (nominal presence per utterance) ----------------
    if len(events):
        cats = ["OI-open", "OI-specific", "OI-cand", "RE-ASK", "NON-UP"]
        # rater is not stored in events csv name-agnostically; derive from file
        ev_by_file = []
        for f in efiles:
            rater = f.stem.replace("events_", "").rsplit("_", 1)[0]
            e = pd.read_csv(f); e["rater"] = rater; ev_by_file.append(e)
        ev = pd.concat(ev_by_file, ignore_index=True)
        ev["breakdown"] = ev["category"].isin(cats)
        e1 = defaultdict(dict)
        for row in ev.itertuples():
            key = (row.corpus, row.session, row.start_utt)
            e1[key][row.rater] = int(bool(row.breakdown))
        # units annotators didn't flag are implicitly 0 — approximate on flagged set
        a1 = krippendorff_alpha(e1, "nominal")
        print(f"[IAA] Tier-1 repair-event alpha (nominal, flagged units): {a1:.3f}")

    print("\nGates: alpha>=0.7 scale up · 0.5-0.7 refine definitions & re-pilot "
          "· <0.5 revise construct before modelling.")
    print("Then: set labels.target: gold_rating in the config and re-run baselines/train.")


if __name__ == "__main__":
    main()
