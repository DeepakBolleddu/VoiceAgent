"""
labels.py — Targets for the difficulty estimator.

Two label sources, kept strictly distinct (review §B1):

1. silver_pdi — Silver Production-Difficulty Index computed from CHAT codes.
   Bootstrap target ONLY: lets baselines/estimator plumbing run before gold
   annotation exists. It is derived from transcript codes, so any model that
   *reads CHAT codes* trained on it is circular BY CONSTRUCTION — silver-PDI
   results are reported as plumbing/oracle numbers, never as findings.
   The audio-only model trained on silver_pdi is a legitimate weak-label
   system (input: audio; label: transcript codes = independent-enough source),
   which is exactly the plan's input/label separation.

2. gold labels — from the annotation protocol:
   {gold_dir}/{corpus}/{session}.ratings.csv  (utt_index, rater, condition, score)
   {gold_dir}/{corpus}/{session}.events.csv   (start_utt, end_utt, category, resolution)
   gold_rating  = mean score per utterance (audio+transcript condition)
   repair_event = 1 if utterance falls inside any OI-*/RE-ASK/NON-UP span.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Silver PDI component weights: per-token disfluency burden + pause burden.
# Includes language-agnostic/lexical signals so non-English-coded corpora
# (Korean, Mandarin, ...) are not scored as marker-free (see disfluency_lex).
PDI_COMPONENTS = {
    "filled_pauses": 1.0,          # CHAT &-um
    "lexical_fillers": 1.0,        # 음/嗯/äh written as words
    "mor_interjections": 0.75,     # intj| in %mor (lang-agnostic)
    "fragments": 1.0,
    "repetitions": 1.0,            # [/]
    "immediate_repetitions": 0.75, # doubled tokens (noisier proxy)
    "retracings": 1.5,
    "reformulations": 2.0,
    "unintelligible": 2.0,
    "untimed_pauses": 0.5,
}


def add_silver_pdi(utts: pd.DataFrame) -> pd.DataFrame:
    """Rate-normalized, winsorized, z-scored WITHIN (corpus, language) so the
    silver target cannot be solved by predicting corpus identity (review §B3)."""
    utts = utts.copy()
    tok = utts["n_tokens"].clip(lower=1)
    burden = sum(w * pd.to_numeric(utts.get(c, 0), errors="coerce").fillna(0)
                 for c, w in PDI_COMPONENTS.items()) / tok
    dur = utts["duration_s"].replace({0: np.nan})
    pause_frac = (utts["timed_pause_total_s"] / dur).fillna(0.0).clip(0, 1)
    trail = (utts["trailing_off"].astype(bool) | utts["self_interruption"].astype(bool)).astype(float)
    raw = burden + pause_frac + 0.5 * trail

    def zscore(g):
        lo, hi = g.quantile(0.01), g.quantile(0.99)
        g = g.clip(lo, hi)
        sd = g.std()
        return (g - g.mean()) / sd if sd and sd > 0 else g * 0.0

    utts["silver_pdi_raw"] = raw
    utts["silver_pdi"] = raw.groupby(
        [utts["corpus"], utts["language"]]).transform(zscore)
    return utts


def load_gold(utts: pd.DataFrame, gold_dir: str | Path) -> pd.DataFrame:
    """Merge gold ratings/events onto the index; leaves NaN where unannotated."""
    utts = utts.copy()
    utts["gold_rating"] = np.nan
    utts["repair_event"] = np.nan
    gold_dir = Path(gold_dir).expanduser()
    if not gold_dir.exists():
        print(f"[labels] no gold dir at {gold_dir} — silver only")
        return utts

    for rcsv in gold_dir.rglob("*.ratings.csv"):
        df = pd.read_csv(rcsv)
        df = df[df.get("condition", "audio+transcript") == "audio+transcript"]
        session = rcsv.name.replace(".ratings.csv", "")
        corpus = rcsv.parent.name
        mean_scores = df.groupby("utt_index")["score"].mean()
        mask = (utts["corpus"] == corpus) & (utts["session"] == session)
        utts.loc[mask, "gold_rating"] = utts.loc[mask, "utt_index"].map(mean_scores)

    breakdown_cats = {"OI-open", "OI-specific", "OI-cand", "RE-ASK", "NON-UP"}
    for ecsv in gold_dir.rglob("*.events.csv"):
        df = pd.read_csv(ecsv)
        session = ecsv.name.replace(".events.csv", "")
        corpus = ecsv.parent.name
        mask = (utts["corpus"] == corpus) & (utts["session"] == session)
        utts.loc[mask, "repair_event"] = 0.0
        for _, ev in df.iterrows():
            if ev["category"] in breakdown_cats:
                span = mask & utts["utt_index"].between(ev["start_utt"], ev["end_utt"])
                utts.loc[span, "repair_event"] = 1.0

    n_r = utts["gold_rating"].notna().sum()
    n_e = utts["repair_event"].notna().sum()
    print(f"[labels] gold: {n_r} rated utts, {n_e} event-annotated utts")
    return utts
