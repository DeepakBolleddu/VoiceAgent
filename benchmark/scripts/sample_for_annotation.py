"""
sample_for_annotation.py — Build annotation manifests (the gold-label critical path).

Selects participant utterances to annotate, each with surrounding turn context
(needed to judge interactional repair), stratified so the pilot covers the range
of difficulty and severity and is enriched for repair sequences.

Per language it writes annotation/manifests/{lang}.json — a list of items the
annotator.html tool consumes. Keyed by (corpus, session, utt_index) so returned
labels merge straight back onto the index via vabench.labels.

Usage:
  python scripts/sample_for_annotation.py --config configs/hpc.yaml \
      --languages eng,zho,deu --per-language 200 --context 2
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vabench.tables import read_utterances                 # noqa: E402
from vabench.labels import add_silver_pdi                   # noqa: E402
from vabench.tiers import is_participant, is_investigator   # noqa: E402


def repair_adjacent(group_df: pd.DataFrame, idx: int, target_tier: str) -> bool:
    """Heuristic enrichment: is this participant turn part of an exchange where
    repair could occur? True if an adjacent turn is by a DIFFERENT speaker tier
    (an interlocutor: INV, another PARn in group talk, or family HIJ/MAR/...),
    especially a question, or the target itself shows trouble markers.
    Screen only; annotators make the real call."""
    window = group_df[(group_df["utt_index"] >= idx - 1) &
                      (group_df["utt_index"] <= idx + 1)]
    other = window[window["tier"] != target_tier]
    other_turn = len(other) > 0
    other_q = bool(other["is_question"].any()) if other_turn else False
    tgt = window[window["utt_index"] == idx]
    trouble = bool((tgt["unintelligible"].astype(float).gt(0) |
                    tgt["trailing_off"].astype(bool) |
                    tgt["self_interruption"].astype(bool)).any())
    return other_q or (other_turn and trouble) or trouble


def context_turns(sess_df: pd.DataFrame, idx: int, k: int) -> list[dict]:
    ctx = sess_df[(sess_df["utt_index"] >= idx - k) &
                  (sess_df["utt_index"] <= idx + 1)]
    return [{"tier": r.tier, "utt_index": int(r.utt_index),
             "text": r.text, "target": int(r.utt_index) == idx}
            for r in ctx.itertuples()]


def sample_language(df: pd.DataFrame, n: int, k: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    par = df[df["tier"].map(is_participant) & (df["n_tokens"] > 0)].copy()
    if par.empty:
        return []
    # difficulty strata: silver tertiles if it varies, else single stratum
    if par["silver_pdi"].std(skipna=True) and par["silver_pdi"].notna().any():
        par["stratum"] = pd.qcut(par["silver_pdi"].rank(method="first"),
                                 q=min(3, par["silver_pdi"].nunique()),
                                 labels=False, duplicates="drop")
    else:
        par["stratum"] = 0
    par["grp"] = par["group"].replace("", "Unknown")

    items, seen = [], set()
    # enrich: half the budget from repair-adjacent utterances
    for enrich in (True, False):
        strata = par.groupby(["grp", "stratum"], dropna=False)
        for _, g in strata:
            take = max(1, (n // 2) // max(1, len(par[["grp", "stratum"]].drop_duplicates())))
            cand = g.sample(frac=1.0, random_state=int(rng.integers(1e9)))
            for r in cand.itertuples():
                if len(items) >= n:
                    break
                key = (r.corpus, r.session, int(r.utt_index))
                if key in seen:
                    continue
                sess = par[(par["corpus"] == r.corpus) & (par["session"] == r.session)]
                full_sess = df[(df["corpus"] == r.corpus) & (df["session"] == r.session)]
                if enrich and not repair_adjacent(full_sess, int(r.utt_index), r.tier):
                    continue
                seen.add(key)
                items.append({
                    "item_id": f"{r.corpus}|{r.session}|{int(r.utt_index)}",
                    "corpus": r.corpus, "session": r.session,
                    "utt_index": int(r.utt_index), "task": r.task,
                    "group": r.group, "language": r.language,
                    "audio_stem": getattr(r, "stem", r.session), "audio_task": r.task,
                    "start_ms": None if pd.isna(r.start_ms) else int(r.start_ms),
                    "end_ms": None if pd.isna(r.end_ms) else int(r.end_ms),
                    "context": context_turns(full_sess, int(r.utt_index), k),
                })
            if len(items) >= n:
                break
        if len(items) >= n:
            break
    return items[:n]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/hpc.yaml")
    ap.add_argument("--languages", default="", help="comma list, e.g. eng,zho,deu; blank = all")
    ap.add_argument("--per-language", type=int, default=200)
    ap.add_argument("--context", type=int, default=2)
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(Path(args.config).expanduser().read_text())
    out_root = Path(cfg["out_root"]).expanduser()
    df = add_silver_pdi(read_utterances(out_root / "index" / "utterances.csv"))

    langs = ([l.strip() for l in args.languages.split(",") if l.strip()]
             or sorted(df["language"].unique()))
    outdir = Path("annotation/manifests")
    outdir.mkdir(parents=True, exist_ok=True)
    summary = []
    for lang in langs:
        sub = df[df["language"] == lang]
        items = sample_language(sub, args.per_language, args.context, args.seed)
        (outdir / f"{lang}.json").write_text(json.dumps(
            {"language": lang, "context_k": args.context, "items": items}, indent=1))
        n_audio = sum(1 for it in items if it["start_ms"] is not None)
        summary.append((lang, len(items), n_audio))
        print(f"[sample] {lang}: {len(items)} items ({n_audio} with timestamps) "
              f"-> {outdir / f'{lang}.json'}")
    print("\nNext: open annotation/annotator.html, load a manifest, annotate, "
          "export CSVs, then scripts/ingest_annotations.py")


if __name__ == "__main__":
    main()
