"""
run_baseline.py — P1 baseline suite (transcript-feature tier).

Baselines (plan §8 "baselines and ablations"):
  B0 length-only     : n_tokens + duration (sanity floor)
  B1 interpretable   : CHAT-derived production markers + rate features
  B2 B1 + language OH: does knowing the language help? (invariance probe)

NOTE these transcript-feature baselines target gold labels legitimately, but
against silver_pdi they are ORACLE/plumbing numbers (silver is computed from
the same codes — review §B1/§2 of labels.py). The audio tier (SSL probe,
estimator) is where headline results come from.

Usage:
  python -m vabench.baselines.run_baseline --config configs/default.yaml \
      [--scheme iid|lolo|lopo] [--target silver_pdi|gold_rating]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..evaluate import print_report, report
from ..labels import add_silver_pdi, load_gold
from ..splits import attach_split

FEAT_B0 = ["n_tokens", "duration_s"]
FEAT_B1 = FEAT_B0 + [
    "speech_rate_tok_per_s", "filled_pauses", "lexical_fillers",
    "mor_interjections", "fragments", "repetitions", "immediate_repetitions",
    "retracings", "reformulations", "untimed_pauses", "timed_pause_total_s",
    "unintelligible", "trailing_off", "self_interruption",
]


def matrix(df: pd.DataFrame, feats: list[str], add_lang: bool = False,
           enc: OneHotEncoder | None = None):
    X = df[feats].astype(float).fillna(0.0).to_numpy()
    if add_lang:
        if enc is None:
            enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            enc.fit(df[["language"]])
        X = np.hstack([X, enc.transform(df[["language"]])])
    return X, enc


def run_one(tr: pd.DataFrame, te: pd.DataFrame, feats: list[str],
            target: str, add_lang: bool, model: str = "gbm") -> pd.Series:
    tr = tr.dropna(subset=[target])
    Xtr, enc = matrix(tr, feats, add_lang)
    Xte, _ = matrix(te, feats, add_lang, enc)
    sc = StandardScaler().fit(Xtr)
    reg = (HistGradientBoostingRegressor(random_state=0) if model == "gbm"
           else Ridge(alpha=1.0))
    reg.fit(sc.transform(Xtr), tr[target].to_numpy())
    return pd.Series(reg.predict(sc.transform(Xte)), index=te.index)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--scheme", default="iid", choices=["iid", "lolo", "lopo"])
    ap.add_argument("--target", default=None)
    ap.add_argument("--model", default="gbm", choices=["gbm", "ridge"])
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(Path(args.config).expanduser().read_text())
    target = args.target or cfg["labels"]["target"]
    out_root = Path(cfg["out_root"]).expanduser()

    from ..tables import read_utterances
    utts = read_utterances(out_root / "index" / "utterances.csv")
    utts = add_silver_pdi(utts)
    utts = load_gold(utts, Path(cfg["labels"]["gold_dir"]).expanduser())
    from ..tiers import is_participant
    utts = utts[utts["n_tokens"] > 0]
    # participant speech only (production locus); handles PAR and PAR0..PARn
    utts = utts[utts["tier"].map(is_participant)]
    splits = json.loads((out_root / "splits" / "splits.json").read_text())

    folds = ([None] if args.scheme == "iid"
             else list(splits[args.scheme].keys()))
    results = {}
    for fold in folds:
        d = attach_split(utts, splits, args.scheme, fold)
        tr, te = d[d["split"] == "train"], d[d["split"] == "test"]
        tag = f"{args.scheme}" + (f"/{fold}" if fold else "")
        for name, feats, add_lang in [("B0_length", FEAT_B0, False),
                                      ("B1_interp", FEAT_B1, False),
                                      ("B2_interp+lang", FEAT_B1, True)]:
            te = te.copy()
            te["pred"] = run_one(tr, te, feats, target, add_lang, args.model)
            rep = report(te, "pred", target)
            results[f"{tag}:{name}"] = rep
            print_report(rep, f"{tag} {name} (target={target}, {args.model})")

    out = out_root / "reports"
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"baselines_{args.scheme}_{target}.json"
    p.write_text(json.dumps(results, indent=2, default=float))
    print(f"\n[baselines] wrote {p}")


if __name__ == "__main__":
    main()
