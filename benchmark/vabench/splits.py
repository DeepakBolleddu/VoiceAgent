"""
splits.py — Speaker-independent splits, decided BEFORE any modelling (plan §6).

Produces {out_root}/splits/splits.json with:
  iid:  train/dev/test speaker lists, stratified by (language, population, group)
  lolo: leave-one-language-out folds  — zero-shot language transfer (RQ3)
  lopo: leave-one-population-out folds — zero-shot population transfer (RQ3)

All membership is at SPEAKER level (longitudinal sessions stay together —
review §B6). Caveat carried from review §B3: held-out language/population is
also held-out corpus; report as cross-corpus transfer.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def _stratified_speaker_split(speakers: pd.DataFrame, test_frac: float,
                              dev_frac: float, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    train, dev, test = [], [], []
    strata = speakers.groupby(["language", "population", "group"], dropna=False)
    for _, g in strata:
        ids = g["speaker"].tolist()
        rng.shuffle(ids)
        n = len(ids)
        n_test = max(1, round(n * test_frac)) if n >= 3 else 0
        n_dev = max(1, round(n * dev_frac)) if n >= 5 else 0
        test += ids[:n_test]
        dev += ids[n_test:n_test + n_dev]
        train += ids[n_test + n_dev:]
    return {"train": sorted(train), "dev": sorted(dev), "test": sorted(test)}


def _leave_one_out_folds(speakers: pd.DataFrame, col: str,
                         min_speakers: int) -> dict:
    folds = {}
    for val, g in speakers.groupby(col):
        held = g["speaker"].tolist()
        if len(held) < min_speakers:
            print(f"[splits] skip {col}={val}: only {len(held)} speakers")
            continue
        rest = speakers.loc[speakers[col] != val, "speaker"].tolist()
        folds[str(val)] = {"train": sorted(rest), "test": sorted(held)}
    return folds


def make_splits(speakers: pd.DataFrame, cfg: dict) -> dict:
    s = cfg["splits"]
    splits = {
        "iid": _stratified_speaker_split(speakers, s["test_frac"],
                                         s["dev_frac"], s["seed"]),
        "lolo": _leave_one_out_folds(speakers, "language",
                                     s["min_speakers_zero_shot"]),
        "lopo": _leave_one_out_folds(speakers, "population",
                                     s["min_speakers_zero_shot"]),
    }
    # hard guarantees
    iid = splits["iid"]
    assert not (set(iid["train"]) & set(iid["test"])), "speaker leakage: train∩test"
    assert not (set(iid["train"]) & set(iid["dev"])), "speaker leakage: train∩dev"
    assert not (set(iid["dev"]) & set(iid["test"])), "speaker leakage: dev∩test"
    return splits


def save_splits(cfg: dict, splits: dict) -> Path:
    out = Path(cfg["out_root"]).expanduser() / "splits"
    out.mkdir(parents=True, exist_ok=True)
    p = out / "splits.json"
    p.write_text(json.dumps(splits, indent=2))
    iid = splits["iid"]
    print(f"[splits] iid: {len(iid['train'])} train / {len(iid['dev'])} dev / "
          f"{len(iid['test'])} test speakers; "
          f"{len(splits['lolo'])} LOLO folds, {len(splits['lopo'])} LOPO folds -> {p}")
    return p


def attach_split(utts: pd.DataFrame, splits: dict, scheme: str = "iid",
                 fold: str | None = None) -> pd.DataFrame:
    """Label each utterance row with its split under a scheme/fold."""
    utts = utts.copy()
    if scheme == "iid":
        m = {}
        for part in ("train", "dev", "test"):
            m.update({s: part for s in splits["iid"][part]})
    else:
        f = splits[scheme][fold]
        m = {s: "train" for s in f["train"]}
        m.update({s: "test" for s in f["test"]})
    utts["split"] = utts["speaker"].map(m)
    return utts[utts["split"].notna()]
