"""
tables.py — Safe I/O + label normalization for the index.

Critical: several TalkBank language codes and tokens collide with pandas'
default NA strings. The worst is Taiwanese, whose language code is 'nan'
(ISO 639-3 Min Nan Chinese) — read_csv turns "nan" into a float NaN and the
whole corpus loses its language. We therefore read string columns with
keep_default_na disabled and explicit str dtype.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

STR_COLS = ["file", "corpus", "language", "population", "speaker", "session",
            "stem", "rel_path", "task", "group", "sex", "tier", "text", "utt_id"]

# Canonical diagnosis labels (folded case-insensitively).
GROUP_NORM = {
    "control": "Control", "hc": "Control", "normal": "Control",
    "mci": "MCI", "probablemci": "MCI", "patient_mci": "MCI", "pd-mci": "MCI",
    "probablead": "ProbableAD", "probable": "ProbableAD",
    "possiblead": "PossibleAD", "possible": "PossibleAD",
    "patient_ad": "AD", "ad": "AD", "dementia": "Dementia",
    "memory": "Memory", "vascular": "Vascular", "other": "Other",
    "aws": "AWS", "awns": "AWNS", "cws": "CWS", "cwns": "CWNS",
}

# Rough severity ranks for known-groups (criterion) validation only. Higher =
# more impaired. Used to check monotonicity of predicted difficulty, never as
# a training target.
GROUP_SEVERITY = {
    "Control": 0, "AWNS": 0, "CWNS": 0,
    "MCI": 1, "AWS": 1, "CWS": 1, "Memory": 1, "Other": 1,
    "PossibleAD": 2, "Vascular": 2,
    "ProbableAD": 3, "AD": 3, "Dementia": 3,
}


def normalize_group(g: str) -> str:
    if not isinstance(g, str) or not g.strip():
        return ""
    return GROUP_NORM.get(g.strip().lower(), g.strip())


def read_utterances(path: str | Path) -> pd.DataFrame:
    """Read utterances.csv without corrupting 'nan'-like string codes."""
    dtypes = {c: "string" for c in STR_COLS}
    df = pd.read_csv(path, dtype=dtypes, keep_default_na=False,
                     na_values=[""], low_memory=False)
    # restore real numerics
    for c in ["start_ms", "end_ms", "duration_s", "speech_rate_tok_per_s",
              "timed_pause_total_s", "silver_pdi", "silver_pdi_raw",
              "gold_rating", "repair_event"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in STR_COLS:
        if c in df.columns:
            df[c] = df[c].fillna("").astype(str)
    if "group" in df.columns:
        df["group"] = df["group"].map(normalize_group)
    return df
