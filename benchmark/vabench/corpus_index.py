"""
corpus_index.py — Scan corpora, parse every .cha, emit the benchmark index.

Outputs (under {out_root}/index/):
  utterances.csv — one row per utterance: identity keys + CHAT-derived
                   production markers (input side, plan §5) + timestamps.
  speakers.csv   — one row per speaker: language, population, group (dx),
                   age, sex, #sessions. THE unit for all splitting.

Speaker identity: corpus + speaker_regex(file stem). This is what prevents
longitudinal leakage (review §B6): Pitt 001-0.cha and 001-4.cha collapse to
speaker Pitt:001.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from .chat_parser import parse_cha_file, utterance_feature_row
from .tables import normalize_group
from .tiers import is_participant, participant_tiers

DEFAULT_SPEAKER_REGEX = r"^(?P<sid>[A-Za-z]*\d+[a-zA-Z]?)"

# @ID: lang|corpus|code|age|sex|group|SES|role|education|custom
ID_FIELDS = ["language", "corpus", "code", "age", "sex", "group",
             "ses", "role", "education", "custom"]


def parse_id_header(raw: str) -> dict:
    parts = [p.strip() for p in raw.split("|")]
    parts += [""] * (len(ID_FIELDS) - len(parts))
    return dict(zip(ID_FIELDS, parts))


def participant_id_info(doc) -> dict:
    """@ID row for the participant tier (PAR/CHI/...), if present."""
    for raw in doc.id_headers:
        info = parse_id_header(raw)
        if info["code"] in {"PAR", "CHI", "PT", "SUB"}:
            return info
    return {k: "" for k in ID_FIELDS}


def speaker_key(corpus: str, stem: str, regex: str, language: str = "",
                subcorpus: str = "") -> str:
    """Speaker identity = corpus + language + sub-corpus + sid.
    - language: identical stems across languages (English 001-0 vs Mandarin
      001-0) must not merge.
    - subcorpus: sub-corpora within one language (e.g. Mandarin Lu vs Chou)
      may reuse ID numbers; keep them apart. Group (Control/Dementia) is NOT
      part of the key — a speaker has one dx, and group-collisions are instead
      detected and warned about in build_index."""
    m = re.match(regex, stem)
    sid = m.group("sid") if m and m.groupdict().get("sid") else stem
    lang = (language or "x").lower()[:3]
    sub = f":{subcorpus}" if subcorpus else ""
    return f"{corpus}:{lang}{sub}:{sid}"


def subcorpus_from_rel(rel_to_corpus: Path) -> str:
    """First directory under the corpus root, unless it is a group/task dir
    (e.g. English/Pitt/... -> 'Pitt'; Control/cookie/x.cha -> '')."""
    parts = rel_to_corpus.parts
    if len(parts) > 1 and parts[0].lower() not in PATH_GROUP_TOKENS:
        return parts[0]
    return ""


# Group/dx tokens that may appear as a directory in the TalkBank tree
# (e.g. .../Pitt/Control/cookie/, Greek .../long/AD/, .../MCI/, .../HC/).
PATH_GROUP_TOKENS = {"control", "hc", "dementia", "ad", "mci", "probablead",
                     "possiblead", "aws", "awns", "cws", "cwns", "aphasia",
                     "tbi", "patient", "patient_ad", "patient_mci", "clinical"}


def group_from_path(rel_path: str) -> str:
    """Deepest dx folder wins; skip parts[0] (the population dir like
    'dementia'/'fluency') so it isn't mistaken for a diagnosis."""
    parts = Path(rel_path).parts[1:]            # drop population dir
    for part in reversed(parts):                # nearest-to-file first
        if part.lower() in PATH_GROUP_TOKENS:
            return part
    return ""


def task_from_path(rel_path: str, corpus_path: str) -> str:
    """Directory just above the file, if it is not the group dir (e.g. 'cookie')."""
    parts = Path(rel_path).parts
    if len(parts) >= 2 and parts[-2].lower() not in PATH_GROUP_TOKENS:
        return parts[-2]
    return ""


def build_index(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_root = Path(cfg["data_root"]).expanduser()
    utt_rows = []
    for corpus, spec in cfg["corpora"].items():
        cdir = data_root / spec["path"]
        files = sorted(cdir.rglob("*.cha"))
        if not files:
            print(f"[warn] no .cha files for corpus {corpus} at {cdir}")
            continue
        regex = spec.get("speaker_regex", DEFAULT_SPEAKER_REGEX)
        keep_langs = spec.get("keep_langs")        # optional allowlist, e.g. [deu]
        skipped_lang = 0
        for f in files:
            doc = parse_cha_file(f)
            pinfo = participant_id_info(doc)
            lang = pinfo["language"] or (doc.languages[0] if doc.languages else "")
            if not lang:                       # config fallback before it's used in the key
                lang = spec.get("lang", "")
            if keep_langs and lang.lower()[:3] not in [x.lower()[:3] for x in keep_langs]:
                skipped_lang += 1              # stray-language file (e.g. eng inside German)
                continue
            sub = subcorpus_from_rel(f.relative_to(cdir))
            rel = str(f.relative_to(data_root))
            # session must be unique per FILE (stems repeat across folders);
            # derived from the corpus-relative path, slash->underscore.
            session_key = str(f.relative_to(cdir).with_suffix("")).replace("/", "_")
            group = normalize_group(pinfo["group"] or group_from_path(rel))
            lang3 = (lang or "x").lower()[:3]
            subk = f":{sub}" if sub else ""
            # Multi-party? (e.g. Greek Dem@Care: PAR0..PAR6 in one file). If so,
            # each participant tier is a DISTINCT speaker scoped to the session;
            # otherwise the single participant is filename-scoped (longitudinal).
            ptiers = participant_tiers(u.speaker for u in doc.utterances)
            multiparty = len(ptiers) > 1
            for u in doc.utterances:
                row = utterance_feature_row(doc, u, corpus=corpus, language=lang)
                row["tier"] = row.pop("speaker")   # PAR/PAR1/INV/... tier code
                if multiparty and is_participant(row["tier"]):
                    skey = f"{corpus}:{lang3}{subk}:{f.stem}_{row['tier']}"
                elif multiparty:                   # investigator in a group session
                    skey = f"{corpus}:{lang3}{subk}:{f.stem}_{row['tier']}"
                else:
                    skey = speaker_key(corpus, f.stem, regex, lang, sub)
                row.update({
                    "population": spec["population"],
                    "speaker": skey,
                    "session": session_key,
                    "stem": f.stem,
                    "rel_path": rel,
                    "task": task_from_path(rel, spec["path"]),
                    "group": group,   # dx label — VALIDATION ONLY, never a training target
                    "age": pinfo["age"].rstrip(";"),
                    "sex": pinfo["sex"],
                    "multiparty": multiparty,
                })
                utt_rows.append(row)
        msg = f"[index] {corpus}: {len(files)} files"
        if skipped_lang:
            msg += f" ({skipped_lang} skipped by keep_langs={keep_langs})"
        print(msg)

    utts = pd.DataFrame(utt_rows)
    if utts.empty:
        raise SystemExit("Index is empty — check data_root/corpora paths.")
    # utt_id MUST be globally unique. File stems repeat across folders (Pitt
    # cookie/002-0 vs fluency/002-0; Greek long/32 vs short/32), so the id is
    # built from the full rel_path, never the stem alone.
    utts["utt_id"] = (utts["corpus"] + "/"
                      + utts["rel_path"].str.replace(r"\.cha$", "", regex=True)
                      + "#" + utts["utt_index"].astype(str))
    dup = utts["utt_id"].duplicated()
    if dup.any():
        raise SystemExit(f"FATAL: {dup.sum()} duplicate utt_ids — merge keys "
                         f"would cross-join. Example: {utts.loc[dup, 'utt_id'].iloc[0]}")

    # Collision guard: one speaker key spanning >1 dx group means two different
    # people were merged (ID reuse across group folders). Inspect before trusting
    # any split — either refine speaker_regex or split that corpus entry.
    is_par = utts["tier"].map(is_participant)
    par_check = utts[is_par]
    multi_grp = (par_check[par_check["group"].astype(str).str.len() > 0]
                 .groupby("speaker")["group"].nunique())
    collisions = multi_grp[multi_grp > 1]
    if len(collisions):
        print(f"[WARN] {len(collisions)} speaker keys span multiple dx groups "
              f"(longitudinal dx change is expected in Pitt; only worry if a "
              f"non-longitudinal corpus appears here):")
        for s in collisions.index[:10]:
            groups = sorted(par_check.loc[par_check['speaker'] == s, 'group'].unique())
            print(f"    {s}: {groups}")

    par = utts[is_par]
    speakers = (
        par.groupby("speaker")
        .agg(corpus=("corpus", "first"), language=("language", "first"),
             population=("population", "first"), group=("group", "first"),
             age=("age", "first"), sex=("sex", "first"),
             n_sessions=("session", "nunique"),
             n_utts=("utt_id", "count"))
        .reset_index()
    )
    return utts, speakers


def save_index(cfg: dict, utts: pd.DataFrame, speakers: pd.DataFrame) -> Path:
    out = Path(cfg["out_root"]).expanduser() / "index"
    out.mkdir(parents=True, exist_ok=True)
    utts.to_csv(out / "utterances.csv", index=False)
    speakers.to_csv(out / "speakers.csv", index=False)
    print(f"[index] {len(utts)} utterances, {speakers.shape[0]} speakers -> {out}")
    return out
