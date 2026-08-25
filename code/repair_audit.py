"""
repair_audit.py — Repair-density + timestamp-coverage audit (plan §13 item 2,
review §D1).

Answers, per corpus/language, the empirical questions behind the v1 construct
decision:
  1. How much investigator speech is there at all? (INV turn share)
  2. How often does INV ask questions / re-ask (candidate other-initiated repair)?
  3. How often does PAR produce NTRI markers ("huh?", "what?", "pardon?" ...)
     — candidate PAR-side comprehension trouble?
  4. What fraction of utterances carry usable media-bullet timestamps?
     (decides whether forced alignment is required — review §B5)

Candidate-repair heuristics are RECALL-ORIENTED screens for the audit, not
labels. The annotation protocol (annotation/annotation_protocol_v0.md) is what
produces labels.

Usage:
  python repair_audit.py <root_dir> [--corpus NAME] [--out report.csv]
  Recursively finds *.cha under root_dir.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

from chat_parser import ChatDocument, parse_cha_file, utterance_feature_row

# Participant-side other-initiation-of-repair (NTRI) markers, per language.
# Extend as corpora arrive; these are screens, not exhaustive lists.
NTRI_MARKERS = {
    "eng": {"huh", "what", "pardon", "sorry", "eh", "hm", "again"},
    "deu": {"was", "wie", "bitte", "hä"},
    "spa": {"qué", "cómo", "perdón", "mande"},
    "fra": {"quoi", "comment", "pardon", "hein"},
    "zho": {"什么", "啊", "嗯", "再说一遍"},
    "ell": {"τι", "πώς", "ορίστε"},
    "kor": {"네", "뭐", "예"},
}
INVESTIGATOR_CODES = {"INV", "IN1", "IN2", "EXA", "EXM", "INT", "CLN"}
PARTICIPANT_CODES = {"PAR", "CHI", "PT", "SUB"}


def jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb) if sa | sb else 0.0


def is_ntri(tokens: list[str], is_question: bool, lang3: str) -> bool:
    """Short question-like turn built (mostly) from repair-initiation markers."""
    if not tokens or len(tokens) > 4:
        return False
    markers = NTRI_MARKERS.get(lang3, NTRI_MARKERS["eng"])
    hit = sum(1 for t in tokens if t.lower().strip("?.!,") in markers)
    return is_question and hit >= 1


def audit_document(doc: ChatDocument, corpus: str = "") -> dict:
    utts = doc.utterances
    lang3 = (doc.languages[0][:3].lower() if doc.languages else "eng")
    inv = [u for u in utts if u.speaker in INVESTIGATOR_CODES]
    par = [u for u in utts if u.speaker in PARTICIPANT_CODES]

    inv_questions = [u for u in inv if u.is_question]
    # Re-ask: an INV question lexically similar to an earlier INV question
    # within the last 6 turns (candidate repeat/rephrase of a failed question).
    reasks = 0
    recent: list[tuple[int, list[str]]] = []
    for u in inv_questions:
        toks = [t.lower() for t in u.clean_tokens]
        for (j, prev) in recent:
            if u.index - j <= 6 and jaccard(toks, prev) >= 0.5:
                reasks += 1
                break
        recent.append((u.index, toks))

    par_ntri = sum(1 for u in par
                   if is_ntri(u.clean_tokens, u.is_question, lang3))

    # Candidate repair sequence: PAR trouble-marked turn (or NTRI) followed
    # within 2 turns by an INV question — a screen for breakdown-and-fix.
    candidate_sequences = 0
    for i, u in enumerate(utts):
        if u.speaker in PARTICIPANT_CODES and (
            u.unintelligible or u.terminator_flags["trailing_off"]
            or is_ntri(u.clean_tokens, u.is_question, lang3)
        ):
            for v in utts[i + 1: i + 3]:
                if v.speaker in INVESTIGATOR_CODES and v.is_question:
                    candidate_sequences += 1
                    break

    with_ts = sum(1 for u in utts if u.has_timestamp)
    return {
        "file": Path(doc.path).name,
        "corpus": corpus,
        "language": doc.languages[0] if doc.languages else "?",
        "n_utterances": len(utts),
        "n_par_utts": len(par),
        "n_inv_utts": len(inv),
        "inv_turn_share": round(len(inv) / len(utts), 3) if utts else 0.0,
        "inv_questions": len(inv_questions),
        "inv_reasks": reasks,
        "par_ntri": par_ntri,
        "candidate_repair_sequences": candidate_sequences,
        "timestamp_coverage": round(with_ts / len(utts), 3) if utts else 0.0,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", help="Directory to scan recursively for .cha files")
    ap.add_argument("--corpus", default="", help="Corpus label for the report")
    ap.add_argument("--out", default="repair_audit_report.csv")
    ap.add_argument("--features-out", default="",
                    help="Also write the per-utterance feature table (CSV)")
    args = ap.parse_args(argv)

    files = sorted(Path(args.root).rglob("*.cha"))
    if not files:
        sys.exit(f"No .cha files under {args.root}")

    rows, feat_rows = [], []
    for f in files:
        doc = parse_cha_file(f)
        rows.append(audit_document(doc, corpus=args.corpus))
        if args.features_out:
            feat_rows += [utterance_feature_row(doc, u, corpus=args.corpus)
                          for u in doc.utterances]

    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    if args.features_out and feat_rows:
        with open(args.features_out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=feat_rows[0].keys())
            w.writeheader(); w.writerows(feat_rows)

    # Console summary per language
    by_lang: Counter = Counter()
    agg: dict[str, Counter] = {}
    for r in rows:
        lang = r["language"]
        by_lang[lang] += 1
        c = agg.setdefault(lang, Counter())
        for k in ("n_utterances", "n_inv_utts", "inv_questions",
                  "inv_reasks", "par_ntri", "candidate_repair_sequences"):
            c[k] += r[k]
        c["ts_cov_sum"] += r["timestamp_coverage"]

    print(f"\n{'lang':8}{'files':>6}{'utts':>8}{'INVutts':>9}{'INV?s':>7}"
          f"{'reasks':>8}{'NTRI':>6}{'repairSeq':>10}{'tsCov':>7}")
    for lang, n in by_lang.items():
        c = agg[lang]
        print(f"{lang:8}{n:>6}{c['n_utterances']:>8}{c['n_inv_utts']:>9}"
              f"{c['inv_questions']:>7}{c['inv_reasks']:>8}{c['par_ntri']:>6}"
              f"{c['candidate_repair_sequences']:>10}"
              f"{c['ts_cov_sum'] / n:>7.2f}")
    print(f"\nWrote {args.out}"
          + (f" and {args.features_out}" if args.features_out else ""))


if __name__ == "__main__":
    main()
