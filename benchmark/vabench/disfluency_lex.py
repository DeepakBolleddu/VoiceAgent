"""
disfluency_lex.py — Language-agnostic + per-language disfluency extraction.

Motivation: CHAT disfluency CODES (&-um, [/], (.)) are used inconsistently
across corpora. Korean/Mandarin/etc. transcribers instead write fillers as
plain WORDS (음, 어, 嗯) and repetitions as doubled TOKENS (옛날 옛날). An
English-code-only extractor scores these languages as marker-free, which is
false and biases every cross-lingual comparison.

Three complementary signals, ordered by robustness:
  1. immediate_repetitions  — consecutive identical tokens (language-agnostic).
  2. mor_interjections       — 'intj|' tags in the %mor tier (language-agnostic
                               where %mor exists; the cleanest filler signal).
  3. lexical_fillers         — tokens in a per-language filler list (noisier;
                               extend as needed).

All are NOISY proxies (a doubled token can be emphasis, not word-search; a
filler word can be a real demonstrative). They exist to make the SILVER target
more comparable across languages — the gold annotation remains the real label.
"""
from __future__ import annotations

# Conservative filler / hesitation lexicons. Keep to clear hesitation markers;
# avoid ambiguous demonstratives/pronouns that double as fillers.
FILLERS = {
    "eng": {"uh", "um", "er", "erm", "uhm", "mm", "hmm", "mhm", "eh"},
    "deu": {"äh", "ähm", "öh", "öhm", "hm", "ähem", "mhm"},
    "nld": {"eh", "ehm", "uh", "uhm"},
    "kor": {"음", "으음", "어", "어어", "에", "에또", "그", "저", "음음", "아"},
    "zho": {"嗯", "呃", "唉", "啊", "呐", "欸", "唔"},
    "yue": {"嗯", "呃", "啊", "唔"},
    "nan": {"嗯", "呃", "啊"},
    "spa": {"eh", "em", "este", "esto", "mmm", "mm", "pues"},
    "ell": {"ε", "εμ", "εε", "μμ", "μμμ", "χμ"},
    "ita": {"eh", "ehm", "mah", "cioè"},
    "fra": {"euh", "ben", "hein", "bah"},
}

# %mor POS tags that indicate filled-pause / interjection tokens.
MOR_FILLER_TAGS = ("intj|", "co|", "fil|", "on|")


def immediate_repetitions(tokens: list[str]) -> int:
    """Count consecutive identical tokens (case-folded). 옛날 옛날 -> 1."""
    n = 0
    prev = None
    for t in tokens:
        tl = t.lower()
        if prev is not None and tl == prev:
            n += 1
        prev = tl
    return n


def lexical_fillers(tokens: list[str], language: str) -> int:
    fset = FILLERS.get((language or "").lower()[:3], set())
    if not fset:
        return 0
    return sum(1 for t in tokens if t.lower().strip(".,!?") in fset)


def mor_interjections(mor_tier: str) -> int:
    if not mor_tier:
        return 0
    toks = mor_tier.split()
    return sum(1 for t in toks if any(t.startswith(tag) for tag in MOR_FILLER_TAGS))
