"""
tiers.py — Central speaker-tier classification (used everywhere, one source).

TalkBank tiers vary: single-participant corpora use *PAR:, but multi-party
corpora (e.g. Greek Dem@Care group conversations) use *PAR0:, *PAR1:, ... and
investigators appear as *INV:, *IN1:, *EXA:, etc. Getting this wrong silently
drops whole corpora (Greek) or merges several speakers into one.
"""
from __future__ import annotations

import re

_PAR = re.compile(r"^(PAR|CHI|PT|SUB|SPE|SPK)\d*$", re.IGNORECASE)
_INV = re.compile(r"^(INV|IN\d+|INT|EXA|EXM|CLN|RES|TEST)\d*$", re.IGNORECASE)


def is_participant(tier: str) -> bool:
    return bool(_PAR.match(str(tier)))


def is_investigator(tier: str) -> bool:
    return bool(_INV.match(str(tier)))


def is_interlocutor(tier: str) -> bool:
    """Any non-participant SPEAKER tier: investigator OR family/caregiver roles
    (Spanish PerLA HIJ/MAR/MUJ/CGS..., group-therapy facilitators, etc.).
    These are conversation partners — the 'other' side of interactional repair —
    but never difficulty targets. Defined as 'a real tier that isn't a
    participant', so it needs no exhaustive role list."""
    t = str(tier).strip()
    return bool(t) and not is_participant(t)


def participant_tiers(tiers) -> set[str]:
    return {t for t in tiers if is_participant(t)}
