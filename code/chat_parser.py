"""
chat_parser.py — Lightweight CHAT (.cha) parser for TalkBank corpora.

Extracts, per utterance:
  speaker tier (*PAR:, *INV:, ...), raw text, media-bullet timestamps,
  disfluency/pause/retracing codes, and dependent tiers (%mor, %gra, ...).

Self-contained (stdlib only). Targets exactly the codes needed for the
communicative-difficulty input features (plan §5/§13):
  [/] repetition, [//] retracing, [///] complex reformulation,
  &-uh / &-um filled pauses, &+frag phonological fragments,
  (.) (..) (...) untimed pauses, (2.5) timed pauses,
  +... trailing off, +//. self-interruption, +/. interrupted by other,
  xxx unintelligible, [: replacement], [* error] codes.

Usage:
  from chat_parser import parse_cha_file
  doc = parse_cha_file("path/to/file.cha")
  for utt in doc.utterances: ...

Note: for production-scale work also consider `pylangacq` (mature CHAT
reader). This parser is kept dependency-free and auditable so every
extraction rule is visible and testable.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

BULLET = "\x15"  # NAK char delimiting media bullets: \x15start_end\x15

# --- regexes -----------------------------------------------------------------
RE_BULLET = re.compile(rf"{BULLET}(\d+)_(\d+){BULLET}")
RE_FILLED_PAUSE = re.compile(r"&-(\w+)")
RE_FRAGMENT = re.compile(r"&\+(\S+)")
RE_TIMED_PAUSE = re.compile(r"\((\d+(?:\.\d+)?)\)")          # (2.5)
RE_UNTIMED_PAUSE = re.compile(r"\((\.{1,3})\)")               # (.) (..) (...)
RE_RETRACE = re.compile(r"\[(/{1,3})\]")                      # [/] [//] [///]
RE_REPLACEMENT = re.compile(r"\[: [^\]]+\]")
RE_ERROR = re.compile(r"\[\*[^\]]*\]")
RE_UNINTELLIGIBLE = re.compile(r"\bxxx\b")
RE_UNTRANSCRIBED = re.compile(r"\bwww\b")
RE_TERMINATORS = {
    "trailing_off": re.compile(r"\+\.\.\."),
    "self_interruption": re.compile(r"\+//[.?]"),
    "interrupted_by_other": re.compile(r"\+/[.?]"),
}
RE_OMITTED_PART = re.compile(r"\((\w+)\)(?=\w)|(?<=\w)\((\w+)\)")  # (be)cause
RE_ANGLE_GROUP = re.compile(r"<[^>]*>")
RE_HEADER_PARTICIPANTS = re.compile(r"^@Participants:\s*(.+)$")
RE_HEADER_ID = re.compile(r"^@ID:\s*(.+)$")
RE_HEADER_LANGUAGES = re.compile(r"^@Languages:\s*(.+)$")


@dataclass
class Utterance:
    index: int
    speaker: str                      # e.g. "PAR", "INV"
    raw: str                          # tier text incl. codes, bullets stripped
    dependent_tiers: dict[str, str] = field(default_factory=dict)
    start_ms: Optional[int] = None
    end_ms: Optional[int] = None

    # --- derived, code-level counts (input-side features, plan §5) ----------
    @property
    def repetitions(self) -> int:          # [/]
        return sum(1 for m in RE_RETRACE.finditer(self.raw) if m.group(1) == "/")

    @property
    def retracings(self) -> int:           # [//]
        return sum(1 for m in RE_RETRACE.finditer(self.raw) if m.group(1) == "//")

    @property
    def reformulations(self) -> int:       # [///]
        return sum(1 for m in RE_RETRACE.finditer(self.raw) if m.group(1) == "///")

    @property
    def filled_pauses(self) -> list[str]:
        return RE_FILLED_PAUSE.findall(self.raw)

    @property
    def fragments(self) -> int:
        return len(RE_FRAGMENT.findall(self.raw))

    @property
    def untimed_pauses(self) -> int:
        return len(RE_UNTIMED_PAUSE.findall(self.raw))

    @property
    def timed_pause_total_s(self) -> float:
        return sum(float(x) for x in RE_TIMED_PAUSE.findall(self.raw)
                   if not re.fullmatch(r"\.{1,3}", x))

    @property
    def unintelligible(self) -> int:
        return len(RE_UNINTELLIGIBLE.findall(self.raw))

    @property
    def terminator_flags(self) -> dict[str, bool]:
        return {k: bool(rx.search(self.raw)) for k, rx in RE_TERMINATORS.items()}

    @property
    def is_question(self) -> bool:
        return self.raw.rstrip().endswith("?") or "+//?" in self.raw or "+/?" in self.raw

    @property
    def clean_tokens(self) -> list[str]:
        """Spoken word tokens with CHAT codes stripped (for text features)."""
        t = self.raw
        t = RE_REPLACEMENT.sub("", t)
        t = RE_ERROR.sub("", t)
        t = RE_RETRACE.sub("", t)
        t = RE_FILLED_PAUSE.sub("", t)
        t = RE_FRAGMENT.sub("", t)
        t = RE_TIMED_PAUSE.sub("", t)
        t = RE_UNTIMED_PAUSE.sub("", t)
        t = RE_UNTRANSCRIBED.sub("", t)
        for rx in RE_TERMINATORS.values():
            t = rx.sub("", t)
        t = t.replace("<", "").replace(">", "")
        t = re.sub(r"\((\w+)\)", r"\1", t)          # (be)cause -> because
        t = re.sub(r"[^\w'@\s]", " ", t)
        return [w for w in t.split() if w and w != "xxx"]

    @property
    def n_tokens(self) -> int:
        return len(self.clean_tokens)

    @property
    def duration_s(self) -> Optional[float]:
        if self.start_ms is not None and self.end_ms is not None:
            return (self.end_ms - self.start_ms) / 1000.0
        return None

    @property
    def has_timestamp(self) -> bool:
        return self.start_ms is not None


@dataclass
class ChatDocument:
    path: str
    languages: list[str] = field(default_factory=list)
    participants: dict[str, str] = field(default_factory=dict)  # code -> role
    id_headers: list[str] = field(default_factory=list)
    utterances: list[Utterance] = field(default_factory=list)

    def by_speaker(self, code: str) -> list[Utterance]:
        return [u for u in self.utterances if u.speaker == code]


def _fold_continuations(lines: list[str]) -> list[str]:
    """CHAT continuation lines start with a tab; fold into previous line."""
    out: list[str] = []
    for line in lines:
        if line.startswith("\t") and out:
            out[-1] += " " + line.strip()
        else:
            out.append(line.rstrip("\n"))
    return out


def parse_cha_file(path: str | Path) -> ChatDocument:
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    doc = ChatDocument(path=str(path))
    lines = _fold_continuations(text.splitlines())

    current: Optional[Utterance] = None
    idx = 0
    for line in lines:
        if line.startswith("@"):
            if m := RE_HEADER_LANGUAGES.match(line):
                doc.languages = [s.strip() for s in re.split(r"[,\s]+", m.group(1)) if s.strip()]
            elif m := RE_HEADER_PARTICIPANTS.match(line):
                for part in m.group(1).split(","):
                    bits = part.strip().split()
                    if bits:
                        doc.participants[bits[0]] = bits[-1] if len(bits) > 1 else ""
            elif m := RE_HEADER_ID.match(line):
                doc.id_headers.append(m.group(1))
            continue
        if line.startswith("*"):
            head, _, body = line.partition(":")
            speaker = head[1:].strip()
            start = end = None
            if tm := RE_BULLET.search(body):
                start, end = int(tm.group(1)), int(tm.group(2))
            body = RE_BULLET.sub("", body).strip()
            current = Utterance(index=idx, speaker=speaker, raw=body,
                                start_ms=start, end_ms=end)
            doc.utterances.append(current)
            idx += 1
        elif line.startswith("%") and current is not None:
            head, _, body = line.partition(":")
            current.dependent_tiers[head[1:].strip()] = body.strip()
    return doc


def utterance_feature_row(doc: ChatDocument, u: Utterance,
                          corpus: str = "", language: str = "") -> dict:
    """Flatten one utterance into the unified per-utterance feature table row
    (plan §13 item 1)."""
    term = u.terminator_flags
    dur = u.duration_s
    return {
        "file": Path(doc.path).name,
        "corpus": corpus,
        "language": language or (doc.languages[0] if doc.languages else ""),
        "utt_index": u.index,
        "speaker": u.speaker,
        "n_tokens": u.n_tokens,
        "start_ms": u.start_ms,
        "end_ms": u.end_ms,
        "duration_s": dur,
        "speech_rate_tok_per_s": (u.n_tokens / dur) if dur and dur > 0 else None,
        "filled_pauses": len(u.filled_pauses),
        "fragments": u.fragments,
        "repetitions": u.repetitions,          # [/]
        "retracings": u.retracings,            # [//]
        "reformulations": u.reformulations,    # [///]
        "untimed_pauses": u.untimed_pauses,
        "timed_pause_total_s": u.timed_pause_total_s,
        "unintelligible": u.unintelligible,
        "trailing_off": term["trailing_off"],
        "self_interruption": term["self_interruption"],
        "interrupted_by_other": term["interrupted_by_other"],
        "is_question": u.is_question,
        "text": " ".join(u.clean_tokens),
    }
