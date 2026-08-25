"""
audio_match.py — Link transcripts to audio across two mirrored TalkBank trees.

The transcript tree and the media tree are EXACT mirrors apart from the leading
'transcripts/' and the file extension:
    transcripts/dementia/English/Pitt/Control/cookie/002-0.cha
       media/dementia/English/Pitt/Control/cookie/002-0.mp3
So the robust, collision-proof key is the RELATIVE PATH (below the population
dir) without extension — not (task, stem), which fails wherever the folder
above a file is a diagnosis (Greek AD/HC/MCI) rather than a task, and not
stem-only, which can match the WRONG corpus's identically-named file.

Primary match: full relative path. Secondary: stem, but only if that stem is
globally UNIQUE across the audio tree (so a fallback can never mis-assign).
"""
from __future__ import annotations

from pathlib import Path

AUDIO_EXTS = ("wav", "mp3", "mp4", "flac", "m4a")


def _relkey(path, root) -> str:
    """Path relative to its media/wav root, no extension, lowercased."""
    return str(Path(path).relative_to(root).with_suffix("")).lower()


def transcript_key(rel_path: str) -> str:
    """Key from a transcript rel_path (relative to the transcripts data_root):
    drop the leading population dir (dementia/fluency) + extension, lowercase.
    'dementia/English/Pitt/Control/cookie/002-0.cha' -> 'english/pitt/control/cookie/002-0'
    'fluency/Voices-AWS/interview/01.cha'            -> 'voices-aws/interview/01'
    """
    parts = Path(rel_path).parts
    sub = parts[1:] if len(parts) > 1 else parts
    return str(Path(*sub).with_suffix("")).lower() if sub else ""


def build_audio_map(roots: list) -> dict:
    """roots: media roots (dementia, fluency) OR the wav16k cache. Keys computed
    relative to each root are consistent because preprocess mirrors that exact
    relative path into the cache."""
    by_rel: dict[str, str] = {}
    by_stem: dict[str, str] = {}
    collide: set[str] = set()
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        for ext in AUDIO_EXTS:
            for p in root.rglob(f"*.{ext}"):
                by_rel.setdefault(_relkey(p, root), str(p))
                s = p.stem.lower()
                if s in by_stem and by_stem[s] != str(p):
                    collide.add(s)
                else:
                    by_stem.setdefault(s, str(p))
    for s in collide:                       # keep only globally-unique stems
        by_stem.pop(s, None)
    return {"by_rel": by_rel, "by_stem": by_stem}


def resolve_audio(rel_path: str, amap: dict) -> str | None:
    """Exact relative-path match first; unique-stem fallback second."""
    k = transcript_key(rel_path)
    if k in amap["by_rel"]:
        return amap["by_rel"][k]
    return amap["by_stem"].get(Path(rel_path).stem.lower())
