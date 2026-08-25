"""Deprecated location. The canonical implementation lives in
``vabench.audio_match``; this shim re-exports it so any stray import of
``vabench.features.audio_match`` gets the current, path-based matcher rather
than an out-of-date copy."""
from ..audio_match import (  # noqa: F401
    AUDIO_EXTS,
    build_audio_map,
    resolve_audio,
    transcript_key,
    _relkey,
)
