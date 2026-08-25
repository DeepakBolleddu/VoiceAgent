"""
ssl_embed.py — Frozen multilingual-SSL embedding cache (HPC, GPU).

For every participant utterance with timestamps, extract the audio segment,
run the SSL backbone, and cache per-layer pooled embeddings to
{out_root}/embeddings/{model_short}/layer{L}.npy (+ utt_ids.json).

Caching once makes the probing sweep (per-layer, plan §7.1) and all
train ablations cheap. Utterances WITHOUT timestamps are skipped and logged —
run forced alignment first if coverage is low (review §B5).

Requires: torch, transformers, soundfile, librosa (see requirements.txt).
Run via slurm/embed.sbatch.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--layers", default="all", help="'all' or comma list, e.g. 8,12,16")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).expanduser().read_text())

    import soundfile as sf
    import torch
    from transformers import AutoFeatureExtractor, AutoModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = cfg["ssl"]["model"]
    short = model_name.split("/")[-1]
    # Compute nodes have no internet -> load from cache / local dir only.
    offline = cfg["ssl"].get("offline", True)
    fe = AutoFeatureExtractor.from_pretrained(model_name, local_files_only=offline)
    model = AutoModel.from_pretrained(model_name, output_hidden_states=True,
                                      local_files_only=offline)
    model.eval().to(device)

    out_root = Path(cfg["out_root"]).expanduser()
    from ..tables import read_utterances
    from ..tiers import is_participant
    utts = read_utterances(out_root / "index" / "utterances.csv")
    utts = utts[utts["tier"].map(is_participant)]
    has_ts = utts["start_ms"].notna() & utts["end_ms"].notna()
    print(f"[embed] {has_ts.sum()}/{len(utts)} utterances have timestamps")
    utts = utts[has_ts]

    sr = cfg["audio"]["sample_rate"]
    max_s = cfg["audio"]["max_segment_s"]

    # Audio lives in a SEPARATE tree from transcripts (media.talkbank.org) and
    # is matched by (task, stem) because stems repeat across task folders.
    from ..audio_match import build_audio_map, resolve_audio
    # Prefer the 16k wav cache (mirrors full relative paths); else raw media.
    wav_cache = cfg["audio"].get("wav_cache", "")
    if wav_cache and Path(wav_cache).expanduser().exists():
        roots = [Path(wav_cache).expanduser()]
    else:
        mr = cfg["audio"].get("media_root", cfg["data_root"])
        media_roots = [mr] if isinstance(mr, str) else list(mr)
        roots = [Path(m).expanduser() for m in media_roots]
    amap = build_audio_map(roots)
    print(f"[embed] indexed {len(amap['by_rel'])} audio files under {roots}")

    ids, per_layer = [], None
    audio_cache: dict[str, np.ndarray] = {}
    missing = 0
    with torch.no_grad():
        for _, row in utts.iterrows():
            key = resolve_audio(str(row["rel_path"]), amap)
            if key is None:
                missing += 1
                continue
            if key not in audio_cache:
                audio_cache.clear()          # keep one file in memory
                y, file_sr = sf.read(key, dtype="float32", always_2d=False)
                if y.ndim > 1:
                    y = y.mean(axis=1)
                if file_sr != sr:
                    import librosa
                    y = librosa.resample(y, orig_sr=file_sr, target_sr=sr)
                audio_cache[key] = y
            y = audio_cache[key]
            s = int(row["start_ms"] / 1000 * sr)
            e = min(int(row["end_ms"] / 1000 * sr), s + int(max_s * sr))
            seg = y[s:e]
            if len(seg) < int(0.2 * sr):
                continue
            inputs = fe(seg, sampling_rate=sr, return_tensors="pt").to(device)
            hs = model(**inputs).hidden_states          # tuple(L+1) [1,T,D]
            if per_layer is None:
                layers = (range(len(hs)) if args.layers == "all"
                          else [int(x) for x in args.layers.split(",")])
                per_layer = {L: [] for L in layers}
            for L in per_layer:
                per_layer[L].append(hs[L].mean(dim=1).squeeze(0).cpu().numpy())
            ids.append(row["utt_id"])

    out = out_root / "embeddings" / short
    out.mkdir(parents=True, exist_ok=True)
    (out / "utt_ids.json").write_text(json.dumps(ids))
    for L, vecs in per_layer.items():
        np.save(out / f"layer{L}.npy", np.stack(vecs))
    print(f"[embed] cached {len(ids)} utterances x {len(per_layer)} layers -> {out}")
    print(f"[embed] {missing} utterances had no matching audio (check media_root)")


if __name__ == "__main__":
    main()
