"""
preprocess_audio.py — Convert the TalkBank media tree (.mp3 dementia, .mp4
fluency) to a flat 16 kHz mono WAV cache, robustly and in parallel.

Design notes from the first HPC run:
  * No system ffmpeg / no audioread backend -> we bundle a static ffmpeg via
    `imageio-ffmpeg` (pip install imageio-ffmpeg). Falls back to system ffmpeg
    if present.
  * Some source mp3s are slightly corrupt -> ffmpeg runs with error tolerance
    and per-file try/except; one bad file never kills the job. Failures are
    logged to {wav_cache}/_failed.txt.
  * 6k+ files -> multiprocessing across cores with progress.

Usage:
  python scripts/preprocess_audio.py --config configs/hpc.yaml [--workers 8] [--limit N]
"""
import argparse
import os
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import yaml


def find_ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        sys.exit("No ffmpeg available. Run: pip install --break-system-packages "
                 "imageio-ffmpeg  (bundles a static ffmpeg binary).")


def convert_one(args) -> tuple[str, bool, str]:
    src, dst, sr, ffmpeg = args
    dst = Path(dst)
    if dst.exists():
        return (src, True, "exists")
    dst.parent.mkdir(parents=True, exist_ok=True)
    # -err_detect ignore_err + -fflags +discardcorrupt tolerate broken frames.
    cmd = [ffmpeg, "-y", "-v", "error", "-err_detect", "ignore_err",
           "-fflags", "+discardcorrupt", "-i", str(src),
           "-ac", "1", "-ar", str(sr), str(dst)]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=300)
        if r.returncode == 0 and dst.exists() and dst.stat().st_size > 44:
            return (src, True, "ok")
        return (src, False, r.stderr.decode("utf-8", "ignore")[:200])
    except Exception as e:
        return (src, False, str(e)[:200])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/hpc.yaml")
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 4))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).expanduser().read_text())

    mr = cfg["audio"]["media_root"]
    media_roots = [Path(m).expanduser() for m in ([mr] if isinstance(mr, str) else mr)]
    wav_cache = Path(cfg["audio"]["wav_cache"]).expanduser()
    wav_cache.mkdir(parents=True, exist_ok=True)
    sr = cfg["audio"]["sample_rate"]
    ffmpeg = find_ffmpeg()
    print(f"[pre] ffmpeg: {ffmpeg}")

    media = []                              # (src, its media_root)
    for root in media_roots:
        for ext in ("mp3", "mp4", "wav", "m4a", "flac"):
            media += [(p, root) for p in sorted(root.rglob(f"*.{ext}"))]
    if args.limit:
        media = media[:args.limit]
    print(f"[pre] {len(media)} media files under {media_roots}; {args.workers} workers")

    # mirror the FULL relative path (below the media root) so audio<->transcript
    # matching is by exact relative path (collision-proof; see audio_match).
    jobs = []
    for src, root in media:
        rel = src.relative_to(root).with_suffix(".wav")
        jobs.append((str(src), str(wav_cache / rel), sr, ffmpeg))

    ok = fail = 0
    failed = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(convert_one, j) for j in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            src, good, msg = fut.result()
            if good:
                ok += 1
            else:
                fail += 1; failed.append(f"{src}\t{msg}")
            if i % 500 == 0 or i == len(jobs):
                print(f"[pre] {i}/{len(jobs)}  ok={ok} fail={fail}", flush=True)

    if failed:
        (wav_cache / "_failed.txt").write_text("\n".join(failed))
        print(f"[pre] {fail} files failed -> {wav_cache/'_failed.txt'} "
              f"(usually corrupt source audio; safe to inspect later)")

    # coverage vs transcript index, keyed on exact relative path
    idx = Path(cfg["out_root"]).expanduser() / "index" / "utterances.csv"
    if idx.exists():
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from vabench.audio_match import _relkey, transcript_key
        utts = pd.read_csv(idx, dtype=str, keep_default_na=False, low_memory=False)
        want = {transcript_key(r) for r in utts["rel_path"].unique()}
        have = {_relkey(p, wav_cache) for p in wav_cache.rglob("*.wav")}
        matched = want & have
        print(f"[pre] transcript sessions={len(want)} audio-matched={len(matched)} "
              f"({100 * len(matched) / max(1, len(want)):.1f}%)")
        miss = sorted(want - have)[:5]
        if miss:
            print(f"[pre] example unmatched transcript keys: {miss}")
    else:
        print("[pre] (run scripts/build_index.py first to check coverage)")
    print(f"[pre] done: converted/exists={ok} failed={fail} -> {wav_cache}")


if __name__ == "__main__":
    main()
