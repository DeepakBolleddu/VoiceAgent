# Data setup on the HPC (u8023272)

You have everything needed: transcripts **and** audio for DementiaBank (7
languages) and FluencyBank. No further downloads required.

## Your layout

```
~/media.talkbank.org/
  transcripts/
    dementia/{English,German,Greek,Korean,Mandarin,Spanish,Taiwanese}/...  (.cha)
    fluency/Voices-AWS/{interview,reading}/...                             (.cha)
  dementia/  ...  (.mp3  audio, mirrors the transcript tree)
  fluency/Voices-AWS/{interview,reading}/*.mp4  (video — audio extracted at preprocess)
  phon/      <- PhonBank, ~1 TB. NOT used. Excluded from every scan.
```

Transcripts and audio are separate trees, linked by **(task, filename-stem)** —
e.g. transcript `.../Control/cookie/002-0.cha` ↔ audio `.../Control/cookie/002-0.mp3`.
The `(task, stem)` key matters because the same stem repeats across task folders.

`configs/hpc.yaml` is already set to these paths:
- `data_root: ~/media.talkbank.org/transcripts`
- `audio.media_root: [~/media.talkbank.org/dementia, ~/media.talkbank.org/fluency]`
  (phon deliberately omitted)

## Preprocessing

1. **Audio → 16 kHz mono WAV** (`pbs/01_preprocess_audio.pbs`): converts both
   `.mp3` (dementia) and `.mp4` (fluency video) via ffmpeg, preserving the task
   folder, and prints `(task, session)` audio↔transcript coverage. Needs ffmpeg
   (`module load ffmpeg`, or it's in dl_env; falls back to librosa).
2. **Forced alignment — only if coverage flags it.** `pbs/00` prints per-corpus
   timestamp coverage; Pitt `.cha` carry media bullets so it should be high. Add
   Montreal Forced Aligner only for any corpus that comes back low.

No diarization needed — CHAT `*PAR:`/`*INV:` tiers already separate speakers.

## Two things to verify on the first real run

1. **FluencyBank speaker grouping.** The regex merges by leading number
   (`24fb`, `24fc` → speaker 24) — conservative, so no split leakage. If
   Voices-AWS metadata says a number denotes a session not a person (so
   `62f`/`62m` are truly different people), tell me and I'll switch to full-stem
   speaker keys. Over-merging is safe; over-splitting is not, so the default errs
   safe.
2. **Within-language multi-corpus.** Each `Dem_<Lang>` corpus points at the whole
   language folder. If a language contains several sub-corpora that reuse the same
   stem numbers for different people, check `speakers.csv`: an implausibly large
   `n_sessions` for one speaker signals an unwanted merge — then we split that
   language into per-sub-corpus entries. (Cross-*language* collisions are already
   handled: the speaker key embeds language.)

## Run order (PBS — matches your dl_env + modules)

```bash
cd ~/VoiceAgent/benchmark
qsub pbs/00_index_baselines.pbs     # CPU: index (7 langs + fluency), splits, coverage, baselines
qsub pbs/01_preprocess_audio.pbs    # CPU: mp3/mp4 -> 16k wav + coverage
qsub pbs/02_embed.pbs               # GPU: cache XLS-R embeddings (all layers)
qsub pbs/03_train_ablations.pbs     # GPU array 0-4: the 5 invariance cells
qsub pbs/04_zeroshot.pbs            # GPU: LOLO (7 languages) + LOPO transfer
```

With 7 languages you now get a real leave-one-language-out sweep (`pbs/04`) —
that is the RQ3 zero-shot result the paper is built around.

## First-run checklist

- [ ] `pip check` in dl_env: torch, transformers, soundfile, librosa, pandas,
      scikit-learn, scipy, pyyaml. Install any missing into dl_env.
- [ ] `qsub pbs/00` → open `artifacts/index/speakers.csv`: sanity-check speaker
      counts per language, the two `n_sessions` caveats above, and the printed
      timestamp coverage.
- [ ] `qsub pbs/01` → confirm audio-match % is high; investigate 0-audio pairs.
- [ ] Then GPU jobs 02 → 03 → 04.
```
