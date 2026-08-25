# English pilot — run this week, no recruitment

Goal: produce the first gold labels + reliability numbers using yourself and one
colleague as the two raters. This alone yields P1's construct-validation section.

## What you need
- `annotation/annotator.html` (the tool — open in Chrome on your laptop).
- `annotation/manifests/eng.json` (150 English items, already sampled).
- Optional audio: the English `wav16k` subset (for the audio+transcript
  condition). If you don't want to move audio yet, start **transcript-only**.

## Protocol (keep it simple and consistent)

Two raters = **you (R1)** and **one colleague (R2)**. Each does the SAME 150
items. Steps:

1. **Calibrate first.** Both rate the same 20 items (the first 20 in the
   manifest), then compare and reconcile understanding of the 0–4 scale and the
   repair tags. Discard these 20 from IAA or keep as a separate calibration
   record.
2. **Rate independently.** No discussion during the main 130. Each rater:
   - Tier-2 difficulty 0–4 for the highlighted PAR turn (keys 0–4).
   - Tier-1 repair tags only if a repair actually occurs (leave blank otherwise).
3. **Two conditions.** Do the whole set once **transcript-only**, and (if audio
   is loaded) once **audio+transcript**. Set the Condition dropdown accordingly;
   it's recorded in the export. Transcript-only is the acoustically-independent
   tier — worth doing even if you skip audio.
4. **Export** after each pass → `ratings_R1_transcript-only.csv`, etc. Put all
   CSVs from both raters in `annotation/returned/`.

## Scale anchors (paste where raters can see them)

- 0 effortless · 1 mild (a filler/pause, no disruption) · 2 moderate (noticeable
  word-search, self-correction, slowing) · 3 severe (message delayed/partly lost,
  listener effort) · 4 breakdown (not communicated without repair).

## Repair tags (tick only if present)

OI-open ("huh?/what?"), OI-specific (targets the trouble), OI-cand ("you mean X?"),
RE-ASK (question re-issued), NON-UP (response doesn't address prior turn),
SISR (speaker self-repairs within the turn). Resolution +/− = did understanding
get restored.

## Then compute agreement

```bash
python scripts/ingest_annotations.py --inbox annotation/returned --out labels
```
Reads the gate: α≥0.7 scale up · 0.5–0.7 refine definitions and re-pilot · <0.5
revise the construct. Also prints the audio-vs-transcript correlation (the
independence check).

## After a healthy pilot
- Set `labels.target: gold_rating` in the config and re-run baselines/train on
  the gold English labels — first real (non-silver) numbers.
- Expand English to ~300–500 items for a proper gold test set.
- Then Stage 1/2 (other languages) per `GOLD_PILOT_STRATEGY.md`.

## Tips
- 150 items is ~60–90 min per pass. Do it in two sittings.
- If a turn's context is too short to judge repair, mark rating only, leave repair
  blank — that's fine and expected.
