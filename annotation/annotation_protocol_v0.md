# Annotation Protocol v0 — Interactional Repair Events + Perceptual Difficulty

Status: pilot draft (plan §13 item 3). Purpose: label the part CHAT cannot supply. Incorporates review corrections B1 (label independence) and B5 (granularity).

## 1. Units

Two units, annotated separately and later linked:

- **Repair events** — turn-level. An event spans a *sequence*: trouble source turn → initiation → repair → (optional) confirmation. Annotators mark the sequence span and its category.
- **Perceptual difficulty ratings** — per participant utterance (primary) and per 5-second window (secondary, only for utterances > 10 s). Utterance-primary avoids the alignment dependency for corpora without timestamps.

## 2. Tier 1 — Repair-event taxonomy (objective; target α ≥ 0.7)

Adapted from Schegloff/Jefferson/Sacks repair organization and aphasia conversation-coding practice. Categories, each with a decision anchor:

| Code | Category | Anchor question |
|---|---|---|
| SISR | Self-initiated self-repair | Speaker interrupts own talk and fixes it within the same turn (maps to `[//]`, `[///]`, `+//.`) — flag but do NOT count as breakdown |
| OI-open | Other-initiation, open class | Next speaker signals trouble without locating it ("huh?", "what?", "pardon?") |
| OI-specific | Other-initiation, specific | Next speaker targets the trouble ("the WHAT?", partial repeat + question, wh-question on prior turn) |
| OI-cand | Other-initiation, candidate understanding | Next speaker offers a check ("you mean X?", "the water is running over?") |
| RE-ASK | Question re-issue | Speaker repeats/rephrases own prior question after inadequate or no uptake |
| NON-UP | Non-uptake / mismatch | Response does not address the prior turn (topic-irrelevant answer, silence where response due) |
| RES+ / RES− | Resolution | Did the sequence restore mutual understanding? (+ resolved, − abandoned/failed) |

Rules: annotate from transcript + audio; the trouble-source turn gets the event anchor; nested sequences allowed; SISR is production-side and is captured automatically by CHAT codes — annotators only verify, never invent it.

## 3. Tier 2 — Perceptual difficulty rating (graded; the continuous target)

Scale: 0–4 per participant utterance.
0 = fluent, effortless · 1 = mild effort (occasional filler/pause, no disruption) · 2 = moderate (noticeable word search, self-correction, slowed delivery) · 3 = severe (message delayed or partly lost; listener effort required) · 4 = breakdown (message not communicated without repair).

**Independence design (review B1):** two rater conditions per language —
(a) **audio+transcript** raters (the deployment-realistic rating), and
(b) **transcript-only** raters (acoustically independent).
Report agreement within and between conditions. If (a) and (b) diverge sharply, the acoustic channel is carrying rating variance and rating-based labels are demoted to convergent evidence; repair events (Tier 1) remain the primary criterion regardless.

Raters: native or near-native speakers per language; 2 raters minimum per item in the pilot, 3 for the gold set; brief calibration set of 20 shared items before production annotation.

## 4. Pilot design (run the moment AphasiaBank/TBIBank land)

1. Sample 10 transcripts per corpus (stratified by severity where available), pre-screened by `repair_audit.py` to include high- and low-repair-density files.
2. Two annotators, full double annotation of Tier 1 and Tier 2.
3. Compute Krippendorff's α (nominal for Tier 1 categories, ordinal for Tier 2; also span-level gamma/agreement for event boundaries).
4. Decision gates: α ≥ 0.7 → scale up; 0.5–0.7 → refine definitions, re-pilot; < 0.5 → construct revision before any modeling (plan §11 risk "construct not reliably annotatable").
5. Log every disagreement with a one-line cause; feed into taxonomy revision.

## 5. Outputs

`labels/{corpus}/{file}.events.csv` (utt_index spans, category, resolution) and `labels/{corpus}/{file}.ratings.csv` (utt_index, rater, condition, score). Keyed to TalkBank file + utterance indices from `chat_parser.py` — this keying is what makes the annotation layer releasable without redistributing audio (review B4).
