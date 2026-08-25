# Gold-annotation strategy under a rater constraint

Reality: native raters for Korean/Greek/etc. are hard to get right now. This
plan gets the project moving with (almost) no recruitment, then scales.

## The reframe that makes this tractable

You do **not** need large hand-labeled *training* sets. The estimator trains on
**silver + audio** (already built). Gold labels are needed for two things only:

1. **Construct validation** (reliability, state-vs-trait, known-groups) — do this
   thoroughly in ONE language.
2. **Gold test sets** for honest evaluation and cross-lingual transfer — these
   are *small* (a few hundred items per language), not full corpora.

So the ask shrinks from "annotate everything, everywhere" to "one language deeply
+ small test sets elsewhere." That is a fundamentally smaller recruitment problem.

## Two label types have very different rater requirements

- **Perceptual difficulty ratings (0–4)** — need a **native/fluent** listener
  (acoustic judgment). This is the recruitment-limited part.
- **Interactional-repair events** — largely **structural/transcript-based**
  (clarification requests, re-asks, non-uptake across turns). A fluent reader —
  or a bilingual collaborator, or you + a translation of the turns — can annotate
  these with far less native-acoustic dependence. Repair is also our *primary,
  independent* criterion (§B1 of the review), so leaning here is principled, not
  just convenient.

## Staged plan

### Stage 0 — English-first, zero recruitment (start now)
- You (fluent English) + one colleague annotate the **English pilot** (150 items,
  already sampled: `annotation/manifests/eng.json`).
- Both label Tier-1 repair + Tier-2 ratings, both conditions, sharing a 20-item
  calibration set first.
- Deliverables: first Krippendorff α, the audio-vs-transcript independence check,
  and the state-vs-trait demonstration — i.e. **the entire construct-validation
  section of P1**, in one language, with no external raters.
- If α is healthy, scale English to a full gold test set (~300–500 items).

### Stage 1 — cheap second language via repair-only
- Pick the highest-value repair-rich language you have *any* access to (Greek
  Dem@Care is multi-party = repair-dense; Korean interviews = repair-dense).
- Annotate **repair events only** (skip perceptual ratings) — doable by a
  bilingual collaborator or with per-turn translation. This already gives a
  cross-lingual criterion label for transfer evaluation.

### Stage 2 — small native test sets for transfer
- For 1–2 target languages, recruit **one** native rater for a **small test set**
  (~200 items), enough to *evaluate* zero-shot transfer (not train).
- Recruitment channels, easiest first: lab/department contacts and TalkBank
  corpus contributors (they know the data); university language/linguistics
  departments; Prolific/MTurk with native-language screening (costs money +
  ethics amendment).

### Stage 3 — full cross-lingual gold (later / if resourced)
- Only if the paper needs it and raters materialize. P1 can submit on Stage 0–2.

## Optional accelerant: model-in-the-loop (with caveats)

An LLM can pre-screen **repair-event candidates** (structural, objective) to cut
human time — but treat its output as a *second annotator to be verified*, never
as gold, and report LLM–human agreement separately. Do **not** pre-fill the
human's rating fields (anchoring inflates IAA). Only viable where you have an LLM
endpoint; our `repair_audit`/sampler heuristics already give a rule-based
first-pass for prioritization without any model.

## What this means for P1

P1's claims are reachable from **Stage 0 + Stage 1–2**:
- Construct + reliability + validity: English (Stage 0). ✓
- Cross-lingual generalization: silver probe consistency (done) + small gold test
  sets / repair-only labels in 1–2 languages (Stage 1–2).
- Full 8-language gold is a *future-work* line, not a submission blocker.

Honest limitation to state in the paper: primary construct validation in English,
with cross-lingual evidence from silver + targeted gold test sets — and name it.

## Immediate next actions

1. Do the English pilot yourself this week (see `annotation/ENGLISH_PILOT.md`).
2. Line up ONE colleague for the second English pass (IAA).
3. In parallel, email 2–3 TalkBank contributors / a Korean or Greek-speaking
   contact to gauge availability for a small test set — low-commitment ask.
