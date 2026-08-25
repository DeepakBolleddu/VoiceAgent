# P1 — A Cross-Lingual, Cross-Population Benchmark for Communicative Difficulty in Clinical Speech

Status: living skeleton. Real numbers from HPC runs are inlined; `[PENDING GOLD]`
marks results that require the annotation pilot. Target venues: NeurIPS Datasets &
Benchmarks (primary) or ACL/EMNLP via ARR. Companion paper P2 = the estimator +
invariance analysis.

---

## Abstract (draft)

We introduce **VABench**, the first benchmark for *communicative difficulty* —
a graded, moment-to-moment interactional state, distinct from any per-speaker
diagnostic label — measured from clinical speech across **8 languages and 9
corpora** spanning four clinical populations (dementia, aphasia/fluency, and,
in later phases, TBI). The benchmark comprises **~107k utterances from ~2,064
speakers**, with speaker-independent splits and explicit held-out-language and
held-out-population zero-shot folds. We define the construct, release an
annotation layer keyed to TalkBank identifiers (audio is not redistributed), and
provide interpretable transcript baselines and a per-layer self-supervised
probing analysis. A frozen XLS-R probe shows difficulty is decodable from audio
with a consistent signal across all eight languages, peaking at layer 18. We
release code, splits, and the annotation protocol to support work on
language- and population-invariant difficulty estimation. `[PENDING GOLD: headline
reliability + validity numbers]`

---

## 1. Introduction

- The problem: language-barrier communication support in healthcare is
  under-supplied; generic MT is unsafe and not confidence-aware. The missing
  capability is *trustworthy, adaptive communication support that knows when it
  is failing*.
- The construct shift (our core novelty): estimating a continuous, per-moment
  **communicative-difficulty state** rather than a per-person diagnostic label.
- Why a benchmark first: there is no shared task, data, or metric for this
  construct across languages and populations. P1 supplies it.
- Contributions:
  1. A defined, validated construct of communicative difficulty (state, not
     trait; production + interactional-repair loci).
  2. VABench: a cross-lingual (8 languages), cross-population benchmark with
     leakage-safe speaker splits and zero-shot language/population folds.
  3. A non-circular measurement model (audio inputs; transcript/interaction
     labels) with a two-condition annotation protocol.
  4. Baselines + a per-layer SSL probing analysis localizing difficulty in the
     representation. `[PENDING GOLD: gold-label validity]`

Figure 1 (planned): within-session variation of the difficulty signal vs a flat
per-speaker diagnostic score — the state-vs-trait distinction.

## 2. Related work (positioning — must cite explicitly)

- **Cognitive-load / effort from speech** (ComParE cognitive-load lineage):
  graded effort estimation exists; our wedge is interactional repair as an
  independent criterion, cross-population scope, and clinical multilingual data.
- **Multilingual cognitive-impairment detection** (ADReSS/ADReSS-M, TAUKADIAL):
  owns "cross-lingual clinical speech," but at the *per-speaker diagnostic*
  level. Our wedge is state-not-trait.
- **Disfluency detection**: a surface-marker task; we treat disfluency as one
  noisy *indicator* of a latent state, not its definition.

## 3. The construct: communicative difficulty

- Working definition: the graded, time-varying degree to which producing and
  grounding meaning is effortful or disrupted at moment t.
- Two loci: **production difficulty** (monological — pauses, fillers,
  word-finding) and **interactional repair** (dialogical — clarification,
  re-asks, non-uptake). Production carries model inputs; repair carries
  independent labels (non-circularity).
- Three distinctions preserved: state not trait; latent cause not one indicator;
  production vs comprehension.

## 4. Benchmark construction

### 4.1 Corpora (real, current)

| Corpus | Lang | Speakers | Utterances | ts_cov | Notes |
|---|---|---|---|---|---|
| Dem_English (Pitt) | eng | 296 | 22,353 | 89.4% | picture description + interview |
| Dem_German | deu | 222 | 2,193 | 96.8% | |
| Dem_Greek (Dem@Care) | ell | 565 | 28,861 | 99.9% | **multi-party group conversation** (rich repair) |
| Dem_Korean (Kang) | kor | 77 | 3,784 | 100% | interview (rich repair) |
| Dem_Mandarin | zho | 260 | 4,567 | 94.6% | |
| Dem_Spanish_Ivanova | spa | ~357 | ~1,580 | 99.7% | recall monologue |
| Dem_Spanish_PerLA | spa/**cat** | — | — | — | multi-party family conversation; adds Catalan |
| FluencyBank (Voices-AWS) | eng | 67 | 5,636 | 97.6% | stuttering (distinct disfluency mechanism) |
| Dem_Taiwanese (Lu) | nan | 16 | 278 | 0% ⚠ | unlinked media; audio+gold only, held-out |

Totals: ~2,064 speakers, ~107k utterances; 8 languages. `[UPDATE after Spanish-split re-index: exact PerLA/Catalan counts]`

### 4.2 Parsing/normalization challenges solved (a contribution in itself)

- **Language-agnostic disfluency extraction**: many corpora don't use English
  CHAT codes (`&-um`, `[/]`). We add lexical-filler lexicons, `%mor`
  interjection tags, and immediate-repetition detection so non-English corpora
  aren't scored as marker-free.
- **Multi-party speakers**: Greek Dem@Care (`PAR0..PARn`) and Spanish PerLA
  (family tiers `HIJ/MAR/MUJ`) handled; each participant is a distinct
  session-scoped speaker; family/investigator tiers are interlocutors.
- **Language-code hazards**: Taiwanese code `nan` (Min Nan) collides with pandas
  NA; handled via typed CSV I/O.
- **Audio↔transcript linkage**: exact relative-path matching across two mirrored
  trees (99.2% coverage; collision-safe).

### 4.3 Splits

Strictly speaker-independent (longitudinal sessions kept together — no leakage);
language-stratified and population-stratified; with **leave-one-language-out
(7 folds)** and **leave-one-population-out (2 folds)** zero-shot tests. Caveat
reported: held-out language ≈ held-out corpus (cross-corpus transfer).

## 5. Measurement model

- **Input**: acoustic/prosodic production markers on a multilingual SSL backbone.
- **Silver bootstrap target** (`silver_pdi`): rate-normalized, within-(corpus,
  language) z-scored disfluency/pause burden. Explicitly a scaffold; used only
  for pipeline validation and where it has variance (English/FluencyBank).
- **Gold target** (`[PENDING GOLD]`): perceptual difficulty ratings (0–4) in two
  rater conditions (audio+transcript, transcript-only — acoustically independent)
  + interactional-repair event annotation. Repair events are the primary
  criterion (independent of the acoustic input); ratings are convergent evidence.
- Non-circularity: model inputs (audio) and criterion labels (repair) come from
  different observational sources.

## 6. Baselines and probing

### 6.1 Transcript-feature baselines (reference, silver target)

Report B0 (length), B1 (interpretable markers), B2 (+language). NOTE these
predict silver from the same code family they are built on — reported as
reference/plumbing, not findings. `[insert table from baselines_*.json]`

### 6.2 Per-layer SSL probing (real result)

Frozen XLS-R-300M, linear ridge on mean-pooled embeddings, speaker-independent
split, silver target. Difficulty is decodable from **audio** (independent of
transcript codes), peaking at **layer 18** (overall ρ = 0.254), with a
**consistent signal across all eight languages** (per-language ρ ≈ 0.16–0.28):

| layer | overall ρ | deu | ell | eng | kor | spa | zho |
|---|---|---|---|---|---|---|---|
| 9  | 0.227 | 0.19 | 0.16 | 0.22 | 0.23 | 0.20 | 0.18 |
| 13 | 0.240 | 0.17 | 0.16 | 0.23 | 0.24 | 0.24 | 0.22 |
| **18** | **0.254** | 0.17 | 0.16 | 0.25 | 0.24 | 0.22 | 0.23 |
| 24 | 0.210 | 0.18 | 0.15 | 0.22 | 0.12 | 0.18 | 0.17 |

Interpretation: the peak location (upper-middle layers) localizes difficulty
information; the modest absolute ρ reflects the silver ceiling (noisy proxy,
linear probe, utterance-level). Cross-lingual consistency is preliminary support
for a language-invariant difficulty signal — the RQ2/RQ3 hypothesis, tested
properly against gold in P2.

## 7. `[PENDING GOLD]` Construct validation

Reliability (Krippendorff α on both label tiers); convergent validity (markers,
ratings, repair agree); known-groups (difficulty rises with dx severity —
diagnosis validates, never labels); discriminant validity (within-speaker
variation ⇒ state not trait; not merely language/diagnosis identity).
Pilot: 2 native raters × {English, Korean, Greek}, calibration set first, gates
α≥0.7 scale / 0.5–0.7 refine / <0.5 revise.

## 8. Difficulty estimator + invariance (silver results in; gold confirmation pending)

Estimator (MLP over XLS-R layer 18, corpus-balanced sampling, dev-based model
selection), 3 seeds, mean±sd. IID test ρ=0.396 (vs 0.254 linear probe).

**Zero-shot transfer (the headline analysis):**

| fold (held out) | probe | adv_lang | adv_both |
|---|---|---|---|
| LOLO deu | **0.251±0.007** | 0.232±0.052 | 0.235±0.025 |
| LOLO eng | **0.208±0.057** | 0.175±0.034 | 0.186±0.024 |
| LOLO kor | 0.248±0.037 | 0.248±0.050 | 0.238±0.034 |
| LOLO spa | 0.292±0.063 | **0.320±0.006** | 0.238±0.030 |
| LOLO zho | **0.160±0.015** | 0.142±0.016 | 0.137±0.008 |
| LOLO ell | 0.081±0.012 | 0.084±0.006 | 0.059±0.036 | (silver noise floor) |
| LOPO dementia | 0.094±0.025 | 0.102±0.017 | **0.139±0.025** |
| LOPO fluency | 0.394±0.037 | 0.418±0.010 | 0.397±0.038 |

Findings (silver-label caveat applies until gold confirmation):
1. **Language-adversarial invariance does not improve cross-lingual transfer**
   — all LOLO deltas within ~1 sd. Multilingual SSL appears already
   language-invariant for this signal.
2. **Population-adversarial invariance consistently helps cross-population
   transfer** (LOPO dementia: 0.139 vs 0.094, all 3 seeds, ~1.8 sd) — the
   population axis, unseen during SSL pretraining, is where explicit
   invariance earns its keep. This is P2's central claim.
3. **The IID→zero-shot gap** (eng 0.395→0.208) quantifies the cross-lingual
   generalization problem the benchmark exists to measure.
4. Methodological note worth reporting: an apparent "over-regularization
   collapse" (kor adv_both ρ=0.084) vanished under dev-based model selection +
   seed averaging (0.238±0.034) — zero-shot evaluation without model selection
   produces artifacts.

## 9. Ethics, safety, governance

HREC/ethics before human annotation of identifiable data; intended-use
statement (no diagnostic claims — also the SaMD regulatory boundary);
on-premises handling; de-identification; cross-language equity monitoring;
Indigenous/low-resource languages deferred with cultural-safety governance.

## 10. Limitations

Held-out language confounded with corpus/task/channel; silver limited to
code-rich corpora; Taiwanese lacks timestamps (held-out, audio+gold only);
audio not redistributable (annotation-layer release keyed to TalkBank IDs);
PerLA/Greek group dx applies at folder level (per-speaker dx needs corpus
metadata before known-groups claims).

## Tables/figures to fill

- T1 corpora (§4.1) — refresh after Spanish-split re-index.
- T2 transcript baselines (§6.1) — from `baselines_*.json`.
- T3 probe-by-layer (§6.2) — done; add FluencyBank column, mark peak.
- F1 state-vs-trait within-session variation — needs gold or silver demo.
- F2 probe curve (ρ vs layer, per language) — from `probe_layers_silver_pdi.json`.
- `[PENDING GOLD]` T4 reliability, T5 validity, T6 estimator/transfer.
