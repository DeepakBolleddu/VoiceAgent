# Cross-Lingual, Cross-Population Communicative-Difficulty Sensing for Adaptive Clinical Communication

**One-line summary.** A cross-lingual, cross-population speech model that estimates *moment-to-moment communicative difficulty* — a graded interactional state, not a diagnostic label — coupled to an offline-validated adaptation policy, and demonstrated inside a multilingual clinical-communication voice agent.

*Status: design phase, revision 2. This document is a living plan, not a fixed specification.*
*Revision 2 changes: v1 construct locus decided (production difficulty + interactional repair); data-access sequencing and its consequences added; CHAT-transcript bootstrapping incorporated into the measurement model; FluencyBank mechanism caveat added; immediate next step replaced with a concrete, currently-unblocked work plan.*

---

## 1. Motivation and significance

Australia is highly multilingual, yet language-barrier support in healthcare is under-supplied. Roughly three in ten Australians were born overseas and a meaningful share self-report limited English, but professional interpreters are used in only a fraction of the encounters that need them — one Sydney study found interpreters were required in about one in six admissions but provided in well under half of those cases, and the gap is widest exactly where need is highest (e.g. Aboriginal patients at northern hospitals). After English, the most common home languages include Mandarin, Arabic, Cantonese, Vietnamese, Italian, Greek, Hindi, Spanish, Punjabi, and Tagalog.

Existing lightweight tools (fixed medical phrasebook apps) cannot handle open conversation, cannot adapt, and are not confidence-aware. Generic machine translation is unsafe for clinical use: it produces fluent-sounding fabricated content, mistranslates critical terms, and has never been validated for patient safety in routine care. The unmet need is therefore not "translation" per se — it is *trustworthy, adaptive* communication support that knows when it is failing.

## 2. What is new (and what is deliberately not claimed)

The scientific contribution is **not** "we built a medical voice agent" (many exist; that is engineering) and **not** "cross-lingual dementia/MCI detection from speech" (an active, competitive area already). The novelty is the intersection that remains open:

1. **A construct shift** — estimating a continuous, per-moment *communicative-difficulty* state rather than a per-person diagnostic label.
2. **Cross-population unification** — one difficulty representation learned across multiple clinical populations (dementia, aphasia, fluency, TBI) rather than a single condition.
3. **Cross-lingual invariance** — a difficulty signal that transfers across languages, distinct from language identity.
4. **Sensing → adaptation coupling** — mapping that signal to communication adaptations (pace, simplification, teach-back, escalation) in a clinical language-barrier setting, evaluated offline.

**Explicitly out of scope / deferred:** prospective evidence of patient-outcome improvement (requires a future user study); Indigenous and low-resource languages (future phase, with cultural-safety governance); quantum ML (offers no benefit to the actual bottlenecks and would undermine the trust story).

## 3. Research questions

- **RQ1 (construct).** Can "moment-to-moment communicative difficulty" be defined and measured reliably and validly from speech, separately from speaker traits such as diagnosis?
- **RQ2 (representation).** Can a speech representation be learned that is simultaneously language-invariant and population-invariant, yet remains sensitive to difficulty — given that different populations produce surface markers (e.g. disfluency) for *different underlying reasons*?
- **RQ3 (transfer).** Does the difficulty estimator generalise to unseen languages and unseen populations (zero-shot)?
- **RQ4 (coupling).** Does an adaptation policy driven by the signal act appropriately — firing the right adaptation at genuinely difficult moments — under offline/simulation evaluation?

## 4. The core construct: communicative difficulty

Communicative difficulty is treated as a **latent state**, inferred from imperfect indicators, not read directly from audio. It is grounded in two established theories: *processing effort* (psycholinguistics — pauses, filled pauses, slowed rate, false starts as fingerprints of cognitive-linguistic load) and *grounding and repair* (Clark's grounding; Conversation Analysis — clarification requests, reformulations, non-uptake as the cost of restoring mutual understanding).

**Working definition.** *Communicative difficulty at moment t is the graded, time-varying degree to which producing and grounding meaning is effortful or disrupted — a state, distinct from stable speaker traits.*

**v1 locus (decided): production difficulty + interactional repair.** The construct spans two loci:

- **Production difficulty** — *monological*: one speaker struggling to produce their own words (pauses, fillers, word-finding stalls, self-correction). Visible from a single speaker's audio. Answers: *is this person finding it effortful to speak?*
- **Interactional repair** — *dialogical*: a breakdown in mutual understanding and its fix across turns (clarification requests, re-asks, reformulations, non-uptake, mismatched responses). Only exists between parties. Answers: *is understanding actually happening?*

Both are included because repair is what the *agent* most needs to sense (comprehension, not just fluency), and — critically — because repair provides a label source independent of the acoustic inputs, which is what makes the measurement model non-circular (Section 5). Production carries the model **inputs**; repair carries the **labels**.

Three distinctions the construct must preserve: **state, not trait** (it varies within a conversation; diagnosis does not — this separates the work from the detection literature); **the latent cause, not one indicator** (disfluency is a *symptom* of difficulty, not its definition); and the **production vs. comprehension** loci above.

## 5. The measurement model (how labels measure the construct)

The single most important design rule: **model inputs and training labels must come from different observational sources**, otherwise the model predicts a quantity from itself (circularity) and the results are meaningless.

- **Model input** — acoustic/prosodic production-effort markers (pauses, rate, filled pauses, voice quality) on top of a multilingual self-supervised backbone.
- **Training target (independent source)** — human perceptual difficulty ratings plus interactional-repair events annotated from the transcript and surrounding turns.

**CHAT transcripts bootstrap the input side.** TalkBank `.cha` files already encode much of the production signal, which removes a large amount of manual work: speaker tiers (`*PAR:` vs `*INV:`) separate participant from investigator without diarization; retracing codes mark disfluency type (`[/]` repetition, `[//]` reformulation, `[///]` complex retracing); filled pauses (`&-uh`, `&-um`), timed pauses (`(.)`, `(..)`, `(...)`), trailing-off (`+...`) and self-interruption (`+//.`) are inline; and `%mor`/`%gra` tiers provide morphology and grammar. These feed the **input** features.

**Annotation concentrates on what CHAT cannot give cleanly: repair as an event.** Clarification requests, other-initiated repair, and breakdown-and-fix across turns are not single codes — they are inferred from turn structure and must be annotated. Design: unit of short fixed windows (~3–5 s) or per-utterance; two tiers — (i) objective repair-*event* annotation (high inter-annotator agreement, defensible anchors, adapting existing Conversation-Analysis repair taxonomies and aphasia conversation-coding schemes) and (ii) graded perceptual difficulty ratings (the continuous target); native-speaker annotators per language; cross-lingual consistency of the construct treated as an empirical question to report. This division — CHAT for inputs, annotation for labels — is exactly what preserves the input/label source separation.

Annotating existing recordings is fully within the "existing datasets only" constraint; it adds a labelled layer to audio that already exists and can become a released resource.

## 6. Data

**Access status and sequencing.**
- **In hand now:** DementiaBank (7 languages: English, German, Greek, Korean, Mandarin, Spanish, Taiwanese) and FluencyBank.
- **Arriving in ~1–2 weeks:** AphasiaBank and TBIBank (access requested; readily granted to researchers).

This timeline maps cleanly onto the two loci. DementiaBank's dominant task (picture description) is *prompt-then-monologue*, so it is rich in **production difficulty** but likely **sparse in genuine interactional repair** — the density of real clarification-and-fix sequences must be measured, not assumed. AphasiaBank and TBIBank contain much richer free-dialogue/conversation protocols, which is where the **interactional-repair** signal will actually live. Therefore: build and validate the production side now with data in hand; the repair side gets its real fuel when Aphasia/TBI arrive.

**FluencyBank caveat (a research issue, not preprocessing).** FluencyBank centres on *stuttering*, whose disfluencies (repetitions, prolongations, blocks) arise from a different mechanism than the word-finding disfluencies of dementia (pauses, fillers, empty speech). It is useful for building the acoustic disfluency-*detector*, but must not be naively pooled into one "difficulty" bucket. That different populations produce disfluency for different reasons *is* RQ2's hard core.

**Not required:** PhonBank (child phonology, ~1 TB), CHILDES, ASDBank, ClassBank, HomeBank, MotorSpeechBank, BilingBank/SLABank (L2 disfluency is a confound), and others. CABank (open-access conversational speech) is available for interaction-context testing.

**Splits (decide before training).** Strictly speaker-independent; language-stratified and population-stratified; with held-out-language and held-out-population zero-shot tests. Split design precedes any modelling. A fully human-annotated gold set in ~2–3 languages; weak/transfer labels for the remainder, clearly marked.

## 7. Methodology

### 7.1 Sensing — the difficulty representation (core novelty)
A multilingual SSL backbone (e.g. XLS-R / mHuBERT-class) with a training objective that suppresses language and population identity (e.g. adversarial or disentanglement approaches) while preserving difficulty sensitivity. The central, must-articulate tension: strip too much and the signal is lost; strip too little and the model merely re-detects language or diagnosis. A frozen backbone plus a classifier is only a baseline, not the contribution. Per-layer probing and interpretable acoustic features (e.g. eGeMAPS) as complements are worthwhile.

### 7.2 Adaptation policy
A mapping from the difficulty signal to concrete actions: slow pace, simplify wording, rephrase, trigger a teach-back comprehension check, or escalate to a human interpreter. Rule-based thresholds first; optionally learned later. Evaluated **offline / in simulation** by replaying real recorded utterances and testing whether the signal fires at genuinely difficult moments and whether the triggered action is appropriate.

### 7.3 The agent (demonstrator, not the contribution)
A **cascaded** pipeline (ASR → MT → TTS), chosen over end-to-end because the exposed intermediate text is required for logging, back-translation, glossary control and auditing. Turn-based, push-to-talk, on-premises. Framed as an assistive aid that defers to a human interpreter — never a replacement. Indicative components: Whisper-class multilingual ASR; MT constrained by a pinned medical glossary; a streaming multilingual TTS; an end-to-end model (e.g. SeamlessM4T-class) as a comparison baseline. Trust layer: turn-level confidence gating, back-translation verification, explicit critical-information readback (drugs/doses/allergies/numbers), escalation logic, and a full audit transcript. Hard boundary: the agent translates and checks understanding only — it never generates medical content.

## 8. Evaluation

Construct and estimator (psychometric playbook): **reliability** (inter-annotator agreement, e.g. Krippendorff's α, on both label tiers); **convergent validity** (acoustic markers, repair events and ratings agree); **criterion/known-groups validity** (difficulty higher on harder tasks and in more-impaired speakers — diagnosis used to *validate*, not label); **discriminant validity** (signal varies *within* a speaker across a session — evidence of a state, not a trait; and is not merely language identity or diagnosis); **transfer** (zero-shot on held-out languages and populations, reported per-language and per-population, never pooled); **calibration** (per-language); **robustness** (accented, noisy, impaired speech).

Baselines and ablations: monolingual, single-population, frozen-SSL, and repurposed cross-lingual detection methods; ablate each invariance component. For the agent: translation quality (COMET/BLEU) plus critical-term preservation, back-translation agreement, and appropriate-escalation precision/recall — since benchmark scores alone hide real-world failure.

## 9. Safety, ethics and governance

Ethics/HREC approval before any human-subject work or new annotation touching identifiable data; a written intended-use-and-limitations statement; on-premises data handling (an advantage over cloud tools and a selling point for hospital partners); de-identification and access control for sensitive audio; and cross-language equity monitoring. Human oversight is always available. No diagnostic or clinical claims are made by the agent.

## 10. Scope and phasing

- **Phase 1 (now, existing data).** Define and validate the construct and annotation scheme; build and evaluate the cross-lingual, cross-population estimator; offline-validate the adaptation policy; stand up a bounded, turn-based agent demonstrator with the trust layer for 2–3 high-resource languages and one clinical scenario.
- **Phase 2+.** Prospective user study for deployed efficacy; Indigenous and low-resource languages with cultural-safety governance; learned adaptation policy; full-duplex interaction; broader clinical scenarios.

## 11. Risks and mitigations

- **Reads as an application, not a contribution** → lead with the construct/benchmark and the invariance method; release an artifact.
- **Circular labels** → strict input/label source separation: production markers as input, repair events and ratings as labels (Section 5).
- **Interactional-repair signal too sparse** → audit repair density empirically before committing; DementiaBank alone may be thin, so lean on AphasiaBank/TBIBank for repair; fall back to production-primary with repair as a secondary signal if the data cannot support it.
- **Construct not reliably annotatable** → pilot annotation early; low agreement is a signal to refine definitions before scaling.
- **Cross-population confound (FluencyBank mechanism)** → treat differing disfluency mechanisms as RQ2, not as pooled preprocessing.
- **Over-scoping** → A* papers are deep in one or two things; do not attempt benchmark + method + full agent + user study in one paper.
- **Cross-language artefacts** → report calibration alongside per-language results.
- **Data access delay** → sequence work so the first two weeks are fully unblocked by data in hand (Section 13).

## 12. Deliverables and target venues

- A defined, validated, released **multilingual cross-population communicative-difficulty benchmark** (resource contribution).
- The **difficulty-estimation model** with cross-lingual/cross-population evaluation (method contribution).
- The **agent demonstrator** with offline adaptation evaluation and a trust/audit layer (systems contribution).

Venue lanes (one primary emphasis per paper): an NLP/benchmark framing (ACL/EMNLP-style) around the task, benchmark and method; or an interaction framing (CHI-style) around the sensing→adaptation loop and its evaluation. The benchmark-plus-method lane has the cleaner path to a top-tier outcome; the construct-and-resource work may be the first paper, with the model as the second.

## 13. Immediate work plan (next 1–2 weeks — fully unblocked by data in hand)

None of the following requires AphasiaBank or TBIBank, so no time is lost while access is pending:

1. **CHAT parsing pipeline** on DementiaBank + FluencyBank: extract turns, timestamps, `*PAR:`/`*INV:` separation, and all existing disfluency/pause/retracing codes into a unified per-utterance feature table. This is the foundation for the input side and is pure engineering.
2. **Repair-density audit**: quantify, per language, how many investigator turns, re-asks, and candidate repair patterns actually exist. This both builds the pipeline and empirically checks the v1 construct decision. If repair proves genuinely too sparse (even after Aphasia/TBI), fall back per Section 11 — from data, not guesswork.
3. **Annotation protocol draft** for interactional-repair events and perceptual ratings (the part CHAT cannot supply): precise definitions, the repair/event taxonomy adapted from existing schemes, the rating scale and window unit, and a small pilot to measure inter-annotator agreement — ready to apply the moment Aphasia/TBI land.

**On arrival of AphasiaBank/TBIBank:** extend the parser to their protocols, run the repair-density audit on their richer dialogue, and apply the piloted annotation protocol to build the repair-labelled gold set.
