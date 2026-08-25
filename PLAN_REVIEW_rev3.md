# Strict Review of Rev 2 — Corrections and Publication Strategy

Reviewer stance: adversarial, as an ICLR/ACL area chair would read it. The plan is unusually strong for a design-phase document — the input/label separation rule, the repair-density audit, and the FluencyBank mechanism caveat are things most plans miss. The corrections below are where it would still get killed in review.

---

## Part A — The honest ICLR verdict

**As written, this is not an ICLR paper, and forcing it into one is the biggest strategic risk in the plan.**

ICLR wants a representation-learning contribution of general ML interest. Your plan's center of gravity is a *construct + benchmark + clinical application* — that is an ACL/EMNLP/Interspeech/NeurIPS-D&B shape. Your own Section 12 already says the benchmark-plus-method lane is the cleaner path; trust your own document over the ICLR ambition.

Three concrete facts:

1. **ICLR 2027 is off the table.** Paper deadline is Sept 25, 2026 (abstracts Sept 18) — ~7 weeks away. You have no annotated labels, no splits, no baselines. A rushed submission would burn the idea's novelty in reviewers' memories.
2. **What *could* be an ICLR paper (2028 cycle, deadline ~Sept 2027):** the invariance method alone, reframed as a general problem — *disentangling time-varying states from stable speaker/domain traits in temporal representations*. To survive ICLR review it would need (a) a formalized objective, not "adversarial or disentanglement approaches (e.g.)"; (b) evidence beyond TalkBank — at least one non-clinical domain (cognitive-load or emotion corpora) showing the state/trait separation generalizes; (c) strong disentanglement baselines (adversarial speaker removal, VICReg-style, ContentVec-style content/speaker separation). If the method turns out to be "adversarial language classifier + gradient reversal," it is not an ICLR contribution — that is 2016 technology applied to new data, and reviewers will say so.
3. **The right sequence given real deadlines:**

| Paper | Content | Venue | Deadline |
|---|---|---|---|
| P1 | Construct + benchmark + annotation scheme + baselines | ARR → NAACL 2027 (Oct 12, 2026) if labels exist in time; otherwise NeurIPS 2027 D&B (~May 2027) — the safer, better-fitting target | Oct 2026 / May 2027 |
| P2 | Difficulty estimator + invariance analysis on the benchmark | Interspeech 2027 (~Feb/Mar 2027) or ACL 2027 via ARR | Feb–Mar 2027 |
| P3 (only if the method is real) | Generalized state/trait disentanglement | ICLR 2028 | ~Sept 2027 |
| Demo | Agent demonstrator | ACL/EMNLP demo track, or supervisor/industry demo only | rolling |

Decision rule for P3: after P2's ablations, if the invariance component gives >2–3 points over frozen-SSL + gradient-reversal baselines on held-out language *and* population, you have an ICLR paper. If not, you don't, and P1+P2 are still a strong PhD spine.

---

## Part B — Corrections to the science (ordered by severity)

### B1. The circularity fix is incomplete — perceptual ratings are not an independent source. [must fix]
Section 5 claims labels come from "a different observational source," but perceptual difficulty ratings are produced by humans *listening to the same audio the model ingests*. A rater hearing pauses and fillers rates "difficult"; the model reads pauses and fillers and predicts "difficult" — you have laundered the acoustics through a human, not broken the loop. Repair events are the only genuinely independent criterion (they come from the *other speaker's* subsequent behavior).
**Correction:** make repair events (and downstream interactional consequences: re-asks, non-uptake, task failure) the primary criterion label. Demote audio-based perceptual ratings to convergent-validity evidence. If you want a rating tier that is defensibly independent, have one rater group rate *transcript-only* (no audio) and report both.

### B2. Train-time features won't exist at deployment. [must fix]
CHAT retracing/pause codes are human annotations. The deployed agent gets raw audio and ASR output — no `[//]`, no `&-um` tiers. If the model is trained on CHAT-derived features, the agent demo cannot run the model, and the sensing→adaptation story collapses.
**Correction:** two-stage design, stated explicitly. Stage 1 (development): CHAT codes bootstrap silver labels and validate that automatic markers work — e.g., train an acoustic disfluency/pause detector whose targets are the CHAT codes. Stage 2 (the actual model): audio-only (SSL features + detector outputs). Report the audio-only model as the headline result; CHAT-feature models are an oracle upper bound. FluencyBank's role slots in here cleanly: it trains the acoustic disfluency detector (surface-form supervision), not the difficulty estimator (mechanism supervision) — which also operationalizes your own FluencyBank caveat instead of just stating it.

### B3. Language transfer is confounded with corpus/task/channel — say so or reviewers will. [must fix]
DementiaBank's seven languages are seven *different corpora*: different elicitation tasks, recording eras, microphones, protocols. "Held-out language zero-shot" is inseparable from "held-out corpus zero-shot." The same applies to population: dementia ≡ DementiaBank, aphasia ≡ AphasiaBank — population and corpus are perfectly confounded.
**Corrections:** (a) match tasks across languages where possible (picture description exists in several) and report transfer within-task; (b) exploit within-corpus healthy controls — every bank has them — so "population sensitivity" has a within-corpus contrast; (c) rename the claim honestly: cross-corpus transfer with language and population as dominant factors; (d) add corpus/channel-level augmentation or normalization, and report per-corpus calibration, not just per-language.

### B4. "Released benchmark" cannot include audio. [must fix in deliverable wording]
TalkBank ground rules prohibit redistribution; access requires membership. Your resource contribution is therefore an *annotation layer + code + splits keyed to TalkBank IDs* (the ADReSS-challenge model). This is still a real resource contribution, but write it that way now — "released multilingual benchmark" implying audio release is a promise you cannot keep and reviewers who know TalkBank will notice.

### B5. Moment-to-moment claims need time alignment the plan doesn't mention. [gap]
Many DementiaBank transcripts lack reliable utterance-level timestamps (media bullets are inconsistent across corpora). A "per-moment" estimator needs alignment.
**Correction:** add a forced-alignment step (e.g., Montreal Forced Aligner or CTC-segmentation from the multilingual ASR) to the pipeline, and audit timestamp coverage per corpus in week 1 alongside the repair-density audit. Also resolve the granularity mismatch explicitly: repair events are turn-level, ratings are window-level (3–5 s), production markers are sub-second. Decide the target granularity (recommend: per-utterance primary, windowed secondary) before annotation, not after.

### B6. Speaker-independent splits must dedupe longitudinal visits. [easy, but fatal if missed]
Pitt and several banks are longitudinal — the same speaker appears across sessions years apart. Split by *speaker ID across all sessions*, not by recording. One leaked speaker inflates every transfer number.

### B7. Related-work positioning has two exposed flanks. [framing]
(a) **Cognitive-load-from-speech** (ComParE cognitive-load challenge lineage) already estimates a graded, time-varying effort state from prosody. Your differentiators — interactional repair as criterion, cross-population, clinical speech — are real, but you must cite and position against this line or reviewers will hand you the "already exists" verdict. (b) **Multilingual cognitive detection** (ADReSS-M, TAUKADIAL) owns the "cross-lingual clinical speech" territory; your state-vs-trait distinction is the wedge — lead with it in the intro, with a figure showing within-session variation of your signal versus a flat diagnostic score.

### B8. Adaptation-policy evaluation: you can only measure trigger appropriateness, not adaptation benefit. [claim discipline]
Offline replay tells you the signal fires at the right moments. It cannot tell you the adaptation *helped* — that counterfactual (what would the patient have done had the agent slowed down?) is unobservable in replay. Rev 2 mostly respects this; tighten the wording so RQ4 claims "appropriate firing" only, and move any "improves communication" language to the Phase-2 user study. Also define the RQ4 metric now: precision/recall of policy triggers against annotated difficulty/repair moments, plus escalation precision (false escalations are the costly error in a clinical setting).

### B9. Minor but worth fixing.
"Taiwanese" in the DementiaBank language list — use "Mandarin (Taiwan)" or the corpus's own label; reviewers from the region will flag it. The eGeMAPS/probing line deserves promotion from aside to a committed interpretability section — clinical reviewers trust interpretable features more than SSL embeddings, and it doubles as your ablation infrastructure.

---

## Part C — Product shaping (kept minimal per your direction)

For the supervisor/industry demo: a turn-based, push-to-talk cascaded pipeline for 2 languages (English ↔ Mandarin or Arabic), with the trust layer visible in the UI (confidence gate, back-translation display, readback of numbers/drugs, escalate button). The demo's purpose is to make the *sensing signal visible* — a live difficulty meter over the conversation — not to be a product.

One regulatory fact to keep in your back pocket for the industry conversation: whether this is regulated by the TGA as Software as a Medical Device turns on **intended purpose**. A tool that translates and flags communication difficulty, with a human always in the loop and no diagnostic/treatment claims, has a defensible position outside SaMD or in the lowest class — but the moment marketing says "detects cognitive impairment," it becomes a regulated device. Your "no diagnostic claims" boundary is thus a regulatory strategy, not just an ethical one. Write the intended-use statement early; it constrains everything downstream. (See TGA guidance on software-based medical devices.)

Defer everything else — procurement, interpreter-service integration, business model — until P1 exists.

---

## Part D — What changes in the immediate work plan

Your Section 13 survives review almost intact. Three amendments:

1. **Add timestamp/alignment audit** to week 1 (per B5): for each corpus, what fraction of utterances carry usable time bullets? This decides whether forced alignment is required before any windowed labeling.
2. **Re-scope FluencyBank's job** (per B2): it feeds the acoustic disfluency detector, not the difficulty model. The parser treats it identically; the modeling plan does not.
3. **Add the transcript-only rating tier** (per B1) to the annotation protocol pilot, so the independence question is answered with pilot data, not argument.

The implementation of Section 13 items 1–3 accompanies this review: `code/` (CHAT parser, feature extraction, repair-density + timestamp audit, verified on synthetic CHAT files) and `annotation/annotation_protocol_v0.md`.

---

## Verdict

Keep: the construct, the input/label separation instinct, the two-locus design, the sequencing, the risk register. Fix: label independence (B1), deployment feature mismatch (B2), corpus confounds (B3), benchmark wording (B4). Abandon: ICLR 2027. Re-aim: NeurIPS D&B / ARR for the benchmark, Interspeech/ACL for the estimator, ICLR 2028 only if the invariance ablations earn it. This is a strong PhD plan pretending to be one paper; let it be three.
