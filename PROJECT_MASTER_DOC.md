# Cross-Lingual, Cross-Population Communicative-Difficulty Sensing
## Project Master Document — concepts, methods, progress, and roadmap

Author: Deepak Bolleddu · Status: living document · Last updated: August 2026
Companion artifacts: `benchmark/` (code), `P1_paper_skeleton.md`, `GOLD_PILOT_STRATEGY.md`, `demo_agent/`

---

# 1. The project in one paragraph

We estimate **communicative difficulty** — a graded, moment-to-moment interactional *state*, explicitly distinct from any per-person diagnostic *label* — from speech, across languages and clinical populations, and couple that signal to an adaptation policy in a clinical-communication voice agent. The scientific contributions are (i) the construct and its non-circular measurement model, (ii) **VABench**, the first cross-lingual (8 languages), cross-population benchmark for this task, and (iii) an invariance analysis showing *which* axis of variation needs explicit handling. The agent is a demonstrator of the sensing→adaptation loop, not the contribution.

# 2. Core concepts

**The construct.** Communicative difficulty at moment *t* = the graded, time-varying degree to which producing and grounding meaning is effortful or disrupted. Grounded in psycholinguistic processing effort (pauses, fillers, restarts as fingerprints of load) and Conversation-Analysis grounding/repair (clarification, re-asks, non-uptake as the cost of restoring understanding). Three distinctions it must preserve: **state not trait** (varies within a conversation; diagnosis does not — this separates us from the detection literature), **latent cause not one indicator** (disfluency is a symptom, not the definition), and **two loci** — *production difficulty* (monological; carries model inputs) and *interactional repair* (dialogical; carries independent labels).

**The non-circularity rule.** Model inputs and training labels must come from different observational sources. Inputs: acoustic/prosodic markers via a multilingual SSL backbone. Labels: interactional-repair events (from the *other* speaker's behavior — genuinely independent) as primary criterion; perceptual ratings as convergent evidence, collected in two conditions (audio+transcript and transcript-only) so their acoustic dependence is itself measured.

**Silver vs gold.** `silver_pdi` — a rate-normalized, within-(corpus,language) z-scored disfluency/pause burden computed from CHAT codes — is a *bootstrap* target: legitimate for audio-input models (weak supervision), never a headline evaluation, and absent/degenerate in code-poor languages. Gold = human ratings (0–4) + repair-event annotation, per the two-tier protocol with Krippendorff-α gates (≥0.7 scale, 0.5–0.7 refine, <0.5 revise construct).

**Invariance (RQ2/RQ3).** Can one representation be language- and population-invariant yet difficulty-sensitive, given that populations produce the same surface disfluency for different reasons (dementia word-finding vs stuttering blocks)? Tested via adversarial (GRL/DANN) ablations and leave-one-language/population-out zero-shot folds.

**State-vs-trait metric.** Within-speaker share of prediction variance + mean within-speaker rank correlation: a covert diagnosis detector scores ~0; our signals score ~0.8 variance share — the construct's discriminant validity, quantified.

# 3. Methods built (and the hard lessons encoded in them)

**Data layer.** CHAT parser (tiers, media bullets + bare-timestamp fallback, retracing/pause codes, `%mor`); **language-agnostic disfluency extraction** — lexical filler lexicons (음/嗯/äh/εμ…), `%mor` interjection tags, immediate word repetitions — after discovering Korean/Greek/Mandarin transcripts encode disfluency as words, not English CHAT codes; **multi-party handling** (Greek Dem@Care `PAR0..PARn` = distinct session-scoped speakers; Spanish PerLA family tiers = interlocutors); speaker keys `corpus:lang:subcorpus:sid` preventing three distinct leakage/merge failure modes; globally unique path-based `utt_id` with a fatal duplicate guard (added after a stem-collision cross-join silently corrupted a full ablation run); dx-label normalization + collision warnings (longitudinal Pitt/Chou dx progression = expected, leakage-safe by construction).

**Audio layer.** Bundled-ffmpeg parallel preprocessing (no cluster modules needed; corrupt-file tolerant) → 16 kHz mono cache mirroring relative paths; **exact-path audio↔transcript matching** (99.2% coverage; collision-proof — replaced a stem-based matcher that risked attaching wrong-corpus audio); XLS-R-300M embedding cache, all 25 layers, 65,367 participant utterances.

**Model layer.** Per-layer ridge probing; MLP estimator over the peak layer with GRL adversaries (language / population / speaker) on a DANN schedule; corpus-balanced sampling (largest corpus cannot dominate batches); dev-based early stopping with best-checkpoint restore **including carved dev sets inside zero-shot folds** (added after last-epoch-lottery artifacts produced a false "over-regularization" finding); multi-seed evaluation (mean±sd).

**Evaluation layer.** Speaker-independent splits, longitudinal sessions kept together; language/population-stratified; LOLO (7 folds) + LOPO (2 folds) zero-shot; metrics: Spearman/CCC/MAE per-language (never pooled-only), AUROC/AP/ECE for repair, known-groups (dx validates, never labels), state-vs-trait.

**Annotation layer.** Stratified repair-enriched sampler with conversational context; browser annotation tool (two rater conditions, audio clip playback, keyboard-driven); ingest + Krippendorff α + audio-vs-transcript independence check; outputs merge back as gold targets keyed to TalkBank IDs (the releasable resource — audio is never redistributed).

**Demonstrator.** Push-to-talk agent: XLS-R→estimator difficulty percentile (calibrated on 65k clinical utterances) fused with interpretable pause/latency features; tiered policy FLOW→SLOW→SIMPLIFY→TEACH_BACK→REGROUND (choice-reduction, validation-therapy rules, anchor memory); 3-turn escalation matrix with caregiver alert; LLM reply brain (Claude/Ollama/reflective fallback); neural TTS with per-tier rate+pitch; session report.

# 4. What we have done (with results)

**The benchmark (P1 backbone) — done.**

| Corpus | Lang | Speakers | Utts | ts_cov | Character |
|---|---|---|---|---|---|
| Dem_English (Pitt) | eng | 296 | 22,353 | 89% | picture description + interview |
| Dem_German | deu | 222 | 2,193 | 97% | |
| Dem_Greek (Dem@Care) | ell | 565 | 28,861 | 99.9% | multi-party group talk — repair-rich |
| Dem_Korean (Kang) | kor | 77 | 3,784 | 100% | interview — repair-rich |
| Dem_Mandarin | zho | 260 | 4,567 | 95% | |
| Dem_Spanish_Ivanova | spa | ~357 | ~1,580 | 99.7% | recall monologue |
| Dem_Spanish_PerLA | spa+**cat** | — | — | — | multi-party family talk (+Catalan) |
| FluencyBank (Voices-AWS) | eng | 67 | 5,636 | 98% | stuttering — distinct mechanism |
| Dem_Taiwanese (Lu) | nan | 16 | 278 | 0% | unlinked media; gold-only holdout |

Totals: **~107k utterances, ~2,064 speakers, 8 languages, 9 corpora**; leakage-safe splits; 99.2% audio matched; repair-density audit identifies Greek/Korean/English/PerLA as annotation priorities.

**Findings so far (silver-label caveat on all):**
1. **Probing:** difficulty is decodable from audio alone, peaking at **XLS-R layer 18** (ρ=0.254 linear), with consistent signal in *every* language (0.16–0.28) — preliminary evidence of a language-general signal.
2. **Estimator:** IID test ρ=0.396 (audio-only vs transcript-derived target = legitimate weak supervision).
3. **Zero-shot (3 seeds, dev-selected):** language-adversarial invariance does **not** improve cross-lingual transfer (all LOLO deltas within noise — multilingual SSL is already language-invariant for this signal). Population-adversarial invariance **does** improve cross-population transfer (LOPO dementia 0.139±0.025 vs probe 0.094±0.025, consistent across seeds). **The axis needing explicit invariance is population, not language** — P2's central claim.
4. **Transfer gap:** IID→zero-shot roughly halves performance (eng 0.395→0.208) — the problem the benchmark exists to measure.
5. **State-vs-trait:** within-speaker variance share ≈0.8 across models — the signal is a state.
6. Methodological finding worth publishing: zero-shot evaluation without in-fold model selection produces artifacts (a fake "over-regularization collapse" that vanished under proper selection + seeds).

**The demonstrator — working.** Live conversation, difficulty meter with interpretable evidence panel, tier adaptation, escalation alerts, neural prosody, session reports. Suitable for the supervisor/industry demo with stated caveats (healthy speakers out-of-domain; thresholds are demo defaults).

# 5. What remains — phases and papers

## Paper 1 — construct + benchmark + annotation + baselines
Target: **NeurIPS 2027 Datasets & Benchmarks (primary; deadline ~May 2027)** or NAACL 2027 (ARR ~Oct 2026 — only if gold annotation moves unusually fast; not recommended to chase).

| Component | Status |
|---|---|
| Construct definition + measurement model | ✅ done |
| Benchmark construction, splits, audio, diagnostics | ✅ done |
| Transcript baselines + SSL probing | ✅ done (tables in skeleton) |
| Annotation protocol + tooling | ✅ built; **pilot not yet run** |
| Gold labels + reliability (α) + validity analyses | ❌ **critical path** |
| AphasiaBank/TBIBank integration | ❌ pending access |
| Related work + writing | ~15% (skeleton exists) |

**Overall P1 progress ≈ 65%.** Everything left is annotation-shaped or writing-shaped, not code-shaped.

## Paper 2 — estimator + invariance analysis
Target: **Interspeech 2027 (deadline ~Feb–Mar 2027)** or ACL 2027 via ARR. Note the ordering: P2's deadline comes *before* P1's — plan for P2 experiments to freeze by ~Jan 2027.

| Component | Status |
|---|---|
| Estimator + ablation grid + zero-shot machinery | ✅ done |
| Silver-label results (probe/adv, LOLO/LOPO, 3 seeds) | ✅ done |
| Gold-label confirmation of the population-invariance claim | ❌ gated on P1 gold |
| Aphasia/TBI populations (makes LOPO 4-way, the real test) | ❌ pending access |
| 5-seed firm-up of LOPO-dementia headline | ❌ cheap, queued |
| Analysis framing + writing | ~10% |

**Overall P2 progress ≈ 55%.** The narrative is already clear ("language is free, population is not"); it needs gold + more populations to be defensible.

## Timeline (working backwards from deadlines)

| When | What |
|---|---|
| **Now – Sep 2026** | English gold pilot (self + 1 colleague, 150 items) → first α. Request AphasiaBank/TBIBank. Prof demo. 5-seed LOPO firm-up. |
| **Oct – Nov 2026** | Aphasia/TBI ingested (config + reindex + embed ~1 week of work). Repair-density audit on their dialogue. Gold campaign: Greek + Korean repair-only, small native test sets (per GOLD_PILOT_STRATEGY). |
| **Dec 2026 – Jan 2027** | Gold-label re-runs of estimator + invariance on the expanded benchmark. Freeze P2 experiments. |
| **Feb – Mar 2027** | **P2 submission (Interspeech).** P1 validity analyses + writing. |
| **Apr – May 2027** | **P1 submission (NeurIPS D&B).** |
| **Phase 2 (2027+)** | Clinical partner + HREC ethics; prospective user study; verbatim clinical ASR (fine-tune Whisper on our own CHAT-verbatim pairs — we already own the training data); streaming VAD with adaptive hold-times; SSML prosody; TGA SaMD boundary formalized; Indigenous/low-resource languages with cultural-safety governance. |

# 6. Data strategy — AphasiaBank, TBIBank, and what else

## Getting AphasiaBank & TBIBank (do this now — longest admin lead time)
Both are password-protected TalkBank clinical banks, same regime as DementiaBank/FluencyBank (which you already hold — so the route is proven):
1. Apply for membership for each bank via TalkBank (email the TalkBank coordinators, e.g. Brian MacWhinney's group at CMU) stating: academic affiliation, supervisor, research purpose, agreement to the Ground Rules (no redistribution, no re-identification, cite the corpora). Supervisor co-sign accelerates it.
2. On approval, download **transcripts and media** (two trees, as with dementia).
3. Integration cost on our side is small and proven: add corpora to `configs/hpc.yaml` (AphasiaBank has non-English subsets — check Mandarin/Cantonese/Spanish availability, which would fill language×population cells), run `pbs/00` (reindex; the diagnostic will flag any new tier/code conventions), `pbs/01–02` (audio+embed for the new corpora only), extend filler lexicons if the diagnostic shows word-encoded disfluency, then run the repair-density audit — AphasiaBank/TBIBank protocols are conversation-rich, which is where the repair-label fuel was always expected to come from.

## Why more data, precisely: deconfounding language × population
The benchmark's main structural weakness (named honestly in the plan review): **population is confounded with corpus and partly with language** (dementia = 7 languages, fluency = English only). New data should be chosen to fill the grid, not just to grow it:

| | English | German | Mandarin | Spanish | Greek/Korean/… |
|---|---|---|---|---|---|
| Dementia | ✅ Pitt | ✅ | ✅ | ✅ | ✅ |
| Fluency/stuttering | ✅ FluencyBank | ❌ **KSoF fills this** | ❌ | ❌ | ❌ |
| Aphasia | ⏳ AphasiaBank | — | ⏳ check AphasiaBank-Mandarin | ⏳ check | — |
| TBI | ⏳ TBIBank | — | — | — | — |
| Healthy conversation | ⏳ CABank/CallHome | — | — | — | — |

## Recommended additional public datasets (priority order)
1. **AphasiaBank non-English subsets** (with your membership) — the single best deconfounder: same population, second+ language.
2. **SEP-28k** (Apple, public): 28k stuttering-event-labeled clips — strengthens the fluency population *and* provides event-level disfluency labels for the acoustic detector.
3. **KSoF** (Kassel State of Fluency, German stuttering; research license): breaks the "fluency = English" confound — German dementia + German stuttering is a clean within-language population contrast.
4. **ADReSS / ADReSSo / ADReSS-M** (curated Pitt/Greek subsets): not new data, but adopting their official test splits gives direct comparability with the detection literature — cheap credibility.
5. **TAUKADIAL** (English+Mandarin MCI, Interspeech challenge): cross-lingual MCI with a known baseline ecosystem.
6. **CABank / CallHome** (healthy conversational speech): a healthy-conversation floor for calibration and false-positive analysis of the difficulty signal (and the agent's escalation thresholds).
7. *(Stretch, for a possible ICLR-2028 generalization)* a cognitive-load-from-speech corpus (ComParE CLSE lineage): tests whether the state/trait disentanglement transfers outside clinical speech entirely.

**Private/hospital data: not needed for P1/P2.** It becomes relevant only in Phase 2 (prospective study with a clinical partner, under HREC ethics) — and the benchmark being public-data-only is a *selling point* for reproducibility.

# 7. Risks (updated)

| Risk | Status / mitigation |
|---|---|
| Gold annotation stalls (rater scarcity) | Staged plan: English self-pilot now; repair-only labels for language 2; small native test sets; full campaign after Aphasia/TBI. P1 is submittable on stages 0–2. |
| Repair too sparse in monologue corpora | Confirmed empirically; Greek/Korean/PerLA are repair-rich; Aphasia/TBI expected richer — audit on arrival. |
| Invariance claim doesn't survive gold | Then P2's finding inverts but remains publishable ("silver artifacts in invariance evaluation") — the analysis framing is robust to either outcome. |
| Language≈corpus confound | Named in limitations; mitigated by task-matched subsets + new grid-filling data. |
| Over-scoping the agent | Demonstrator is feature-frozen except bug fixes; all "clinical-grade" work is Phase 2 by design. |
| Deadline collision (P2 before P1) | Freeze P2 experiments by Jan 2027; P1 writing proceeds in parallel from the skeleton. |

# 8. Deliverables inventory (where everything lives)

`benchmark/` — full pipeline (parser, index, splits, labels, baselines, probing, estimator, PBS jobs, diagnostics) · `benchmark/annotation/` — protocol, sampler, annotator.html, ingest+IAA, ENGLISH_PILOT.md · `artifacts/` (HPC) — index, splits, embeddings (65k×25 layers), runs, reports · `P1_paper_skeleton.md` — paper scaffold with real numbers · `GOLD_PILOT_STRATEGY.md` — rater-constrained annotation plan · `demo_agent/` — live demonstrator + README · `PLAN_REVIEW_rev3.md` — the strict review that set the decision rules this project has been following.
