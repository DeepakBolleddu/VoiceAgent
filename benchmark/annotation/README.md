# Gold annotation — the critical path to the paper's claims

Silver-PDI only has signal in English + FluencyBank (other languages' transcripts
lack disfluency coding — see `scripts/diagnose_index.py`). Every cross-lingual /
cross-population claim therefore needs **gold labels in ≥2–3 languages**. This is
the pipeline for producing them.

## Loop

```bash
# 1. Sample utterances to annotate (stratified + repair-enriched, with context)
python scripts/sample_for_annotation.py --config configs/hpc.yaml \
    --languages eng,zho,deu --per-language 200 --context 2
#    -> annotation/manifests/{eng,zho,deu}.json

# 2. Each annotator opens annotation/annotator.html in a browser:
#      - enters Rater ID, picks Condition (audio+transcript / transcript-only)
#      - loads a manifest json
#      - (optional) loads the wav16k folder to hear each turn
#      - rates difficulty 0-4 + ticks any repair events
#      - clicks Export -> ratings_{rater}_{cond}.csv, events_{rater}_{cond}.csv

# 3. Collect exports into annotation/returned/, then:
python scripts/ingest_annotations.py --inbox annotation/returned --out labels
#    -> labels/{corpus}/{session}.{ratings,events}.csv  + IAA report

# 4. If alpha passes the gate, switch the target and re-run:
#    edit configs/hpc.yaml: labels.target: gold_rating
python -m vabench.baselines.run_baseline --config configs/hpc.yaml --scheme iid --target gold_rating
```

## Design points (from the plan review)

- **Two rater conditions.** `audio+transcript` is the deployment-realistic rating;
  `transcript-only` is acoustically independent. `ingest` reports the correlation
  between them — if it's high, acoustics dominate the ratings and we treat
  **repair events (Tier 1) as the primary criterion** (review §B1).
- **Context travels with each item** so repair (a cross-turn phenomenon) is
  judgeable. The target turn is highlighted.
- **Enrichment, not labels.** The sampler biases toward repair-adjacent turns so
  the pilot isn't all easy utterances; the annotator makes the actual call.
- **IAA gates:** α≥0.7 scale up · 0.5–0.7 refine definitions & re-pilot · <0.5
  revise the construct before any modelling.

## Pilot recommendation

Start with 2 native raters each in **English, Mandarin, German** (languages with
audio + some silver signal to sanity-check against), ~150–200 items each, both
conditions on a shared 20-item calibration subset first. That yields the first
gold set and the reliability numbers P1 needs.
