# vabench — P1 (benchmark + baselines) & P2 (estimator + invariance)

Export this whole `benchmark/` directory to your HPC. Everything below `torch`
in requirements.txt is GPU-side; the data pipeline and baselines run anywhere.

## Paper mapping

| Repo component | Paper | Claim it supports |
|---|---|---|
| `corpus_index` + `splits` + annotation layer | P1 | benchmark construction, no-leakage guarantee |
| `labels.py` (silver vs gold separation) | P1 | non-circular measurement model (review §B1) |
| `baselines/run_baseline.py` (B0/B1/B2) | P1 | transcript-feature reference points |
| `features/ssl_embed.py` + probe (`--adv-* 0`) | P1/P2 | frozen-SSL baseline, per-layer probing |
| `models/estimator.py` + `train.py` ablations | P2 | invariance method (RQ2) |
| LOLO/LOPO folds + `evaluate.py` | P2 | zero-shot transfer (RQ3), per-language, calibration |
| `state_vs_trait` metric | P1+P2 | state-not-trait discriminant validity |

## Workflow on HPC

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 0. edit configs/default.yaml: data_root + corpora paths for your layout
# 1. index + splits (CPU, minutes). Prints timestamp coverage —
#    if low for a corpus, forced-align before windowed work (MFA/CTC-seg).
python scripts/build_index.py --config configs/default.yaml

# 2. transcript-feature baselines (CPU)
python -m vabench.baselines.run_baseline --scheme iid
python -m vabench.baselines.run_baseline --scheme lolo
python -m vabench.baselines.run_baseline --scheme lopo

# 3. cache SSL embeddings (GPU)
sbatch slurm/embed.sbatch

# 4. probing + invariance ablations (GPU array job: probe/lang/pop/both/all)
sbatch slurm/train_ablations.sbatch

# 5. zero-shot transfer, best config on every LOLO/LOPO fold (GPU)
sbatch slurm/zeroshot.sbatch
```

## Verify locally (no data, no GPU needed)

```bash
python tests/make_synth_corpus.py tests/synth_data
python scripts/build_index.py --config configs/synth_test.yaml
python -m vabench.baselines.run_baseline --config configs/synth_test.yaml --scheme iid
```

## Label discipline (do not skip)

- `silver_pdi` is computed from CHAT codes. Transcript-feature models scored
  against it are **plumbing/oracle numbers** — never report them as findings.
  Audio-only models against silver_pdi are legitimate weak-label results.
- `gold_rating` / `repair_event` come from the annotation protocol
  (`../annotation/annotation_protocol_v0.md`); drop CSVs under `labels/{corpus}/`
  and set `labels.target: gold_rating`. Headline results = audio-only model,
  gold labels, per-language reporting.
- `group` (diagnosis) is for known-groups **validation only** — it must never
  appear in a feature matrix or training target.

## P2 decision rule (from the plan review)

After `train_ablations` + `zeroshot`: if `adv_both`/`adv_all` beat `probe` by
a clear margin on held-out language AND population **while keeping
within-speaker sensitivity** (state-vs-trait block), the invariance method is
a paper (Interspeech/ACL, then possibly generalized for ICLR 2028). If not,
P1 stands alone and P2 becomes an analysis paper on why invariance is hard —
which is itself publishable given RQ2's mechanism argument.
