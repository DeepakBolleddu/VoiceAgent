# Pipeline — Section 13 implementation

Stdlib-only Python (3.10+). No installs needed.

## Files
- `chat_parser.py` — CHAT (.cha) parser: tiers, media-bullet timestamps, disfluency/pause/retracing codes, per-utterance feature rows.
- `repair_audit.py` — repair-density + timestamp-coverage audit (CLI), also emits the unified per-utterance feature table.
- `tests/` — synthetic .cha samples + verification tests (all passing).

## Run on real data
When DementiaBank is in this folder (e.g. `data/DementiaBank/Pitt/`):

```bash
cd code
python repair_audit.py ../data/DementiaBank/Pitt --corpus Pitt \
  --out ../reports/pitt_audit.csv --features-out ../reports/pitt_features.csv
```

Repeat per corpus/language. The console prints the per-language summary that answers the §13 audit questions: INV turn share, question/re-ask/NTRI density, candidate repair sequences, and **timestamp coverage** (if low, forced alignment is required before windowed labeling — see review §B5).

## Verify
```bash
cd code && python tests/make_samples.py && python tests/test_parser.py
```
