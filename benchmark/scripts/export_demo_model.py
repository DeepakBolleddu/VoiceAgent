"""
export_demo_model.py — Package the trained estimator for the laptop demo.

Run on the HPC, then scp the resulting demo_model/ folder to your laptop:
  python scripts/export_demo_model.py --config configs/hpc.yaml --run probe
  scp -r demo_model/ you@laptop:~/VoiceAgent/demo_agent/

Bundles: model weights, input dim, SSL layer, and CALIBRATION quantiles of the
model's own predictions on the training corpus — the demo maps a raw score to
a percentile ("this moment is more difficult than X% of clinical speech"),
which is far more presentable than an unanchored z-score.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import yaml

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vabench.tables import read_utterances          # noqa: E402
from vabench.labels import add_silver_pdi            # noqa: E402
from vabench.tiers import is_participant             # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/hpc.yaml")
    ap.add_argument("--run", default="probe")
    ap.add_argument("--out", default="demo_model")
    args = ap.parse_args()

    import torch
    from vabench.models.estimator import DifficultyEstimator

    cfg = yaml.safe_load(Path(args.config).expanduser().read_text())
    out_root = Path(cfg["out_root"]).expanduser()
    layer = cfg["ssl"]["layer"]
    short = cfg["ssl"]["model"].split("/")[-1]
    emb_dir = out_root / "embeddings" / short

    ids = json.loads((emb_dir / "utt_ids.json").read_text())
    id_pos = {u: i for i, u in enumerate(ids)}
    X = np.load(emb_dir / f"layer{layer}.npy")

    state = torch.load(out_root / "runs" / args.run / "model.pt", map_location="cpu")
    n_langs = state.get("head_lang.weight", torch.zeros(0, 0)).shape[0]
    n_pops = state.get("head_pop.weight", torch.zeros(0, 0)).shape[0]
    n_spks = state.get("head_spk.weight", torch.zeros(0, 0)).shape[0]
    model = DifficultyEstimator(in_dim=X.shape[1], n_langs=n_langs,
                                n_pops=n_pops, n_spks=n_spks)
    model.load_state_dict(state)
    model.eval()

    # calibration: model's own score distribution over all embedded PAR speech
    utts = add_silver_pdi(read_utterances(out_root / "index" / "utterances.csv"))
    utts = utts[utts["tier"].map(is_participant)]
    utts = utts[utts["utt_id"].isin(id_pos)]
    rows = utts["utt_id"].map(id_pos).to_numpy()
    with torch.no_grad():
        preds = model(torch.tensor(X[rows], dtype=torch.float32))["difficulty"].numpy()
    qs = np.quantile(preds, np.linspace(0, 1, 101)).tolist()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(state, out / "model.pt")
    (out / "meta.json").write_text(json.dumps({
        "ssl_model": cfg["ssl"]["model"], "layer": int(layer),
        "in_dim": int(X.shape[1]),
        "n_langs": int(n_langs), "n_pops": int(n_pops), "n_spks": int(n_spks),
        "score_quantiles": qs,
        "trained_run": args.run,
    }, indent=1))
    print(f"[export] wrote {out}/model.pt + meta.json "
          f"(calibrated on {len(preds)} clinical utterances)")


if __name__ == "__main__":
    main()
