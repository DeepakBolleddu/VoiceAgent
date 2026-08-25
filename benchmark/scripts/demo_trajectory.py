"""
demo_trajectory.py — The supervisor-demo figure: moment-to-moment communicative
difficulty over real conversations, from the trained estimator.

For a handful of sessions (chosen to contrast severity groups), plot predicted
difficulty per participant utterance across the session timeline. This is the
"state, not trait" money-shot: the signal moves WITHIN a conversation, unlike a
flat diagnostic label.

Usage (after pbs/03 has produced a run):
  python scripts/demo_trajectory.py --config configs/hpc.yaml \
      --run probe            # run name under artifacts/runs/
      [--sessions 6] [--out demo_figs]

Outputs: demo_figs/trajectory_{corpus}_{session}.png + a combined overview.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vabench.tables import read_utterances          # noqa: E402
from vabench.labels import add_silver_pdi            # noqa: E402
from vabench.tiers import is_participant             # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/hpc.yaml")
    ap.add_argument("--run", default="probe", help="run dir name under artifacts/runs/")
    ap.add_argument("--sessions", type=int, default=6)
    ap.add_argument("--out", default="demo_figs")
    args = ap.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
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

    utts = add_silver_pdi(read_utterances(out_root / "index" / "utterances.csv"))
    utts = utts[utts["tier"].map(is_participant)]
    utts = utts[utts["utt_id"].isin(id_pos)].copy()
    utts["_row"] = utts["utt_id"].map(id_pos)

    run_dir = out_root / "runs" / args.run
    state = torch.load(run_dir / "model.pt", map_location="cpu")
    # infer head sizes from the checkpoint
    n_langs = state.get("head_lang.weight", torch.zeros(0, 0)).shape[0]
    n_pops = state.get("head_pop.weight", torch.zeros(0, 0)).shape[0]
    n_spks = state.get("head_spk.weight", torch.zeros(0, 0)).shape[0]
    model = DifficultyEstimator(in_dim=X.shape[1], n_langs=n_langs,
                                n_pops=n_pops, n_spks=n_spks)
    model.load_state_dict(state)
    model.eval()

    with torch.no_grad():
        preds = model(torch.tensor(X[utts["_row"].to_numpy()],
                                   dtype=torch.float32))["difficulty"].numpy()
    utts["pred"] = preds

    # pick contrasting sessions: longest per (corpus, group), max diversity
    utts["n"] = 1
    sess = (utts.groupby(["corpus", "session", "group"])
            .agg(n=("n", "sum"), lang=("language", "first")).reset_index()
            .sort_values("n", ascending=False))
    picks, seen_groups = [], set()
    for r in sess.itertuples():
        key = (r.corpus, r.group)
        if key in seen_groups or r.n < 10:
            continue
        seen_groups.add(key)
        picks.append((r.corpus, r.session, r.group, r.lang))
        if len(picks) >= args.sessions:
            break

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(picks), 1, figsize=(10, 2.2 * len(picks)),
                             sharex=False)
    if len(picks) == 1:
        axes = [axes]
    for ax, (corp, sessn, grp, lang) in zip(axes, picks):
        g = utts[(utts["corpus"] == corp) & (utts["session"] == sessn)].sort_values("utt_index")
        t = (g["start_ms"].astype(float) / 60000.0).ffill()
        ax.plot(t, g["pred"], marker="o", ms=3, lw=1, label="predicted difficulty")
        ax.axhline(g["pred"].mean(), color="gray", ls="--", lw=1,
                   label="session mean (a 'trait' view)")
        ax.fill_between(t, g["pred"], g["pred"].mean(),
                        where=g["pred"] > g["pred"].mean(), alpha=0.2)
        ax.set_title(f"{corp} / {sessn}  [{grp or 'unknown'}, {lang}]", fontsize=9)
        ax.set_ylabel("difficulty")
        ax.legend(fontsize=7, loc="upper right")
    axes[-1].set_xlabel("time (minutes)")
    fig.suptitle("Communicative difficulty is a STATE: it moves within a conversation",
                 fontsize=11)
    fig.tight_layout()
    p = outdir / "trajectories_overview.png"
    fig.savefig(p, dpi=160)
    print(f"[demo] wrote {p} ({len(picks)} sessions)")

    # bonus: within-speaker variance share (the state-vs-trait number, on preds)
    total = utts["pred"].var()
    within = (utts["pred"] - utts.groupby("speaker")["pred"].transform("mean")).var()
    print(f"[demo] within-speaker share of prediction variance: {within/total:.2f} "
          f"(higher = state-like; a pure trait detector would be ~0)")


if __name__ == "__main__":
    main()
