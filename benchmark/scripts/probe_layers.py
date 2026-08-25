"""
probe_layers.py — Per-layer linear probing of communicative difficulty on the
cached XLS-R embeddings (plan §7.1 "per-layer probing").

For each SSL layer, fit a ridge probe on frozen embeddings (speaker-independent
train split) and predict the difficulty target on the test split. Reports ρ/CCC
per layer, overall and per language, and picks the best layer.

Why this is a real result (not circular): the probe input is AUDIO embeddings;
the silver target is derived from TRANSCRIPT codes — different observational
sources. So "audio predicts difficulty" is a legitimate weak-label finding, and
where it peaks across layers is a genuine interpretability result. Swap to
--target gold_rating once annotation lands.

Usage:
  python scripts/probe_layers.py --config configs/hpc.yaml \
      [--languages eng,eng_fluency] [--target silver_pdi] [--layers all]
CPU-friendly; run on a login/compute node without a GPU.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vabench.tables import read_utterances          # noqa: E402
from vabench.labels import add_silver_pdi, load_gold  # noqa: E402
from vabench.tiers import is_participant             # noqa: E402
from vabench.evaluate import graded_metrics          # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/hpc.yaml")
    ap.add_argument("--target", default="silver_pdi")
    ap.add_argument("--languages", default="", help="restrict eval to these (comma list)")
    ap.add_argument("--layers", default="all", help="'all' or comma list e.g. 8,12,16")
    ap.add_argument("--alpha", type=float, default=10.0, help="ridge regularization")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).expanduser().read_text())
    out_root = Path(cfg["out_root"]).expanduser()
    short = cfg["ssl"]["model"].split("/")[-1]
    emb_dir = out_root / "embeddings" / short

    ids = json.loads((emb_dir / "utt_ids.json").read_text())
    id_pos = {u: i for i, u in enumerate(ids)}
    layers = (sorted(int(p.stem[5:]) for p in emb_dir.glob("layer*.npy"))
              if args.layers == "all" else [int(x) for x in args.layers.split(",")])

    utts = read_utterances(out_root / "index" / "utterances.csv")
    utts = add_silver_pdi(utts)
    utts = load_gold(utts, Path(cfg["labels"]["gold_dir"]).expanduser())
    utts = utts[utts["tier"].map(is_participant)]
    utts = utts[utts["utt_id"].isin(id_pos)].dropna(subset=[args.target])
    utts["_row"] = utts["utt_id"].map(id_pos)

    splits = json.loads((out_root / "splits" / "splits.json").read_text())["iid"]
    tr_spk, te_spk = set(splits["train"]), set(splits["test"])
    tr = utts[utts["speaker"].isin(tr_spk)]
    te = utts[utts["speaker"].isin(te_spk)]

    langs = [l.strip() for l in args.languages.split(",") if l.strip()]
    report = {}
    print(f"{'layer':>6}{'overall_rho':>12}{'overall_ccc':>12}   per-language rho")
    best = (-1, -2.0)
    for L in layers:
        X = np.load(emb_dir / f"layer{L}.npy")
        Xtr, ytr = X[tr["_row"].to_numpy()], tr[args.target].to_numpy()
        Xte = X[te["_row"].to_numpy()]
        sc = StandardScaler().fit(Xtr)
        reg = Ridge(alpha=args.alpha).fit(sc.transform(Xtr), ytr)
        te = te.copy(); te["pred"] = reg.predict(sc.transform(Xte))
        overall = graded_metrics(te[args.target], te["pred"])
        per = {}
        for lang, g in te.groupby("language"):
            if langs and lang not in langs:
                continue
            per[lang] = graded_metrics(g[args.target], g["pred"])
        report[L] = {"overall": overall, "per_language": per}
        lang_str = "  ".join(f"{k}:{v['spearman']:.2f}" for k, v in per.items()
                             if v["spearman"] == v["spearman"])
        print(f"{L:>6}{overall['spearman']:>12.3f}{overall['ccc']:>12.3f}   {lang_str}")
        if overall["spearman"] == overall["spearman"] and overall["spearman"] > best[1]:
            best = (L, overall["spearman"])

    out = out_root / "reports"; out.mkdir(parents=True, exist_ok=True)
    (out / f"probe_layers_{args.target}.json").write_text(
        json.dumps(report, indent=2, default=float))
    print(f"\nBest layer overall: {best[0]} (rho={best[1]:.3f}). "
          f"Set ssl.layer: {best[0]} in the config for the estimator.")
    print(f"Wrote {out / f'probe_layers_{args.target}.json'}")


if __name__ == "__main__":
    main()
