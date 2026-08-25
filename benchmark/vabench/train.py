"""
train.py — Train the difficulty estimator on cached SSL embeddings (P2).

Runs one (scheme, fold, ablation) cell:
  python -m vabench.train --config configs/default.yaml --scheme iid
  python -m vabench.train --scheme lolo --fold zho --adv-lang 0.1 --adv-pop 0.1

Outputs: {out_root}/runs/{run_name}/ {model.pt, report.json}
Evaluation goes through vabench.evaluate.report — per-language + state-vs-trait
always included, so every ablation cell is directly comparable.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def load_embeddings(cfg: dict, layer: int):
    out_root = Path(cfg["out_root"]).expanduser()
    short = cfg["ssl"]["model"].split("/")[-1]
    d = out_root / "embeddings" / short
    ids = json.loads((d / "utt_ids.json").read_text())
    X = np.load(d / f"layer{layer}.npy")
    return ids, X


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--scheme", default="iid", choices=["iid", "lolo", "lopo"])
    ap.add_argument("--fold", default=None)
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--adv-lang", type=float, default=None)
    ap.add_argument("--adv-pop", type=float, default=None)
    ap.add_argument("--adv-spk", type=float, default=None)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from .evaluate import print_report, report
    from .labels import add_silver_pdi, load_gold
    from .models.estimator import DifficultyEstimator, dann_lambda, loss_fn
    from .splits import attach_split

    cfg = yaml.safe_load(Path(args.config).expanduser().read_text())
    t = cfg["train"]
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    weights = {"lang": args.adv_lang if args.adv_lang is not None else t["adv_language_weight"],
               "pop": args.adv_pop if args.adv_pop is not None else t["adv_population_weight"],
               "spk": args.adv_spk if args.adv_spk is not None else t["adv_speaker_weight"]}
    layer = args.layer if args.layer is not None else cfg["ssl"]["layer"]
    target = t["target"]
    out_root = Path(cfg["out_root"]).expanduser()

    # --- data ---------------------------------------------------------------
    from .tables import read_utterances
    utts = read_utterances(out_root / "index" / "utterances.csv")
    utts = add_silver_pdi(utts)
    utts = load_gold(utts, Path(cfg["labels"]["gold_dir"]).expanduser())
    ids, X = load_embeddings(cfg, layer)
    emb = pd.DataFrame({"utt_id": ids, "_row": range(len(ids))})
    df = utts.merge(emb, on="utt_id").dropna(subset=[target])
    splits = json.loads((out_root / "splits" / "splits.json").read_text())
    df = attach_split(df, splits, args.scheme, args.fold)

    # Zero-shot folds (lolo/lopo) carry only train/test — carve a speaker-level
    # dev set out of train so model selection & early stopping work there too.
    # Without this, zero-shot numbers are last-epoch-weights (a lottery).
    if args.scheme != "iid":
        # dev carve is deterministic per fold (splits seed), independent of the
        # training seed, so all seeds of one fold share the same dev speakers
        rng = np.random.default_rng(cfg["splits"]["seed"])
        tr_spk = df.loc[df["split"] == "train", "speaker"].unique()
        rng.shuffle(tr_spk)
        dev_spk = set(tr_spk[:max(1, len(tr_spk) // 10)])
        df.loc[df["speaker"].isin(dev_spk), "split"] = "dev"

    # Graceful skip: a fold with no embedded test rows (e.g. Taiwanese — no
    # timestamps -> no embeddings) must not crash the whole sweep.
    if (df["split"] == "test").sum() == 0:
        print(f"[train] SKIP {args.scheme}/{args.fold}: no test utterances with "
              f"embeddings (e.g. corpus without timestamps).")
        return

    langs = sorted(df["language"].unique())
    pops = sorted(df["population"].unique())
    spks = sorted(df.loc[df["split"] == "train", "speaker"].unique())
    lmap = {v: i for i, v in enumerate(langs)}
    pmap = {v: i for i, v in enumerate(pops)}
    smap = {v: i for i, v in enumerate(spks)}

    def tensors(part):
        d = df[df["split"] == part]
        return d, TensorDataset(
            torch.tensor(X[d["_row"].to_numpy()], dtype=torch.float32),
            torch.tensor(d[target].to_numpy(), dtype=torch.float32),
            torch.tensor(d["language"].map(lmap).to_numpy(), dtype=torch.long),
            torch.tensor(d["population"].map(pmap).to_numpy(), dtype=torch.long),
            torch.tensor(d["speaker"].map(smap).fillna(0).to_numpy(), dtype=torch.long),
        )

    dtr, ds_tr = tensors("train")
    dte, ds_te = tensors("test")
    has_dev = (df["split"] == "dev").any()
    if has_dev:
        ddv, ds_dv = tensors("dev")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DifficultyEstimator(
        in_dim=X.shape[1],
        n_langs=len(langs) if weights["lang"] > 0 else 0,
        n_pops=len(pops) if weights["pop"] > 0 else 0,
        n_spks=len(spks) if weights["spk"] > 0 else 0,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=t["lr"])
    # Corpus-balanced sampling: without it the largest corpus (Greek, ~50% of
    # utterances) dominates every batch and the model optimizes its weakest
    # silver signal. Weight each utterance inversely to its corpus size.
    from torch.utils.data import WeightedRandomSampler
    corpus_counts = dtr["corpus"].value_counts()
    w = dtr["corpus"].map(lambda c: 1.0 / corpus_counts[c]).to_numpy()
    sampler = WeightedRandomSampler(torch.tensor(w, dtype=torch.double),
                                    num_samples=len(dtr), replacement=True)
    loader = DataLoader(ds_tr, batch_size=t["batch_size"], sampler=sampler)

    def eval_spearman(ds, d):
        from scipy.stats import spearmanr
        model.eval()
        with torch.no_grad():
            ps = [model(xb.to(device))["difficulty"].cpu().numpy()
                  for (xb, *_) in DataLoader(ds, batch_size=512)]
        p = np.concatenate(ps)
        y = d[target].to_numpy()
        return float(spearmanr(y, p).statistic) if np.std(p) > 0 else -1.0

    steps_total = max(1, len(loader) * t["epochs"])
    step, best_dev, best_state, patience_left = 0, -2.0, None, 5
    for epoch in range(t["epochs"]):
        model.train()
        ep_losses = []
        for xb, yb, lb, pb, sb in loader:
            lam = dann_lambda(step / steps_total, t["grl_gamma"])
            out = model(xb.to(device), lambd=lam)
            loss, parts = loss_fn(out, {"target": yb.to(device),
                                        "lang": lb.to(device),
                                        "pop": pb.to(device),
                                        "spk": sb.to(device)}, weights)
            opt.zero_grad(); loss.backward(); opt.step()
            step += 1
            ep_losses.append(parts["mse"])
        msg = f"[train] epoch {epoch + 1}/{t['epochs']} mean_mse={np.mean(ep_losses):.4f}"
        if has_dev:  # model selection on dev, not last-epoch weights
            dev_rho = eval_spearman(ds_dv, ddv)
            msg += f" dev_rho={dev_rho:.3f}"
            if dev_rho > best_dev:
                best_dev, patience_left = dev_rho, 5
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
            else:
                patience_left -= 1
                if patience_left <= 0:
                    print(msg + "  [early stop]")
                    break
        print(msg)
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"[train] restored best-dev checkpoint (dev_rho={best_dev:.3f})")

    # --- evaluate -------------------------------------------------------------
    model.eval()
    with torch.no_grad():
        preds = []
        for (xb, *_) in DataLoader(ds_te, batch_size=256):
            preds.append(model(xb.to(device))["difficulty"].cpu().numpy())
    dte = dte.copy()
    dte["pred"] = np.concatenate(preds)
    rep = report(dte, "pred", target)
    name = args.run_name or (
        f"{args.scheme}{'_' + args.fold if args.fold else ''}"
        f"_L{layer}_lang{weights['lang']}_pop{weights['pop']}_spk{weights['spk']}")
    print_report(rep, name)

    run_dir = out_root / "runs" / name
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), run_dir / "model.pt")
    (run_dir / "report.json").write_text(json.dumps(
        {"config": {"layer": layer, "weights": weights, "scheme": args.scheme,
                    "fold": args.fold, "target": target},
         "report": rep}, indent=2, default=float))
    print(f"[train] wrote {run_dir}")


if __name__ == "__main__":
    main()
