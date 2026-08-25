"""
evaluate.py — Evaluation harness implementing plan §8.

Never pools across languages/populations without also reporting per-group:
report() returns overall AND per-language, per-population, per-corpus blocks.

Metrics:
  graded target : Spearman ρ, Pearson r, CCC, MAE
  repair events : AUROC, average precision, ECE (10-bin)
  known-groups  : monotonicity of mean prediction across dx severity (validation-only use of diagnosis)
  state-vs-trait: within-speaker share of prediction variance + within-speaker ρ
                  (a trait detector scores ~0 here — the plan's key discriminant)
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import ConstantInputWarning, pearsonr, spearmanr

# Constant-input correlations are expected for word-only languages (silver
# constant) and reported as NaN on purpose — don't spam the log about it.
warnings.filterwarnings("ignore", category=ConstantInputWarning)


def ccc(y: np.ndarray, p: np.ndarray) -> float:
    ym, pm = y.mean(), p.mean()
    yv, pv = y.var(), p.var()
    cov = ((y - ym) * (p - pm)).mean()
    d = yv + pv + (ym - pm) ** 2
    return float(2 * cov / d) if d > 0 else 0.0


def ece_binary(y: np.ndarray, prob: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    n, e = len(y), 0.0
    for i in range(bins):
        m = (prob >= edges[i]) & (prob < edges[i + 1] + (i == bins - 1))
        if m.sum():
            e += m.sum() / n * abs(y[m].mean() - prob[m].mean())
    return float(e)


def graded_metrics(y, p) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    ok = ~(np.isnan(y) | np.isnan(p))
    y, p = y[ok], p[ok]
    if len(y) < 3 or y.std() == 0 or p.std() == 0:
        return {"n": int(len(y)), "spearman": np.nan, "pearson": np.nan,
                "ccc": np.nan, "mae": np.nan}
    return {"n": int(len(y)),
            "spearman": float(spearmanr(y, p).statistic),
            "pearson": float(pearsonr(y, p).statistic),
            "ccc": ccc(y, p),
            "mae": float(np.abs(y - p).mean())}


def binary_metrics(y, prob) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score
    y, prob = np.asarray(y, float), np.asarray(prob, float)
    ok = ~(np.isnan(y) | np.isnan(prob))
    y, prob = y[ok], prob[ok]
    if len(np.unique(y)) < 2:
        return {"n": int(len(y)), "auroc": np.nan, "ap": np.nan, "ece": np.nan}
    return {"n": int(len(y)),
            "auroc": float(roc_auc_score(y, prob)),
            "ap": float(average_precision_score(y, prob)),
            "ece": ece_binary(y, prob)}


def known_groups(df: pd.DataFrame, pred_col: str,
                 severity_order: list[str] | None = None) -> dict:
    """Mean prediction should rise with dx severity (criterion validity).
    Diagnosis is used to VALIDATE, never to label (plan §8)."""
    g = df[df["group"].astype(str).str.len() > 0]
    if g.empty:
        return {"groups": {}, "monotonic_spearman": np.nan}
    means = g.groupby("group")[pred_col].mean().to_dict()
    result = {"groups": {k: float(v) for k, v in means.items()},
              "monotonic_spearman": np.nan}
    if severity_order:
        ordered = [means[k] for k in severity_order if k in means]
        if len(ordered) >= 3:
            result["monotonic_spearman"] = float(
                spearmanr(range(len(ordered)), ordered).statistic)
    return result


def state_vs_trait(df: pd.DataFrame, pred_col: str, target_col: str) -> dict:
    """Discriminant validity: does the estimator track WITHIN-speaker
    fluctuation, or is it a covert trait/diagnosis detector?"""
    d = df.dropna(subset=[pred_col, target_col])
    total_var = d[pred_col].var()
    if not total_var or np.isnan(total_var):
        return {"within_speaker_var_share": np.nan, "mean_within_speaker_rho": np.nan}
    grand = d.groupby("speaker")[pred_col].transform("mean")
    within_share = float((d[pred_col] - grand).var() / total_var)
    rhos = []
    for _, g in d.groupby("speaker"):
        if len(g) >= 8 and g[target_col].std() > 0 and g[pred_col].std() > 0:
            rhos.append(spearmanr(g[target_col], g[pred_col]).statistic)
    return {"within_speaker_var_share": within_share,
            "mean_within_speaker_rho": float(np.nanmean(rhos)) if rhos else np.nan,
            "n_speakers_scored": len(rhos)}


def report(df: pd.DataFrame, pred_col: str, target_col: str,
           repair_prob_col: str | None = None) -> dict:
    out = {"overall": graded_metrics(df[target_col], df[pred_col]),
           "state_vs_trait": state_vs_trait(df, pred_col, target_col),
           "known_groups": known_groups(df, pred_col)}
    for by in ("language", "population", "corpus"):
        out[f"per_{by}"] = {
            str(k): graded_metrics(g[target_col], g[pred_col])
            for k, g in df.groupby(by)}
    if repair_prob_col and "repair_event" in df:
        out["repair_detection"] = binary_metrics(df["repair_event"],
                                                 df[repair_prob_col])
    return out


def print_report(rep: dict, title: str = "") -> None:
    print(f"\n===== {title} =====")
    o = rep["overall"]
    print(f"overall  n={o['n']}  ρ={o['spearman']:.3f}  CCC={o['ccc']:.3f}  MAE={o['mae']:.3f}")
    for by in ("per_language", "per_population"):
        for k, m in rep[by].items():
            print(f"  {by[4:]:11}={k:12} n={m['n']:>6}  ρ={m['spearman'] if m['spearman']==m['spearman'] else float('nan'):.3f}  CCC={m['ccc'] if m['ccc']==m['ccc'] else float('nan'):.3f}")
    st = rep["state_vs_trait"]
    print(f"state-vs-trait: within-speaker var share={st['within_speaker_var_share']:.3f}, "
          f"mean within-speaker ρ={st['mean_within_speaker_rho']:.3f}")
    if "repair_detection" in rep:
        r = rep["repair_detection"]
        print(f"repair: AUROC={r['auroc']:.3f} AP={r['ap']:.3f} ECE={r['ece']:.3f}")
