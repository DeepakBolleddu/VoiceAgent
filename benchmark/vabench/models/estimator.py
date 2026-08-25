"""
estimator.py — Difficulty estimator with trait-invariance adversaries (P2 core).

Architecture (over cached SSL embeddings; end-to-end fine-tune is a later
ablation):
    z = MLP_enc(embedding)
    difficulty  = head_diff(z)                      <- regression target
    lang_logits = head_lang(GRL(z))                 <- adversary 1
    pop_logits  = head_pop(GRL(z))                  <- adversary 2
    spk_logits  = head_spk(GRL(z))                  <- adversary 3 (optional)

Loss = MSE(difficulty) + λ_lang·CE + λ_pop·CE + λ_spk·CE, with GRL reversing
adversary gradients into the encoder. λ ramps 0→weight via the DANN schedule.

Ablation grid (config train.adv_*_weight):
  all zero            -> frozen-probe baseline (the "not the contribution" model)
  lang only / pop only / both / +speaker  -> the invariance analysis (RQ2/RQ3)

The central tension (plan §7.1): strip too much and difficulty dies with it.
Diagnose via the eval harness's state-vs-trait and per-language blocks — an
invariance win must show BOTH held-out transfer gain AND preserved
within-speaker sensitivity.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.lambd * grad, None


def grl(x: torch.Tensor, lambd: float) -> torch.Tensor:
    return GradientReversal.apply(x, lambd)


def dann_lambda(progress: float, gamma: float = 10.0) -> float:
    """DANN schedule: 0 -> 1 as training progresses (progress in [0,1])."""
    import math
    return 2.0 / (1.0 + math.exp(-gamma * progress)) - 1.0


def mlp(dims: list[int], dropout: float = 0.2) -> nn.Sequential:
    layers: list[nn.Module] = []
    for a, b in zip(dims[:-1], dims[1:]):
        layers += [nn.Linear(a, b), nn.GELU(), nn.Dropout(dropout)]
    return nn.Sequential(*layers[:-2])  # no activation/dropout after last


class DifficultyEstimator(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256, z_dim: int = 128,
                 n_langs: int = 0, n_pops: int = 0, n_spks: int = 0):
        super().__init__()
        self.encoder = mlp([in_dim, hidden, z_dim])
        self.head_diff = nn.Linear(z_dim, 1)
        self.head_lang = nn.Linear(z_dim, n_langs) if n_langs > 1 else None
        self.head_pop = nn.Linear(z_dim, n_pops) if n_pops > 1 else None
        self.head_spk = nn.Linear(z_dim, n_spks) if n_spks > 1 else None

    def forward(self, x: torch.Tensor, lambd: float = 0.0) -> dict:
        z = self.encoder(x)
        out = {"z": z, "difficulty": self.head_diff(z).squeeze(-1)}
        for name, head in [("lang", self.head_lang), ("pop", self.head_pop),
                           ("spk", self.head_spk)]:
            if head is not None:
                out[name] = head(grl(z, lambd))
        return out


def loss_fn(out: dict, batch: dict, weights: dict) -> tuple[torch.Tensor, dict]:
    mse = nn.functional.mse_loss(out["difficulty"], batch["target"])
    total, parts = mse, {"mse": mse.item()}
    ce = nn.functional.cross_entropy
    for name in ("lang", "pop", "spk"):
        w = weights.get(name, 0.0)
        if w > 0 and name in out:
            l = ce(out[name], batch[name])
            total = total + w * l
            parts[f"adv_{name}"] = l.item()
    return total, parts
