import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MILHead(nn.Module):
    """Baseline CNN+MIL (no self-attention): per-snippet MLP -> logit. (Sultani-style head.)"""
    def __init__(self, in_dim, hidden=128, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, 32), nn.Dropout(dropout), nn.Linear(32, 1))

    def forward(self, x, mask=None):                  # x [B,T,D] -> logits [B,T]
        return self.net(x).squeeze(-1)


class TransformerMIL(nn.Module):
    """Linear proj -> LN -> +learned positions -> Transformer encoder (pre-norm) -> MLP head. mask: True = valid."""
    def __init__(self, in_dim, d_model=256, n_layers=2, n_heads=4, ff_mult=2, dropout=0.1, head_hidden=128, max_len=256):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_dim, d_model), nn.LayerNorm(d_model), nn.Dropout(dropout))
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, d_model * ff_mult, dropout, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Linear(d_model, head_hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(head_hidden, 1))

    def forward(self, x, mask=None):                  # x [B,T,D], mask [B,T] bool -> logits [B,T]
        T = x.shape[1]
        h = self.proj(x) + self.pos[:, :T]
        kpm = None if mask is None else ~mask
        h = self.norm(self.enc(h, src_key_padding_mask=kpm))
        return self.head(h).squeeze(-1)


def build_model(mcfg: dict, in_dim: int):
    if mcfg["type"] == "baseline":
        return MILHead(in_dim, mcfg["head_hidden"], max(mcfg["dropout"], 0.1))
    max_len = max(mcfg["segments"], mcfg["max_len"])
    return TransformerMIL(in_dim, mcfg["d_model"], mcfg["n_layers"], mcfg["n_heads"], mcfg["ff_mult"],
                          mcfg["dropout"], mcfg["head_hidden"], max_len)


def mil_ranking_loss(logits, mask, labels, margin=1.0, lam_smooth=8e-5, lam_sparse=8e-5):
    """Sultani et al. ranking loss on bag maxima + temporal smoothness + sparsity on anomalous bags.
    logits/mask: [B,T]; labels: [B] (1 anomalous). Uses every anomalous/normal pair in the batch."""
    s = torch.sigmoid(logits)
    vmax = s.masked_fill(~mask, -1.0).max(1).values
    pos, neg = labels == 1, labels == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return logits.sum() * 0.0, {}
    rank = F.relu(margin - vmax[pos][:, None] + vmax[neg][None, :]).mean()
    sa, ma = s[pos], mask[pos]
    pair = (ma[:, 1:] & ma[:, :-1]).float()
    smooth = (((sa[:, 1:] - sa[:, :-1]) ** 2) * pair).sum(1).mean()
    sparse = (sa * ma.float()).sum(1).mean()
    loss = rank + lam_smooth * smooth + lam_sparse * sparse
    return loss, dict(rank=rank.item(), smooth=smooth.item(), sparse=sparse.item())


def video_scores(logits, mask):
    return torch.sigmoid(logits).masked_fill(~mask, -1.0).max(1).values


def pick_device(name="auto"):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
