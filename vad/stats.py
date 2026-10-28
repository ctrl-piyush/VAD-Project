"""Paired bootstrap over test videos + Holm-Bonferroni (torch-free)."""
from __future__ import annotations
import numpy as np


def bootstrap_ci(metric_fn, n_videos, B=1000, seed=0, alpha=0.05):
    """metric_fn(idx: int array of resampled video indices) -> float"""
    rng = np.random.RandomState(seed)
    vals = []
    for _ in range(B):
        v = metric_fn(rng.randint(0, n_videos, n_videos))
        if not np.isnan(v):
            vals.append(v)
    vals = np.array(vals)
    return dict(mean=float(vals.mean()), lo=float(np.quantile(vals, alpha / 2)),
                hi=float(np.quantile(vals, 1 - alpha / 2)), n_valid=len(vals))


def paired_bootstrap_diff(metric_a, metric_b, n_videos, B=1000, seed=0, alpha=0.05):
    """Same resampled videos for both models (paired). Returns CI of metric_a - metric_b."""
    rng = np.random.RandomState(seed)
    d = []
    for _ in range(B):
        idx = rng.randint(0, n_videos, n_videos)
        a, b = metric_a(idx), metric_b(idx)
        if not (np.isnan(a) or np.isnan(b)):
            d.append(a - b)
    d = np.array(d)
    lo, hi = float(np.quantile(d, alpha / 2)), float(np.quantile(d, 1 - alpha / 2))
    p = float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))  # two-sided bootstrap p (approx.)
    return dict(diff=float(d.mean()), lo=lo, hi=hi, p=max(p, 1.0 / (len(d) + 1)), excludes_zero=bool(lo > 0 or hi < 0))


def holm(pvals, alpha=0.05):
    """Holm-Bonferroni. Returns list of (adjusted_p, reject)."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.zeros(m); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * pvals[i]))
        adj[i] = run
    return [(float(a), bool(a < alpha)) for a in adj]
