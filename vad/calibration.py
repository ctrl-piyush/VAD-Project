"""Calibration of the video-level anomaly score. MIL ranking scores are NOT probabilities; we fit
temperature (and optional bias) on VALIDATION data only, then report ECE/Brier on test."""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize

EPS = 1e-6


def logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(z, dtype=np.float64), -60, 60)))


def fit_scaler(scores, labels, with_bias=False):
    """p = sigmoid((logit(score) - b)/T). Returns dict(T, b). with_bias=False is plain temperature scaling."""
    z, y = logit(scores), np.asarray(labels, dtype=np.float64)

    def nll(theta):
        T = np.exp(theta[0]); b = theta[1] if with_bias else 0.0
        p = np.clip(sigmoid((z - b) / T), EPS, 1 - EPS)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

    r = minimize(nll, x0=np.array([0.0, 0.0]), method="Nelder-Mead")
    return dict(T=float(np.exp(r.x[0])), b=float(r.x[1]) if with_bias else 0.0)


def apply_scaler(scores, sc):
    return sigmoid((logit(scores) - sc["b"]) / sc["T"])


def brier(p, y):
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def reliability(p, y, n_bins=15):
    p, y = np.asarray(p, dtype=float), np.asarray(y, dtype=float)
    edges = np.linspace(0, 1, n_bins + 1)
    ids = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = ids == b
        if m.any():
            rows.append(dict(bin=b, lo=edges[b], hi=edges[b + 1], n=int(m.sum()),
                             conf=float(p[m].mean()), acc=float(y[m].mean())))
    return rows


def ece(p, y, n_bins=15):
    rows = reliability(p, y, n_bins)
    n = sum(r["n"] for r in rows)
    return float(sum(r["n"] / n * abs(r["acc"] - r["conf"]) for r in rows)) if n else float("nan")


def risk_coverage(p, y):
    """Defer least-confident alerts first. confidence = max(p, 1-p); error = wrong binary decision at 0.5."""
    p, y = np.asarray(p, dtype=float), np.asarray(y)
    conf = np.maximum(p, 1 - p)
    wrong = ((p >= 0.5).astype(int) != y).astype(float)
    order = np.argsort(-conf)
    cum_err = np.cumsum(wrong[order]) / (np.arange(len(p)) + 1)
    cov = (np.arange(len(p)) + 1) / len(p)
    return cov, cum_err


def route(p, seed_std=None, lo=0.3, hi=0.7, std_max=0.15):
    """'alert' | 'normal' | 'human_review'. Borderline probability or high seed disagreement -> human review."""
    if seed_std is not None and seed_std > std_max:
        return "human_review"
    if p >= hi:
        return "alert"
    if p <= lo:
        return "normal"
    return "human_review"
