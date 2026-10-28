"""Smoothing + thresholding of snippet scores into time intervals (torch-free)."""
from __future__ import annotations
import numpy as np


def smooth(scores, window: int) -> np.ndarray:
    s = np.asarray(scores, dtype=np.float64)
    if window <= 1 or len(s) < 2:
        return s
    w = min(window, len(s))
    pad = w // 2
    sp = np.pad(s, (pad, w - 1 - pad), mode="edge")
    return np.convolve(sp, np.ones(w) / w, mode="valid")


def extract_intervals(snip_scores, window=5, rel_threshold=0.5, min_len=1, merge_gap=2):
    """Returns list of (start_snip, end_snip_exclusive). Threshold = rel_threshold * max(smoothed)."""
    sm = smooth(snip_scores, window)
    if len(sm) == 0 or sm.max() <= 0:
        return []
    mask = sm >= rel_threshold * sm.max()
    iv, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        if not m and start is not None:
            iv.append([start, i]); start = None
    if start is not None:
        iv.append([start, len(mask)])
    merged = []
    for s, e in iv:
        if merged and s - merged[-1][1] <= merge_gap:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged if e - s >= min_len]


def intervals_to_frames(iv, stride: int, n_frames: int):
    return [(s * stride, min(n_frames, e * stride)) for s, e in iv]


def intervals_to_seconds(iv, stride: int, fps: float):
    return [(s * stride / fps, e * stride / fps) for s, e in iv]


def fmt_time(sec: float) -> str:
    m, s = divmod(int(round(sec)), 60)
    return f"{m:02d}:{s:02d}"
