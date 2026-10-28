"""Torch-free sequence utilities: fixed-T resampling and the exact inverse mapping used for localisation."""
from __future__ import annotations
import numpy as np


def segment_bounds(n: int, T: int) -> np.ndarray:
    """For n >= T: T contiguous, non-empty segments covering snippets [0, n)."""
    return (np.arange(T + 1) * n) // T


def resample(feats: np.ndarray, T: int) -> np.ndarray:
    """[n, D] -> [T, D]. n>=T: mean-pool contiguous segments. n<T: repeat nearest snippets."""
    n = feats.shape[0]
    if n == 0:
        raise ValueError("empty feature array")
    if n < T:
        idx = (np.arange(T) * n) // T
        return feats[idx].astype(np.float32)
    b = segment_bounds(n, T)
    f = feats.astype(np.float32)
    csum = np.concatenate([np.zeros((1, f.shape[1]), np.float32), np.cumsum(f, axis=0, dtype=np.float64).astype(np.float32)])
    return (csum[b[1:]] - csum[b[:-1]]) / (b[1:] - b[:-1])[:, None]


def resample_variable(feats: np.ndarray, max_len: int) -> np.ndarray:
    return feats.astype(np.float32) if feats.shape[0] <= max_len else resample(feats, max_len)


def segments_to_snippets(scores: np.ndarray, n: int) -> np.ndarray:
    """Inverse of resample for scores: [T] -> [n] (n snippets)."""
    T = len(scores)
    if n == T:
        return np.asarray(scores, dtype=np.float64)
    if n > T:
        b = segment_bounds(n, T)
        k = np.clip(np.searchsorted(b, np.arange(n), side="right") - 1, 0, T - 1)
        return np.asarray(scores, dtype=np.float64)[k]
    idx = (np.arange(T) * n) // T
    out = np.zeros(n)
    for j in range(n):
        out[j] = np.mean(scores[idx == j])
    return out


def snippet_frame_ranges(n_snip: int, stride: int, n_frames: int):
    """Partition of [0, n_frames) into n_snip ranges [a, b): snippet j owns [j*stride, (j+1)*stride); last extended."""
    r = []
    for j in range(n_snip):
        a = j * stride
        b = n_frames if j == n_snip - 1 else (j + 1) * stride
        r.append((a, max(a, b)))
    return r


def snippet_times(n_snip: int, stride: int, clip_len: int, fps: float):
    """Start/end seconds of the frames actually covered by each snippet's clip."""
    s = np.arange(n_snip) * stride / fps
    return s, s + clip_len / fps
