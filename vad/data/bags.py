"""Cached feature loading, TRAIN-only feature normalisation, fixed-T / masked variable-length bags, balanced sampling."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from vad.seq import resample, resample_variable


def load_raw(feat_dir: Path, vid: str):
    arr = np.load(feat_dir / f"{vid}.npy")
    meta = json.load(open(feat_dir / f"{vid}.json"))
    assert arr.shape[0] == meta["n_snippets"], f"{vid}: feature/meta mismatch"
    return arr, meta


class FeatureNorm:
    """mode: none | l2 | zscore.  zscore statistics are fit on TRAIN bags only and stored in the checkpoint."""
    def __init__(self, mode="l2", mean=None, std=None):
        self.mode, self.mean, self.std = mode, mean, std

    def fit(self, bags: np.ndarray):
        if self.mode == "zscore":
            flat = bags.reshape(-1, bags.shape[-1])
            self.mean, self.std = flat.mean(0), flat.std(0) + 1e-6
        return self

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self.mode == "l2":
            return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)
        if self.mode == "zscore":
            return (x - self.mean) / self.std
        return x

    def state(self):
        return dict(mode=self.mode, mean=None if self.mean is None else self.mean.tolist(),
                    std=None if self.std is None else self.std.tolist())

    @staticmethod
    def from_state(s):
        return FeatureNorm(s["mode"], None if s["mean"] is None else np.array(s["mean"], np.float32),
                           None if s["std"] is None else np.array(s["std"], np.float32))


def pool(raw: np.ndarray, mcfg: dict) -> np.ndarray:
    if mcfg["seq_mode"] == "variable":
        return resample_variable(raw, mcfg["max_len"])
    return resample(raw, mcfg["segments"])


class Bags:
    """In-memory bags for one split. Each item: pooled [t,D] (t=T fixed, or <=max_len if variable)."""
    def __init__(self, df: pd.DataFrame, feat_dir: Path, mcfg: dict, norm: FeatureNorm | None = None):
        self.df = df.reset_index(drop=True)
        self.mcfg = mcfg
        self.ids = self.df.video_id.tolist()
        self.labels = self.df.label.to_numpy().astype(int)
        self.cls = self.df.cls.tolist()
        self.metas, self.pooled = [], []
        for v in self.ids:
            raw, meta = load_raw(feat_dir, v)
            self.metas.append(meta)
            self.pooled.append(pool(raw, mcfg))
        self.norm = norm
        self.dim = self.pooled[0].shape[-1]

    def set_norm(self, norm: FeatureNorm):
        self.norm = norm

    def stack_for_fit(self):
        return np.concatenate([p for p in self.pooled])[None]

    def item(self, i):
        x = self.pooled[i]
        return self.norm(x) if self.norm else x

    def collate(self, idx):
        xs = [self.item(i) for i in idx]
        L = max(x.shape[0] for x in xs)
        X = np.zeros((len(xs), L, self.dim), np.float32)
        M = np.zeros((len(xs), L), bool)
        for k, x in enumerate(xs):
            X[k, :len(x)] = x
            M[k, :len(x)] = True
        return X, M, self.labels[list(idx)]

    def __len__(self):
        return len(self.ids)


def balanced_batch_indices(bags: Bags, batch_size: int, rng: np.random.RandomState, class_balanced=True):
    """Half anomalous (class-balanced if requested), half normal."""
    a_idx = np.where(bags.labels == 1)[0]
    n_idx = np.where(bags.labels == 0)[0]
    half = batch_size // 2
    if class_balanced:
        by_cls = {}
        for i in a_idx:
            by_cls.setdefault(bags.cls[i], []).append(i)
        classes = sorted(by_cls)
        a = [rng.choice(by_cls[classes[rng.randint(len(classes))]]) for _ in range(half)]
    else:
        a = rng.choice(a_idx, half, replace=len(a_idx) < half)
    n = rng.choice(n_idx, half, replace=len(n_idx) < half)
    return np.array(list(a) + list(n))
