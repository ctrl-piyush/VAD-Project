"""
Dataset-centric analysis (synopsis 4.8): imbalance, length distributions, feature shift (MMD, Frechet), optional UMAP.
  python -m vad.analysis [--umap]
Needs manifest.csv (+ cached features for the shift statistics). Writes artifacts/reports/dataset_analysis.json + png.
"""
from __future__ import annotations
import argparse, json, math
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import linalg
from vad.config import load_config, paths


def class_imbalance(df):
    out = {}
    for split, g in df.groupby("split"):
        c = Counter(g.cls); an = {k: v for k, v in c.items() if k != "Normal"}
        tot = sum(an.values())
        ent = -sum(v / tot * math.log2(v / tot) for v in an.values()) if tot else 0
        out[split] = dict(counts=dict(c), imbalance_ratio=(max(an.values()) / min(an.values())) if an else None,
                          entropy_bits=round(ent, 4), max_entropy_bits=round(math.log2(len(an)), 4) if len(an) > 1 else 0)
    return out


def mmd_rbf(X, Y, max_n=2000, seed=0):
    """Unbiased MMD^2 with RBF kernel, median heuristic bandwidth."""
    rng = np.random.RandomState(seed)
    X = X[rng.choice(len(X), min(max_n, len(X)), replace=False)].astype(np.float64)
    Y = Y[rng.choice(len(Y), min(max_n, len(Y)), replace=False)].astype(np.float64)
    Z = np.concatenate([X, Y])
    d2 = ((Z[:, None] - Z[None]) ** 2).sum(-1) if len(Z) <= 1500 else None
    if d2 is None:
        sub = Z[rng.choice(len(Z), 1500, replace=False)]
        d2s = ((sub[:, None] - sub[None]) ** 2).sum(-1); med = np.median(d2s[d2s > 0])
    else:
        med = np.median(d2[d2 > 0])
    gamma = 1.0 / (med + 1e-12)
    def k(A, B):
        a2, b2 = (A ** 2).sum(1)[:, None], (B ** 2).sum(1)[None]
        return np.exp(-gamma * np.maximum(a2 + b2 - 2 * A @ B.T, 0))
    Kxx, Kyy, Kxy = k(X, X), k(Y, Y), k(X, Y)
    n, m = len(X), len(Y)
    return float((Kxx.sum() - np.trace(Kxx)) / (n * (n - 1)) + (Kyy.sum() - np.trace(Kyy)) / (m * (m - 1)) - 2 * Kxy.mean())


def frechet(X, Y):
    mu1, mu2 = X.mean(0), Y.mean(0)
    s1, s2 = np.cov(X, rowvar=False), np.cov(Y, rowvar=False)
    cs, _ = linalg.sqrtm(s1.dot(s2), disp=False)
    cs = cs.real if np.iscomplexobj(cs) else cs
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1) + np.trace(s2) - 2 * np.trace(cs))


def sample_snippets(feat_dir, ids, per_video=20, seed=0):
    rng = np.random.RandomState(seed); out = []
    for v in ids:
        a = np.load(feat_dir / f"{v}.npy").astype(np.float32)
        out.append(a[rng.choice(len(a), min(per_video, len(a)), replace=False)])
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml"); ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--umap", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config, a.set); P = paths(cfg); P["reports"].mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(P["manifest"]); df = df[df.status == "ok"]
    rep = dict(imbalance=class_imbalance(df))
    rep["duration_s"] = {s: df[df.split == s].duration_s.describe().round(1).to_dict() for s in df.split.unique()}
    have = df[df.video_id.map(lambda v: (P["features"] / f"{v}.npy").exists())]
    if len(have):
        ns = have.video_id.map(lambda v: int(np.load(P["features"] / f"{v}.npy", mmap_mode="r").shape[0]))
        have = have.assign(n_snippets=ns)
        rep["snippets_per_video"] = {s: have[have.split == s].n_snippets.describe().round(1).to_dict() for s in have.split.unique()}
        tr, te = have[have.split == "train"], have[have.split == "test"]
        if len(tr) and len(te):
            Xtr, Xte = sample_snippets(P["features"], tr.video_id), sample_snippets(P["features"], te.video_id)
            rep["shift_train_vs_test"] = dict(mmd2=mmd_rbf(Xtr, Xte), frechet=frechet(Xtr, Xte))
            Xn = sample_snippets(P["features"], tr[tr.label == 0].video_id); Xa = sample_snippets(P["features"], tr[tr.label == 1].video_id)
            rep["shift_normal_vs_anomalous_train"] = dict(mmd2=mmd_rbf(Xn, Xa), frechet=frechet(Xn, Xa),
                                                          note="all snippets of anomalous videos (most are normal content)")
            if a.umap:
                try:
                    import umap, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
                    Z = np.concatenate([Xtr, Xte]); lab = np.array([0] * len(Xtr) + [1] * len(Xte))
                    E = umap.UMAP(random_state=0).fit_transform(Z)
                    plt.figure(figsize=(5, 5)); plt.scatter(E[:, 0], E[:, 1], c=lab, s=3, cmap="coolwarm")
                    plt.title("UMAP train(blue) vs test(red)"); plt.savefig(P["reports"] / "umap_train_test.png", dpi=120)
                except ImportError:
                    rep["umap"] = "umap-learn not installed"
    else:
        rep["feature_shift"] = "skipped: no cached features yet"
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        c = df[df.split == "train"].cls.value_counts(); ax[0].barh(c.index, c.values); ax[0].set_title("train videos per class")
        ax[1].hist(df.duration_s / 60, bins=40); ax[1].set(title="video duration (min)")
        fig.tight_layout(); fig.savefig(P["reports"] / "dataset_overview.png", dpi=120)
    except Exception as e:
        rep["plot_error"] = str(e)
    json.dump(rep, open(P["reports"] / "dataset_analysis.json", "w"), indent=2, default=float)
    print(json.dumps(rep, indent=2, default=float)[:3000])


if __name__ == "__main__":
    main()
