"""
Carve a FIXED validation split out of TRAIN (video-level, stratified by class).
  python -m vad.data.make_val_split
Adds split='val' rows to manifest.csv and saves artifacts/dataset/split_manifest.json
(video ids + seed) so the split can never silently change. The test split is never touched.
"""
from __future__ import annotations
import argparse, json
import numpy as np
import pandas as pd
from vad.config import load_config, paths


def make_val_split(df: pd.DataFrame, frac: float, seed: int) -> pd.DataFrame:
    df = df.copy()
    df["split"] = df["split"].replace({"val": "train"})  # idempotent
    rng = np.random.RandomState(seed)
    train = df[(df.split == "train") & (df.status == "ok")]
    val_ids = []
    for cls, g in train.groupby("cls"):
        ids = sorted(g.video_id.tolist())
        k = max(1, int(round(len(ids) * frac))) if len(ids) >= 2 else 0
        val_ids += list(rng.choice(ids, size=k, replace=False))
    df.loc[df.video_id.isin(val_ids) & (df.split == "train"), "split"] = "val"
    return df


def check_no_overlap(df: pd.DataFrame):
    sets = {s: set(df[df.split == s].video_id) for s in ["train", "val", "test"]}
    for a in sets:
        for b in sets:
            if a < b:
                inter = sets[a] & sets[b]
                assert not inter, f"LEAKAGE: {a}/{b} share {len(inter)} videos, e.g. {sorted(inter)[:3]}"
    assert df.video_id.is_unique, "video_id must be unique across the manifest"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    P = paths(cfg)
    df = pd.read_csv(P["manifest"])
    out = make_val_split(df, cfg["data"]["val_fraction"], cfg["data"]["val_seed"])
    check_no_overlap(out)
    out.to_csv(P["manifest"], index=False)
    info = {s: sorted(out[out.split == s].video_id.tolist()) for s in ["train", "val", "test"]}
    info["seed"] = cfg["data"]["val_seed"]
    with open(P["dataset"] / "split_manifest.json", "w") as f:
        json.dump(info, f)
    print(out.groupby(["split", "label"]).size().unstack(fill_value=0))
    print("val per class:", out[out.split == "val"].cls.value_counts().to_dict())


if __name__ == "__main__":
    main()
