"""
Non-convolutional frame embeddings (CLIP ViT) for the ablation "Transformer+MIL" (synopsis 4.9).
One 512-d vector per 16-frame snippet = mean of CLIP embeddings of 3 frames (start/middle/end of the snippet).
Same snippet grid / metadata as the 3D-CNN features, stored in artifacts/features/clip_vitb32/.

  python -m vad.data.clip_features --limit 10
  python -m vad.data.clip_features
Needs: pip install transformers pillow   (downloads openai/clip-vit-base-patch32 on first run)
"""
from __future__ import annotations
import argparse, json, os, time
from pathlib import Path
import cv2, numpy as np, pandas as pd
from tqdm import tqdm
from vad.config import load_config, paths
from vad.data.features import is_done

BACKBONE = "clip_vitb32"


def snippet_frames(path, clip_len, stride, offsets=None):
    """Returns (list of RGB uint8 frames, per-snippet index list, fps, n_frames_read). Only needed frames are decoded."""
    offsets = offsets or [0, clip_len // 2, clip_len - 1]
    cap = cv2.VideoCapture(path)
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    need = {}
    # we do not know n_frames in advance reliably -> decode sequentially, keep a rolling dict of wanted frames
    wanted = lambda fr: [(j, o) for o in offsets for j in [(fr - o) // stride] if j >= 0 and (fr - o) % stride == 0]
    n_read, last = 0, None
    while True:
        ok = cap.grab()
        if not ok:
            break
        w = wanted(n_read)
        if w:
            ok, img = cap.retrieve()
            if not ok:
                break
            rgb = np.ascontiguousarray(img[:, :, ::-1])
            for j, o in w:
                need[(j, o)] = rgb
        n_read += 1
    cap.release()
    if n_read == 0:
        raise RuntimeError("no frames")
    n_snip = max(1, (n_read - clip_len) // stride + 1) if n_read >= clip_len else 1
    per = []
    for j in range(n_snip):
        trio = [need.get((j, o)) for o in offsets]
        fallback = next(x for x in trio if x is not None) if any(x is not None for x in trio) else next(iter(need.values()))
        per.append([x if x is not None else fallback for x in trio])
    return per, fps, n_read


def snippet_frames_from_paths(paths, clip_len, stride, offsets=None):
    """Frame-folder version: returns (per-snippet list of RGB frames, n_frames)."""
    from vad.data.frames import read_bgr
    offsets = offsets or [0, clip_len // 2, clip_len - 1]
    n = len(paths)
    n_snip = max(1, (n - clip_len) // stride + 1) if n >= clip_len else 1
    cache, per = {}, []
    for j in range(n_snip):
        trio = []
        for o in offsets:
            k = min(j * stride + o, n - 1)
            if k not in cache:
                cache[k] = np.ascontiguousarray(read_bgr(paths[k])[:, :, ::-1])
            trio.append(cache[k])
        per.append(trio)
    return per, n


def extract_clip_video(embed_fn, path, clip_len=16, stride=16, batch=96, frame_paths=None, fps=3.0):
    if frame_paths is not None:
        per, n_read = snippet_frames_from_paths(frame_paths, clip_len, stride)
    else:
        per, fps, n_read = snippet_frames(path, clip_len, stride)
    flat = [fr for trio in per for fr in trio]
    embs = np.concatenate([embed_fn(flat[i:i + batch]) for i in range(0, len(flat), batch)])
    embs = embs.reshape(len(per), len(per[0]), -1).mean(1)
    return embs.astype(np.float16), dict(fps=fps, n_frames_read=n_read)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml"); ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    cfg = load_config(a.config, a.set); cfg["preprocess"]["backbone"] = BACKBONE
    pc, P = cfg["preprocess"], paths(cfg)
    import torch
    from PIL import Image
    from vad.explain.models import Clip
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip = Clip(cfg["explain"]["clip_model"], dev)
    embed = lambda frames: clip.image([Image.fromarray(f) for f in frames])
    df = pd.read_csv(P["manifest"]); df = df[(df.status == "ok") & (df.readable == True)]
    if a.limit:
        df = pd.concat([df[df.label == 1].head(a.limit // 2 + 1), df[df.label == 0].head(a.limit // 2 + 1)]).head(a.limit)
    P["features"].mkdir(parents=True, exist_ok=True)
    fidx = froot = None
    if cfg["data"].get("source") == "frames":
        from vad.data import frames as FR
        fidx = FR.load_index(P["dataset"] / "frame_index.json.gz"); froot = FR.frames_root(cfg, fidx)
    done, skipped, failed, t0 = 0, 0, [], time.time()
    for r in tqdm(list(df.itertuples()), desc="clip-extract"):
        npy, js = P["features"] / f"{r.video_id}.npy", P["features"] / f"{r.video_id}.json"
        if is_done(npy, js):
            skipped += 1; continue
        try:
            if fidx is not None:
                arr, meta = extract_clip_video(embed, None, pc["clip_len"], pc["clip_stride"], frame_paths=FR.frame_paths(fidx, froot, r.video_id), fps=fidx["orig_fps"] / fidx["every"])
            else:
                arr, meta = extract_clip_video(embed, r.resolved_path, pc["clip_len"], pc["clip_stride"])
            tmp = npy.with_suffix(".tmp.npy"); np.save(tmp, arr); os.replace(tmp, npy)
            json.dump(dict(video_id=r.video_id, n_snippets=int(arr.shape[0]), dim=int(arr.shape[1]), fps=meta["fps"],
                           n_frames_read=meta["n_frames_read"], clip_len=pc["clip_len"], stride=pc["clip_stride"],
                           backbone=BACKBONE, weights="clip"), open(js, "w"))
            done += 1
        except Exception as e:
            failed.append(dict(video_id=r.video_id, error=repr(e)))
    rep = dict(processed=done, skipped_already_done=skipped, failed=failed, seconds=round(time.time() - t0, 1))
    P["reports"].mkdir(parents=True, exist_ok=True)
    json.dump(rep, open(P["reports"] / "clip_extraction_report.json", "w"), indent=2); print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
