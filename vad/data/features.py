"""
Resumable 3D-CNN feature extraction (frozen backbone, cached to disk).

  python -m vad.data.features --limit 10          # smoke test on 10 videos (do this first!)
  python -m vad.data.features                     # all videos in manifest (resumes automatically)

Per video -> artifacts/features/<backbone>/<video_id>.npy  [n_snippets, D] float16
           + <video_id>.json  (fps, n_frames_read, clip_len, stride, n_snippets, backbone, weights ...)
Snippet j covers frames [j*stride, j*stride+clip_len).  Videos with <clip_len frames get 1 padded snippet.
"""
from __future__ import annotations
import argparse, json, os, time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from vad.config import load_config, paths

MEAN = np.array([0.43216, 0.394666, 0.37645], dtype=np.float32)   # Kinetics-400 (torchvision video models)
STD = np.array([0.22803, 0.22145, 0.216989], dtype=np.float32)


def build_backbone(name: str, weights: str, device):
    import torchvision.models.video as tv
    ctor = {"r3d_18": tv.r3d_18, "r2plus1d_18": tv.r2plus1d_18, "mc3_18": tv.mc3_18}[name]
    wenum = {"r3d_18": "R3D_18_Weights", "r2plus1d_18": "R2Plus1D_18_Weights", "mc3_18": "MC3_18_Weights"}[name]
    w = getattr(tv, wenum).KINETICS400_V1 if weights == "kinetics400" else None
    m = ctor(weights=w)
    dim = m.fc.in_features
    m.fc = torch.nn.Identity()
    return m.eval().to(device), dim


def preprocess_frame(frame_bgr, resize: int, crop: int) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    s = resize / min(h, w)
    fr = cv2.resize(frame_bgr, (max(crop, int(round(w * s))), max(crop, int(round(h * s)))), interpolation=cv2.INTER_AREA)
    h, w = fr.shape[:2]
    y0, x0 = (h - crop) // 2, (w - crop) // 2
    fr = fr[y0:y0 + crop, x0:x0 + crop, ::-1].astype(np.float32) / 255.0   # BGR->RGB
    return (fr - MEAN) / STD                                                # [crop,crop,3]


def _clips_from_iter(frames_iter, clip_len, stride, max_snippets=0):
    """frames_iter yields preprocessed frames [H,W,3]. Yields (clips, None) chunks then (rest, dict(n_frames_read))."""
    clips, n_read, start_next, frames = [], 0, 0, {}
    done = False
    for fr in frames_iter:
        frames[n_read] = fr
        n_read += 1
        while start_next + clip_len <= n_read:
            clip = np.stack([frames[start_next + k] for k in range(clip_len)])        # [L,H,W,3]
            clips.append(np.ascontiguousarray(clip.transpose(3, 0, 1, 2)))             # [3,L,H,W]
            start_next += stride
            for k in [k for k in frames if k < start_next]:
                del frames[k]
            if max_snippets and len(clips) >= max_snippets:
                done = True; break
        if done:
            break
        if len(clips) >= 64:                                                            # flush in chunks
            yield clips, None
            clips = []
    if n_read > 0 and start_next == 0 and not clips:                                    # video shorter than clip_len
        pad = [frames[min(k, n_read - 1)] for k in range(clip_len)]
        clips.append(np.ascontiguousarray(np.stack(pad).transpose(3, 0, 1, 2)))
    yield clips, dict(n_frames_read=n_read)


def video_frame_iter(path, resize, crop, frame_every=1, pre_resize=0, meta=None):
    cap = cv2.VideoCapture(path)
    if meta is not None:
        meta["fps"] = (float(cap.get(cv2.CAP_PROP_FPS)) or 30.0) / frame_every
    i = 0
    try:
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            if i % frame_every == 0:
                if pre_resize:
                    fr = cv2.resize(fr, (pre_resize, pre_resize), interpolation=cv2.INTER_AREA)
                yield preprocess_frame(fr, resize, crop)
            i += 1
    finally:
        cap.release()


def image_frame_iter(paths, resize, crop):
    from vad.data.frames import read_bgr
    for p in paths:
        yield preprocess_frame(read_bgr(p), resize, crop)


def clips_from_video(path, clip_len, stride, resize, crop, max_snippets=0, frame_every=1, pre_resize=0):
    meta = {}
    for clips, m in _clips_from_iter(video_frame_iter(path, resize, crop, frame_every, pre_resize, meta), clip_len, stride, max_snippets):
        yield clips, (dict(m, fps=meta["fps"]) if m else None)


def clips_from_frames(paths, clip_len, stride, resize, crop, max_snippets=0, fps=3.0):
    for clips, m in _clips_from_iter(image_frame_iter(paths, resize, crop), clip_len, stride, max_snippets):
        yield clips, (dict(m, fps=fps) if m else None)


@torch.no_grad()
def embed(model, clips, device, batch, fp16):
    out, i = [], 0
    while i < len(clips):
        b = clips[i:i + batch]
        x = torch.from_numpy(np.stack(b)).to(device)
        try:
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=fp16 and device.type == "cuda"):
                f = model(x)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if batch == 1:
                raise
            batch = max(1, batch // 2)
            continue
        out.append(f.float().cpu().numpy())
        i += len(b)
    return np.concatenate(out) if out else np.zeros((0, 0), np.float32), batch


def extract_video(model, device, vpath, pc, out_npy: Path, batch, frame_paths=None, frames_fps=3.0):
    """vpath: video file, or None when frame_paths (list of image files, already subsampled) is given."""
    feats, meta = [], None
    src = (clips_from_frames(frame_paths, pc["clip_len"], pc["clip_stride"], pc["resize"], pc["crop"], pc["max_snippets"], frames_fps)
           if frame_paths is not None else
           clips_from_video(vpath, pc["clip_len"], pc["clip_stride"], pc["resize"], pc["crop"], pc["max_snippets"],
                            pc.get("frame_every", 1), pc.get("pre_resize", 0)))
    for clips, m in src:
        if len(clips):
            f, batch = embed(model, clips, device, batch, pc["fp16"])
            feats.append(f)
        if m:
            meta = m
    if not feats or meta is None:
        raise RuntimeError("no clips produced")
    arr = np.concatenate(feats).astype(np.float16)
    if not np.isfinite(arr.astype(np.float32)).all():
        raise RuntimeError("non-finite features")
    tmp = out_npy.with_suffix(".tmp.npy")
    np.save(tmp, arr)
    os.replace(tmp, out_npy)                                    # atomic: a crash never leaves a half file
    return arr, meta, batch


def is_done(npy: Path, js: Path) -> bool:
    if not (npy.exists() and js.exists()):
        return False
    try:
        m = json.load(open(js))
        return np.load(npy, mmap_mode="r").shape[0] == m["n_snippets"]
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--limit", type=int, default=0, help="only first N videos (smoke test)")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    pc, P = cfg["preprocess"], paths(cfg)
    if pc["weights"] == "none":
        print("WARNING: weights=none -> random backbone. Smoke test only; features are NOT meaningful.")
    df = pd.read_csv(P["manifest"])
    df = df[(df.status == "ok") & (df.readable == True)]
    fidx, froot = None, None
    if cfg["data"].get("source") == "frames":
        from vad.data import frames as FR
        fidx = FR.load_index(P["dataset"] / "frame_index.json.gz"); froot = FR.frames_root(cfg, fidx)
        print(f"frames mode: root={froot}  every={fidx['every']}  effective fps={fidx['orig_fps']/fidx['every']}")
    if args.limit:
        # balanced small sample: alternate normal / anomalous
        a, n = df[df.label == 1].head(args.limit // 2 + 1), df[df.label == 0].head(args.limit // 2 + 1)
        df = pd.concat([a, n]).head(args.limit)
    P["features"].mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else
                       ("cpu" if args.device == "auto" else args.device))
    model, dim = build_backbone(pc["backbone"], pc["weights"], dev)
    batch, failed, t0, done, skipped = pc["batch_clips"], [], time.time(), 0, 0
    for r in tqdm(list(df.itertuples()), desc="extract"):
        npy, js = P["features"] / f"{r.video_id}.npy", P["features"] / f"{r.video_id}.json"
        if is_done(npy, js):
            skipped += 1
            continue
        try:
            if fidx is not None:
                arr, meta, batch = extract_video(model, dev, None, pc, npy, batch, FR.frame_paths(fidx, froot, r.video_id), fidx["orig_fps"] / fidx["every"])
            else:
                arr, meta, batch = extract_video(model, dev, r.resolved_path, pc, npy, batch)
            json.dump(dict(video_id=r.video_id, n_snippets=int(arr.shape[0]), dim=int(arr.shape[1]), fps=meta["fps"],
                           n_frames_read=meta["n_frames_read"], clip_len=pc["clip_len"], stride=pc["clip_stride"],
                           backbone=pc["backbone"], weights=pc["weights"], resize=pc["resize"], crop=pc["crop"],
                           source=cfg["data"].get("source", "video")), open(js, "w"))
            done += 1
        except Exception as e:
            failed.append(dict(video_id=r.video_id, error=repr(e)))
    rep = dict(processed=done, skipped_already_done=skipped, failed=failed, seconds=round(time.time() - t0, 1),
               seconds_per_new_video=round((time.time() - t0) / max(done, 1), 2), device=str(dev), dim=dim)
    P["reports"].mkdir(parents=True, exist_ok=True)
    json.dump(rep, open(P["reports"] / "extraction_report.json", "w"), indent=2)
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
