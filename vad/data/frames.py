"""Access helpers for the frame-folder dataset (Kaggle 'odins0n/ucf-crime-dataset': every 10th frame, 64x64 PNG)."""
from __future__ import annotations
import gzip, json, os
from pathlib import Path
import cv2, numpy as np


def frames_root(cfg, index=None) -> Path:
    r = os.environ.get("VAD_FRAMES_ROOT") or cfg["data"].get("frames_root") or (index or {}).get("root")
    if not r:
        raise FileNotFoundError("frames root unknown: pass --root to vad.data.index_frames, or set data.frames_root / VAD_FRAMES_ROOT")
    return Path(r)


def load_index(path: Path) -> dict:
    with gzip.open(path, "rt") as f:
        return json.load(f)


def save_index(path: Path, root: str, videos: dict, every: int, orig_fps: float):
    with gzip.open(path, "wt") as f:
        json.dump(dict(root=root, every=every, orig_fps=orig_fps, videos=videos), f)


def frame_paths(index: dict, root: Path, vid: str):
    return [str(root / p) for p in index["videos"][vid]]


def read_bgr(path: str):
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    if im is None:
        raise IOError(f"unreadable image {path}")
    return im


_CACHE = {}


def get_pil_frames(cfg, P, vid, resolved_path, idxs):
    """PIL RGB frames at the given indices (extracted-frame index in frames mode, video frame index otherwise)."""
    from PIL import Image
    if cfg["data"].get("source") == "frames":
        if "i" not in _CACHE:
            ix = load_index(P["dataset"] / "frame_index.json.gz"); _CACHE["i"] = (ix, frames_root(cfg, ix))
        ix, root = _CACHE["i"]; ps = ix["videos"][vid]
        return [Image.open(root / ps[min(max(int(i), 0), len(ps) - 1)]).convert("RGB") for i in idxs]
    cap, out = cv2.VideoCapture(resolved_path), []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i)); ok, fr = cap.read()
        if ok: out.append(Image.fromarray(fr[:, :, ::-1]))
    cap.release()
    return out
