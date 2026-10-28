"""
Single-video inference + validation-derived operating point.
  python -m vad.infer fit --run tf_s1                       # writes runs/tf_s1/operating_point.json (VALIDATION only)
  python -m vad.infer analyze --run tf_s1 --video clip.mp4
"""
from __future__ import annotations
import argparse, json, tempfile
from pathlib import Path
import numpy as np
import torch

from vad import calibration as cal
from vad.config import load_config, paths
from vad.data.features import build_backbone, extract_video
from vad.localize import extract_intervals, intervals_to_seconds, fmt_time
from vad.metrics import pick_threshold_f1
from vad.models import pick_device
from vad.seq import segments_to_snippets
from vad.data.bags import pool
from vad.train import load_model, load_split_bags, predict_bags


def fit_operating_point(cfg, run, dev):
    P = paths(cfg)
    model, norm, mcfg, ck = load_model(P["runs"] / run / "best.pt", dev)
    c2 = dict(cfg); c2["model"] = ck["cfg"]["model"]; c2["preprocess"] = ck["cfg"]["preprocess"]
    val = load_split_bags(c2, "val", norm)
    vs = np.array([s.max() for s in predict_bags(model, val, dev)])
    thr, f1 = pick_threshold_f1(val.labels, vs)
    sc = cal.fit_scaler(vs, val.labels)
    op = dict(threshold=thr, val_f1=f1, scaler=sc, n_val=len(val), source="validation split only")
    json.dump(op, open(P["runs"] / run / "operating_point.json", "w"), indent=2)
    return op


class Analyzer:
    def __init__(self, cfg, run, dev=None):
        self.P = paths(cfg); self.dev = dev or pick_device(cfg["train"]["device"])
        self.run = run
        self.model, self.norm, _, self.ck = load_model(self.P["runs"] / run / "best.pt", self.dev)
        self.cfg = cfg; self.mcfg = self.ck["cfg"]["model"]; self.pc = self.ck["cfg"]["preprocess"]
        opf = self.P["runs"] / run / "operating_point.json"
        self.op = json.load(open(opf)) if opf.exists() else None
        self.backbone = None

    def _bb(self):
        if self.backbone is None:
            self.backbone, _ = build_backbone(self.pc["backbone"], self.pc["weights"], self.dev)
        return self.backbone

    def analyze(self, video_path, features_npy: Path | None = None):
        tmpd = tempfile.mkdtemp()
        out = Path(features_npy) if features_npy else Path(tmpd) / "f.npy"
        arr, meta, _ = extract_video(self._bb(), self.dev, str(video_path), self.pc, out, self.pc["batch_clips"])
        x = self.norm(pool(arr.astype(np.float32), self.mcfg))
        with torch.no_grad():
            seg = torch.sigmoid(self.model(torch.from_numpy(x[None]).float().to(self.dev))).cpu().numpy()[0]
        n = arr.shape[0]
        sn = segments_to_snippets(seg, n)
        stride, fps = self.pc["clip_stride"], meta["fps"]
        ec = self.cfg["eval"]
        iv = extract_intervals(sn, ec["smooth_window"], ec["rel_threshold"], ec["min_len_snippets"], ec["merge_gap_snippets"])
        vscore = float(seg.max())
        res = dict(video_score=vscore, snippet_scores=sn.tolist(), fps=fps, n_frames=meta["n_frames_read"], stride=stride,
                   intervals_snippets=iv, intervals_sec=intervals_to_seconds(iv, stride, fps),
                   intervals_fmt=[f"{fmt_time(a)}-{fmt_time(b)}" for a, b in intervals_to_seconds(iv, stride, fps)])
        if self.op:
            p = float(cal.apply_scaler(np.array([vscore]), self.op["scaler"])[0])
            res.update(threshold=self.op["threshold"], flagged=vscore >= self.op["threshold"], calibrated_p=p,
                       route=cal.route(p), note="single model: no seed-spread; p is calibrated on validation")
        else:
            res["note"] = "no operating point yet: run `python -m vad.infer fit --run <run>`"
        return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fit", "analyze"]); ap.add_argument("--run", required=True)
    ap.add_argument("--video"); ap.add_argument("--config", default="configs/default.yaml"); ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    if a.cmd == "fit":
        print(json.dumps(fit_operating_point(cfg, a.run, pick_device(cfg["train"]["device"])), indent=2))
    else:
        r = Analyzer(cfg, a.run).analyze(a.video); r.pop("snippet_scores")
        print(json.dumps(r, indent=2, default=float))


if __name__ == "__main__":
    main()
