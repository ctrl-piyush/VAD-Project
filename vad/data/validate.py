"""
Dataset validation + manifest builder for UCF-Crime.

  python -m vad.data.validate                 # full validation
  python -m vad.data.validate --limit 20      # quick test on first 20 videos per split
  python -m vad.data.validate --hash          # also find byte-identical duplicates (slow)

Outputs (artifacts/dataset/): validation_report.json, manifest.csv (split = train|test)
Exit code 1 if blocking problems were found.
"""
from __future__ import annotations
import argparse, hashlib, json, math, re, sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm

from vad.config import load_config, paths

VIDEO_EXT = {".mp4", ".avi", ".mkv", ".mov"}


def class_from_name(name: str) -> str:
    """'RoadAccidents012_x264.mp4' -> 'RoadAccidents'; 'Normal_Videos_003_x264.mp4' -> 'Normal'."""
    m = re.match(r"[A-Za-z]+", Path(name).stem)
    return m.group(0) if m else "UNKNOWN"


def read_split(path: Path) -> list[str]:
    entries = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(line.split()[0].replace("\\", "/"))
    return entries


def index_videos(videos_dir: Path) -> dict:
    idx = defaultdict(list)
    for p in videos_dir.rglob("*"):
        if p.suffix.lower() in VIDEO_EXT:
            idx[p.name].append(p)
    return idx


def resolve(rel: str, videos_dir: Path, idx):
    direct = videos_dir / rel
    if direct.exists():
        return direct, "ok"
    cands = idx.get(Path(rel).name, [])
    if len(cands) == 1:
        return cands[0], "ok"
    if len(cands) > 1:
        return None, "ambiguous"
    return None, "missing"


def probe(path: Path) -> dict:
    out = dict(readable=False, n_frames=0, fps=0.0, width=0, height=0, duration_s=0.0, error="")
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            out["error"] = "cannot open"
            return out
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ok, _ = cap.read()
        if not ok:
            out["error"] = "first frame unreadable"
            return out
        if n > 10:  # seek near end to catch truncated files
            cap.set(cv2.CAP_PROP_POS_FRAMES, n - 5)
            ok2, _ = cap.read()
            if not ok2:
                out["error"] = "tail frames unreadable (possibly truncated)"
                return out
        if n <= 0 or fps <= 0:
            out["error"] = f"bad metadata n_frames={n} fps={fps}"
            return out
        out.update(readable=True, n_frames=n, fps=fps, width=w, height=h, duration_s=n / fps)
        return out
    finally:
        cap.release()


def parse_temporal(path: Path) -> dict:
    """Line format: <name> <class> s1 e1 s2 e2 ; -1 = absent. Verify against YOUR file."""
    ann = {}
    with open(path, "r") as f:
        for line in f:
            tok = line.split()
            if len(tok) < 4:
                continue
            name = tok[0]
            nums = [int(float(t)) for t in tok[2:]]
            segs = [(nums[i], nums[i + 1]) for i in range(0, len(nums) - 1, 2)
                    if nums[i] >= 0 and nums[i + 1] >= 0]
            ann[name] = segs
    return ann


def md5_file(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def class_stats(classes: list) -> dict:
    c = Counter(classes)
    anom = {k: v for k, v in c.items() if k != "Normal"}
    total = sum(anom.values())
    ent = -sum((v / total) * math.log2(v / total) for v in anom.values()) if total else 0.0
    return dict(counts=dict(c),
                imbalance_ratio_anomaly=(max(anom.values()) / min(anom.values())) if anom else None,
                anomaly_class_entropy_bits=round(ent, 4),
                max_entropy_bits=round(math.log2(len(anom)), 4) if len(anom) > 1 else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--set", nargs="*", default=[], help="config overrides key=value")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--hash", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    root = Path(cfg["data"]["root_abs"])
    vdir = root / cfg["data"]["videos_dir"]
    out_dir = paths(cfg)["dataset"]
    out_dir.mkdir(parents=True, exist_ok=True)

    errors, warnings = [], []
    for label, p in [("videos_dir", vdir), ("train_split", root / cfg["data"]["train_split"]),
                     ("test_split", root / cfg["data"]["test_split"])]:
        if not p.exists():
            errors.append(f"{label} not found: {p}")
    if errors:
        print(json.dumps({"errors": errors}, indent=2))
        sys.exit(1)

    idx = index_videos(vdir)
    known = set(cfg["data"]["anomaly_classes"]) | {"Normal"}
    splits = {"train": read_split(root / cfg["data"]["train_split"]),
              "test": read_split(root / cfg["data"]["test_split"])}

    rows, problems = [], defaultdict(list)
    for split, entries in splits.items():
        dups = [k for k, v in Counter(Path(e).name for e in entries).items() if v > 1]
        if dups:
            problems[f"duplicate_entries_in_{split}_list"] = dups
        use = entries[: args.limit] if args.limit else entries
        for rel in tqdm(use, desc=f"probe {split}"):
            path, status = resolve(rel, vdir, idx)
            cls = class_from_name(rel)
            row = dict(video_id=Path(rel).stem, rel_path=rel, split=split, cls=cls,
                       label=0 if cls == "Normal" else 1,
                       resolved_path=str(path) if path else "", status=status)
            if cls not in known:
                problems["unknown_class"].append(rel)
            if path is None:
                problems[status].append(rel)
                row.update(readable=False, n_frames=0, fps=0.0, width=0, height=0, duration_s=0.0, error=status)
            else:
                pr = probe(path)
                row.update(pr)
                if not pr["readable"]:
                    problems["unreadable"].append(f"{rel}: {pr['error']}")
            rows.append(row)

    tr = {Path(e).name for e in splits["train"]}
    te = {Path(e).name for e in splits["test"]}
    overlap = sorted(tr & te)
    if overlap:
        problems["TRAIN_TEST_OVERLAP_by_filename"] = overlap

    if args.hash:
        by_hash = defaultdict(list)
        for r in tqdm([r for r in rows if r["resolved_path"]], desc="hashing"):
            by_hash[md5_file(Path(r["resolved_path"]))].append((r["split"], r["rel_path"]))
        dups = [v for v in by_hash.values() if len(v) > 1]
        if dups:
            problems["identical_content_duplicates"] = dups
            cross = [g for g in dups if len({s for s, _ in g}) > 1]
            if cross:
                problems["CROSS_SPLIT_CONTENT_DUPLICATES"] = cross

    temporal = {"found": False}
    tpath = root / cfg["data"]["temporal_annotations"]
    if tpath.exists():
        ann = parse_temporal(tpath)
        temporal.update(found=True, n_annotated=len(ann))
        miss, bad = [], []
        for r in [r for r in rows if r["split"] == "test"]:
            name = Path(r["rel_path"]).name
            segs = ann.get(name)
            if segs is None:
                segs = ann.get(Path(name).stem)
            if r["label"] == 1 and segs is None:
                miss.append(name)
            if segs and r["n_frames"]:
                for s, e in segs:
                    if not (0 <= s < e <= r["n_frames"] + 1):
                        bad.append(f"{name}: ({s},{e}) vs n_frames={r['n_frames']}")
        temporal["anomalous_test_videos_without_annotation"] = miss
        temporal["segments_outside_video_bounds"] = bad
        if miss: problems["test_anomaly_missing_temporal_annotation"] = miss
        if bad: problems["temporal_segment_out_of_bounds"] = bad
    else:
        warnings.append(f"Temporal annotation file not found ({tpath}); frame-level AUC / temporal IoU unavailable.")

    stats = {}
    for split in splits:
        sr = [r for r in rows if r["split"] == split]
        dur = sorted(r["duration_s"] for r in sr if r["readable"])
        stats[split] = dict(n_listed=len(sr), n_readable=len(dur), **class_stats([r["cls"] for r in sr]))
        if dur:
            stats[split]["duration_s"] = dict(min=round(dur[0], 1), median=round(dur[len(dur) // 2], 1),
                                              max=round(dur[-1], 1), total_hours=round(sum(dur) / 3600, 2))
    if stats["train"]["counts"] and set(stats["train"]["counts"]) - {"Normal"} != set(cfg["data"]["anomaly_classes"]) and not args.limit:
        warnings.append("Anomaly classes found in train list differ from the 13 expected classes.")

    blocking = [k for k in problems if k in ("missing", "ambiguous", "unreadable", "unknown_class",
                "TRAIN_TEST_OVERLAP_by_filename", "CROSS_SPLIT_CONTENT_DUPLICATES")]
    report = dict(videos_dir=str(vdir), n_files_indexed=sum(len(v) for v in idx.values()),
                  problems={k: v for k, v in problems.items()}, temporal_annotations=temporal,
                  stats=stats, warnings=warnings, blocking_problem_types=blocking,
                  note="--limit used: partial run" if args.limit else "full run")
    suffix = "_partial" if args.limit else ""      # a --limit run must NEVER overwrite the real manifest / val split
    with open(out_dir / f"validation_report{suffix}.json", "w") as f:
        json.dump(report, f, indent=2)
    pd.DataFrame(rows).to_csv(out_dir / f"manifest{suffix}.csv", index=False)
    summary = {k: (len(v) if isinstance(v, list) else v) for k, v in problems.items()}
    print(json.dumps(dict(problem_counts=summary, stats=stats, warnings=warnings), indent=2))
    print(f"\nReport: {out_dir/f'validation_report{suffix}.json'}\nManifest: {out_dir/f'manifest{suffix}.csv'}")
    sys.exit(1 if blocking else 0)


if __name__ == "__main__":
    main()
