"""
Index the Kaggle frame dataset (odins0n/ucf-crime-dataset) into the SAME manifest format as vad.data.validate.

  python -m vad.data.index_frames --root /path/to/dataset_folder          # folder that contains Train/ and Test/
  python -m vad.data.index_frames --root ... --limit 5                    # inspect only, writes *_partial files

Assumes: <root>/**/{Train|Test}/<ClassFolder>/<VideoName>_<frameIndex>.png  (verify with the printed examples!)
Writes artifacts/dataset/{manifest.csv, validation_report.json, frame_index.json.gz}
"""
from __future__ import annotations
import argparse, json, os, re, sys
from collections import Counter, defaultdict
from pathlib import Path
import cv2, numpy as np, pandas as pd

from vad.config import load_config, paths
from vad.data import frames as F
from vad.data.validate import class_from_name

IMG = {".png", ".jpg", ".jpeg"}
EXPECTED = dict(train_videos=1610, train_anomalous=810, train_normal=800, test_videos=290, test_anomalous=140, test_normal=150)  # official split


def canon_class(folder: str, known: set) -> str:
    k = re.sub(r"[^a-z]", "", folder.lower())
    if k.startswith("normal"):
        return "Normal"
    for c in known:
        if re.sub(r"[^a-z]", "", c.lower()) == k:
            return c
    return folder


def find_split_dirs(root: Path):
    found = {}
    for dp, dns, _ in os.walk(root):
        depth = len(Path(dp).relative_to(root).parts)
        for d in dns:
            if d.lower() in ("train", "test") and d.lower() not in found:
                found[d.lower()] = Path(dp) / d
        if depth >= 3:
            dns[:] = []
    return found


def parse_name(stem: str):
    m = re.match(r"^(.*?)[_\-]?(\d+)$", stem)
    if not m:
        return None
    vid = m.group(1).rstrip("_-")
    return (vid, int(m.group(2))) if vid else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/kaggle_frames.yaml"); ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--root", required=True); ap.add_argument("--every", type=int, default=10); ap.add_argument("--orig-fps", type=float, default=30.0)
    ap.add_argument("--limit", type=int, default=0, help="only first N videos per split (inspection)")
    a = ap.parse_args()
    cfg = load_config(a.config, a.set); P = paths(cfg); P["dataset"].mkdir(parents=True, exist_ok=True)
    root = Path(a.root)
    splits = find_split_dirs(root)
    if set(splits) != {"train", "test"}:
        sys.exit(f"Could not find Train/ and Test/ folders under {root}. Found: {list(splits)}. Run: ls {root}  and tell me what you see.")
    known = set(cfg["data"]["anomaly_classes"]) | {"Normal"}
    rows, problems, index, examples = [], defaultdict(list), {}, []
    per_video_files = {}
    for split, sdir in splits.items():
        vids = defaultdict(list)
        for cdir in sorted(p for p in sdir.iterdir() if p.is_dir()):
            cls_folder = canon_class(cdir.name, known)
            for fp in cdir.rglob("*"):
                if fp.suffix.lower() not in IMG:
                    continue
                parsed = parse_name(fp.stem)
                if len(examples) < 5: examples.append(str(fp.relative_to(root)))
                if parsed is None:
                    problems["unparsable_filename"].append(str(fp.relative_to(root))); continue
                vids[(cls_folder, parsed[0])].append((parsed[1], str(fp.relative_to(root))))
        for k, (key, items) in enumerate(sorted(vids.items())):
            if a.limit and k >= a.limit: break
            cls, vid = key
            items.sort()
            idxs = np.array([i for i, _ in items])
            gaps = np.diff(idxs)
            if len(gaps) and np.unique(gaps).size > 1:
                problems["irregular_frame_gaps"].append(f"{vid}: gaps {np.unique(gaps)[:5].tolist()}")
            if len(idxs) != len(set(idxs.tolist())):
                problems["duplicate_frame_index"].append(vid)
            name_cls = class_from_name(vid)
            if name_cls != cls and not (cls == "Normal" and name_cls == "Normal"):
                problems["class_folder_vs_filename_mismatch"].append(f"{vid}: folder={cls}")
            index[vid] = [p for _, p in items]
            rows.append(dict(video_id=vid, rel_path=f"{split}/{cls}/{vid}", split=split, cls=cls, label=0 if cls == "Normal" else 1,
                             resolved_path="", status="ok", readable=True, n_frames=len(items) * a.every, fps=a.orig_fps, width=64, height=64,
                             duration_s=len(items) * a.every / a.orig_fps, n_extracted=len(items), error=""))
    df = pd.DataFrame(rows)
    tr, te = set(df[df.split == "train"].video_id), set(df[df.split == "test"].video_id)
    if tr & te: problems["TRAIN_TEST_OVERLAP"] = sorted(tr & te)
    rng = np.random.RandomState(0); bad = []
    for vid in rng.choice(list(index), min(30, len(index)), replace=False):
        p = root / index[vid][len(index[vid]) // 2]
        im = cv2.imread(str(p))
        if im is None: bad.append(str(p))
    if bad: problems["unreadable_sample_images"] = bad
    g = df.groupby("split")
    counts = {s: dict(videos=int(len(x)), anomalous=int((x.label == 1).sum()), normal=int((x.label == 0).sum()), images=int(x.n_extracted.sum())) for s, x in g}
    warn = []
    if not a.limit:
        for s, e in [("train", ("train_videos", "train_anomalous", "train_normal")), ("test", ("test_videos", "test_anomalous", "test_normal"))]:
            got = (counts[s]["videos"], counts[s]["anomalous"], counts[s]["normal"]); exp = tuple(EXPECTED[k] for k in e)
            if got != exp: warn.append(f"{s}: found (videos, anomalous, normal)={got}, official UCF-Crime split is {exp}. If it differs, this copy may not use the official split: ask me before training.")
    blocking = [k for k in problems if k in ("unparsable_filename", "TRAIN_TEST_OVERLAP", "unreadable_sample_images", "duplicate_frame_index")]
    suffix = "_partial" if a.limit else ""
    rep = dict(root=str(root), filename_examples=examples, counts=counts, class_counts={s: x.cls.value_counts().to_dict() for s, x in g},
               problems={k: v[:50] for k, v in problems.items()}, problem_counts={k: len(v) for k, v in problems.items()}, warnings=warn,
               blocking_problem_types=blocking, assumptions=f"every={a.every}, orig_fps={a.orig_fps} (UCF-Crime videos are ~30 fps; effective fps={a.orig_fps/a.every})")
    json.dump(rep, open(P["dataset"] / f"validation_report{suffix}.json", "w"), indent=2)
    df.to_csv(P["dataset"] / f"manifest{suffix}.csv", index=False)
    if not a.limit:
        F.save_index(P["dataset"] / "frame_index.json.gz", str(root), index, a.every, a.orig_fps)
    print(json.dumps({k: rep[k] for k in ("filename_examples", "counts", "problem_counts", "warnings", "assumptions")}, indent=2))
    sys.exit(1 if blocking else 0)


if __name__ == "__main__":
    main()
