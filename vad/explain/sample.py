"""
Generate explanations for flagged TEST snippets and export a CSV for HUMAN labelling (synopsis 4.6, ~150 explanations).
  python -m vad.explain.sample --run tf_s1 --n 50            # 50 snippets x 3 variants = 150 rows (needs BLIP-2)
  python -m vad.explain.sample --run tf_s1 --n 150 --no-vlm  # retrieval-only variant (light)
Then two annotators fill columns annotator1 / annotator2 (1 = explanation is supported by the frame, 0 = not) in the CSV
(open frames/<video_id>.jpg next to each row), and you run:  python -m vad.explain.evaluate_checker --csv <file>
"""
from __future__ import annotations
import argparse, json
import cv2, numpy as np, pandas as pd
from PIL import Image
from tqdm import tqdm
from vad.config import load_config, paths
from vad.data.frames import get_pil_frames
from vad.explain.pipeline import Explainer
from vad.explain.rag import Store
from vad.models import pick_device
from vad.seq import segments_to_snippets
from vad.train import load_model, load_split_bags, predict_bags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml"); ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--run", required=True); ap.add_argument("--n", type=int, default=50); ap.add_argument("--no-vlm", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config, a.set); P = paths(cfg); dev = pick_device(cfg["train"]["device"])
    model, norm, mcfg, ck = load_model(P["runs"] / a.run / "best.pt", dev)
    c2 = dict(cfg); c2["model"] = ck["cfg"]["model"]; c2["preprocess"] = ck["cfg"]["preprocess"]
    bags = load_split_bags(c2, "test", norm); segs = predict_bags(model, bags, dev)
    op = json.load(open(P["runs"] / a.run / "operating_point.json"))
    flagged = [i for i in range(len(bags)) if segs[i].max() >= op["threshold"]]
    rng = np.random.RandomState(0); pick = sorted(rng.choice(flagged, min(a.n, len(flagged)), replace=False))
    ex = Explainer(cfg, dev, Store.load(P["artifacts"] / "rag_store"), use_vlm=not a.no_vlm)
    out_dir = P["artifacts"] / "explanations"; (out_dir / "frames").mkdir(parents=True, exist_ok=True)
    rows = []
    for i in tqdm(pick):
        meta, path = bags.metas[i], bags.df.resolved_path[i]
        j = int(np.argmax(segments_to_snippets(segs[i], meta["n_snippets"])))
        frames = get_pil_frames(cfg, P, bags.ids[i], path, [j * meta["stride"] + k for k in (0, meta["clip_len"] // 2, meta["clip_len"] - 1)])
        if not frames: continue
        fp = out_dir / "frames" / f"{bags.ids[i]}.jpg"; frames[len(frames) // 2].save(fp)
        for v in ex.variants(frames):
            rows.append(dict(video_id=bags.ids[i], true_class=bags.cls[i], snippet=j, frame=str(fp), **v, annotator1="", annotator2=""))
    csv = out_dir / "explanations_to_label.csv"; pd.DataFrame(rows).to_csv(csv, index=False)
    print(f"{len(rows)} rows -> {csv}\nFill annotator1/annotator2 (1=grounded, 0=not), then run evaluate_checker.")


if __name__ == "__main__":
    main()
