"""
Build the retrieval store from TRAIN videos only.
  python -m vad.explain.build_index --run tf_s1 [--captions-json my_uca_captions.json] [--max-normal 300]
captions-json (optional): {"<video_id>": "caption", ...} e.g. converted from the UCA annotations YOU downloaded.
Otherwise the category-label templates in rag.TEMPLATES are used (no captions are invented).
"""
from __future__ import annotations
import argparse, json
import cv2, numpy as np, pandas as pd
from PIL import Image
from tqdm import tqdm
from vad.config import load_config, paths
from vad.data.frames import get_pil_frames
from vad.explain.models import Clip
from vad.explain.rag import Store, TEMPLATES, assert_train_only
from vad.models import pick_device
from vad.seq import segments_to_snippets
from vad.train import load_model, load_split_bags, predict_bags


def read_frame(path, idx):
    cap = cv2.VideoCapture(path); cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx)); ok, fr = cap.read(); cap.release()
    return Image.fromarray(fr[:, :, ::-1]) if ok else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml"); ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--run", required=True); ap.add_argument("--captions-json"); ap.add_argument("--max-normal", type=int, default=300)
    a = ap.parse_args()
    cfg = load_config(a.config, a.set); P = paths(cfg); dev = pick_device(cfg["train"]["device"])
    model, norm, mcfg, ck = load_model(P["runs"] / a.run / "best.pt", dev)
    c2 = dict(cfg); c2["model"] = ck["cfg"]["model"]; c2["preprocess"] = ck["cfg"]["preprocess"]
    bags = load_split_bags(c2, "train", norm)
    segs = predict_bags(model, bags, dev)
    df = pd.read_csv(P["manifest"]); mf = df.set_index("video_id")
    caps_user = json.load(open(a.captions_json)) if a.captions_json else {}
    clip = Clip(cfg["explain"]["clip_model"], dev)
    rng = np.random.RandomState(0); normals = 0
    imgs, caps, vids, cls = [], [], [], []
    for i, v in enumerate(tqdm(bags.ids)):
        meta = bags.metas[i]; row = mf.loc[v]
        if bags.labels[i] == 0:
            if normals >= a.max_normal: continue
            normals += 1; j = rng.randint(meta["n_snippets"])
        else:
            j = int(np.argmax(segments_to_snippets(segs[i], meta["n_snippets"])))
        got = get_pil_frames(cfg, P, v, row.resolved_path, [j * meta["stride"] + meta["clip_len"] // 2])
        if not got: continue
        im = got[0]
        imgs.append(im); vids.append(v); cls.append(row.cls); caps.append(caps_user.get(v, TEMPLATES.get(row.cls, row.cls)))
    assert_train_only(vids, df)
    embs = np.concatenate([clip.image(imgs[k:k + 32]) for k in range(0, len(imgs), 32)])
    Store(embs, caps, vids, cls).save(P["artifacts"] / "rag_store")
    print(f"Indexed {len(vids)} train videos -> {P['artifacts'] / 'rag_store'}")


if __name__ == "__main__":
    main()
