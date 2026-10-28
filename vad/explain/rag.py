"""Retrieval store of similar PAST incidents (train videos only). FAISS if installed, numpy otherwise."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

TEMPLATES = {  # category label -> short template caption (synopsis 4.5). Edit freely; these are labels, not annotations.
    "Abuse": "a person is being physically abused", "Arrest": "police are arresting a person",
    "Arson": "a fire is deliberately set", "Assault": "a person is assaulting another person",
    "Burglary": "someone is breaking into a building", "Explosion": "an explosion occurs",
    "Fighting": "people are fighting", "RoadAccidents": "a road accident occurs",
    "Robbery": "a person is being robbed", "Shooting": "a person is shooting a weapon",
    "Shoplifting": "a person is shoplifting in a store", "Stealing": "a person is stealing something",
    "Vandalism": "someone is vandalising property", "Normal": "ordinary activity with nothing unusual"}


def _unit(x):
    x = np.asarray(x, dtype=np.float32)
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


class Store:
    def __init__(self, embs, captions, video_ids, classes):
        self.embs, self.captions, self.video_ids, self.classes = _unit(embs), captions, video_ids, classes
        try:
            import faiss
            self.index = faiss.IndexFlatIP(self.embs.shape[1]); self.index.add(self.embs)
        except ImportError:
            self.index = None

    def search(self, q, k=3, exclude_video=None):
        q = _unit(np.atleast_2d(q))
        if self.index is not None:
            sims, ids = self.index.search(q, k + 5); sims, ids = sims[0], ids[0]
        else:
            s = (self.embs @ q.T).ravel(); ids = np.argsort(-s)[:k + 5]; sims = s[ids]
        out = [(int(i), float(s)) for i, s in zip(ids, sims) if i >= 0 and self.video_ids[i] != exclude_video][:k]
        return [dict(video_id=self.video_ids[i], cls=self.classes[i], caption=self.captions[i], sim=s) for i, s in out]

    def save(self, d: Path):
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "embs.npy", self.embs)
        json.dump(dict(captions=self.captions, video_ids=self.video_ids, classes=self.classes), open(d / "meta.json", "w"))

    @staticmethod
    def load(d: Path):
        m = json.load(open(d / "meta.json"))
        return Store(np.load(d / "embs.npy"), m["captions"], m["video_ids"], m["classes"])


def assert_train_only(video_ids, manifest_df):
    """Leakage guard: every indexed video must be in split == 'train'."""
    split = dict(zip(manifest_df.video_id, manifest_df.split))
    bad = [v for v in video_ids if split.get(v) != "train"]
    if bad:
        raise AssertionError(f"Retrieval store contains non-train videos (leakage): {bad[:5]}")
