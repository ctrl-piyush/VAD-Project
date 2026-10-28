"""Explain a flagged snippet and compute groundedness. Distinguishes: anomaly score, visual evidence, textual consistency."""
from __future__ import annotations
import numpy as np
from vad.explain.groundedness import clip_visual_support, groundedness
from vad.explain.rag import Store


class Explainer:
    def __init__(self, cfg, device, store: Store, use_vlm=True):
        from vad.explain.models import Clip, NLI, Blip2
        ec = cfg["explain"]
        self.cfg, self.store = ec, store
        self.clip, self.nli = Clip(ec["clip_model"], device), NLI(ec["nli_model"], device)
        self.vlm = Blip2(ec["vlm_model"], device) if use_vlm else None

    def explain(self, frames_pil, exclude_video=None):
        """frames_pil: list of PIL frames from the flagged snippet (top-scoring first)."""
        fe = self.clip.image(frames_pil)
        hits = self.store.search(fe.mean(0), self.cfg["top_k"], exclude_video)
        caps = [h["caption"] for h in hits]
        if self.vlm is not None:
            text, source = self.vlm.explain(frames_pil[0], caps), "BLIP-2 + retrieved context"
        else:
            text, source = f"Resembles past incidents: {caps[0]}" if caps else "no similar incident retrieved", "retrieval-only (no VLM)"
        te = self.clip.text([text])[0]
        pool_ids = [i for i, v in enumerate(self.store.video_ids) if v not in {h["video_id"] for h in hits}]
        vis = clip_visual_support(te, fe, self.store.embs[pool_ids])
        nli = self.nli.entail_prob(" ".join(caps), text) if caps else 0.0
        g = groundedness(vis["percentile"], nli)
        return dict(explanation=text, source=source, retrieved=hits, visual_support=vis, nli_entailment=nli,
                    groundedness=g, grounded=bool(g >= self.cfg["groundedness_threshold"]),
                    caveat=("NLI is circular for retrieval-only text (explanation is built from the captions)."
                            if self.vlm is None else "Threshold is a placeholder until tuned on human labels."))


    def _score(self, text, fe, hits):
        te = self.clip.text([text])[0]
        pool_ids = [i for i, v in enumerate(self.store.video_ids) if v not in {h["video_id"] for h in hits}]
        vis = clip_visual_support(te, fe, self.store.embs[pool_ids])
        caps = [h["caption"] for h in hits]
        nli = self.nli.entail_prob(" ".join(caps), text) if caps else 0.0
        return dict(clip_percentile=vis["percentile"], clip_cos=vis["cos"], nli=nli, groundedness=groundedness(vis["percentile"], nli))

    def variants(self, frames_pil, exclude_video=None):
        """Ablation variants (synopsis 4.9): rag = retrieval-only text; vlm = BLIP-2 without context; rag_vlm = BLIP-2 + retrieved context."""
        fe = self.clip.image(frames_pil)
        hits = self.store.search(fe.mean(0), self.cfg["top_k"], exclude_video)
        caps = [h["caption"] for h in hits]
        texts = {"rag": f"Resembles past incidents: {caps[0]}" if caps else "no similar incident retrieved"}
        if self.vlm is not None:
            texts["vlm"] = self.vlm.explain(frames_pil[0], [])
            texts["rag_vlm"] = self.vlm.explain(frames_pil[0], caps)
        return [dict(variant=k, explanation=t, retrieved_captions=" | ".join(caps), **self._score(t, fe, hits)) for k, t in texts.items()]
