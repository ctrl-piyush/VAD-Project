"""Pure-numpy groundedness logic (synopsis 4.6). Heavy models are loaded elsewhere (vad/explain/models.py)."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, cohen_kappa_score


def _unit(x):
    x = np.asarray(x, dtype=np.float64)
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def clip_visual_support(text_emb, frame_embs, mismatched_pool_embs):
    """text_emb [d]; frame_embs [k,d] (flagged frames); mismatched_pool_embs [m,d] (frames of OTHER incidents).
    Returns dict(cos, clipscore=2.5*max(cos,0), percentile): percentile = share of mismatched pairs with lower cosine."""
    t = _unit(text_emb)
    cos = float((_unit(frame_embs) @ t).max())
    pool = _unit(mismatched_pool_embs) @ t
    return dict(cos=cos, clipscore=2.5 * max(cos, 0.0), percentile=float((pool < cos).mean()))


def groundedness(clip_percentile, nli_entail_prob):
    """Decision rule from the synopsis: minimum of the two normalised scores."""
    return float(min(clip_percentile, nli_entail_prob))


def tune_threshold(scores, human_labels):
    """Threshold maximising F1 against human 'grounded' labels (1 = grounded). Do this on a VALIDATION part of the labelled set."""
    s, y = np.asarray(scores), np.asarray(human_labels)
    best = (0.5, -1.0)
    for t in np.unique(s):
        p = (s >= t).astype(int)
        tp = ((p == 1) & (y == 1)).sum(); fp = ((p == 1) & (y == 0)).sum(); fn = ((p == 0) & (y == 1)).sum()
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0
        if f1 > best[1]: best = (float(t), float(f1))
    return best


def checker_report(scores, human_labels, thr, annotator_a=None, annotator_b=None):
    """Checker quality vs human labels. Positive class = UNGROUNDED (hallucination) as in the synopsis' hallucination-rate use."""
    s, y = np.asarray(scores), np.asarray(human_labels)
    flagged_ungrounded = (s < thr).astype(int); truth_ungrounded = (1 - y)
    rep = dict(thr=thr, n=len(s), hallucination_rate_human=float(truth_ungrounded.mean()),
               precision_ungrounded=float(precision_score(truth_ungrounded, flagged_ungrounded, zero_division=0)),
               recall_ungrounded=float(recall_score(truth_ungrounded, flagged_ungrounded, zero_division=0)))
    if len(set(y.tolist())) == 2:
        rep["roc_auc"] = float(roc_auc_score(y, s))
    if annotator_a is not None and annotator_b is not None:
        rep["cohen_kappa"] = float(cohen_kappa_score(annotator_a, annotator_b))
    return rep
