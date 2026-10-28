"""Torch-free evaluation metrics. Video-level and frame-level are kept strictly separate."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from vad.seq import segments_to_snippets, snippet_frame_ranges


def video_level(labels, scores) -> dict:
    labels, scores = np.asarray(labels), np.asarray(scores)
    if len(set(labels.tolist())) < 2:
        return dict(auc=float("nan"), ap=float("nan"))
    return dict(auc=float(roc_auc_score(labels, scores)), ap=float(average_precision_score(labels, scores)))


def merge_segments(segs):
    segs = sorted((int(s), int(e)) for s, e in segs if e > s)
    out = []
    for s, e in segs:
        if out and s <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def frame_arrays(seg_scores, n_snip: int, stride: int, n_frames: int, gt_segments):
    """Per-snippet (score, #positive frames, #negative frames). Frame scores are constant inside a snippet,
    so weighted snippet-level AUC equals the exact frame-level AUC."""
    sn = segments_to_snippets(np.asarray(seg_scores), n_snip)
    ranges = snippet_frame_ranges(n_snip, stride, n_frames)
    gt = merge_segments(gt_segments or [])
    pos = np.zeros(n_snip)
    for j, (a, b) in enumerate(ranges):
        pos[j] = sum(max(0, min(b, e) - max(a, s)) for s, e in gt)
    tot = np.array([b - a for a, b in ranges], dtype=float)
    return sn, pos, tot - pos


def frame_level_auc(items) -> float:
    """items: list of (scores, pos, neg) tuples. Needs at least one positive and one negative frame."""
    if not items:
        return float("nan")
    s = np.concatenate([i[0] for i in items])
    p = np.concatenate([i[1] for i in items])
    n = np.concatenate([i[2] for i in items])
    if p.sum() == 0 or n.sum() == 0:
        return float("nan")
    y = np.concatenate([np.ones_like(s), np.zeros_like(s)])
    w = np.concatenate([p, n])
    ss = np.concatenate([s, s])
    keep = w > 0
    return float(roc_auc_score(y[keep], ss[keep], sample_weight=w[keep]))


def pick_threshold_f1(labels, scores):
    """Operating threshold maximising F1 on VALIDATION data. Returns (thr, f1)."""
    labels, scores = np.asarray(labels), np.asarray(scores)
    best = (0.5, -1.0)
    for t in np.unique(scores):
        f = prf(labels, scores, t)["f1"]
        if f > best[1] + 1e-12:
            best = (float(t), f)
    return best


def prf(labels, scores, thr) -> dict:
    labels = np.asarray(labels)
    pred = (np.asarray(scores) >= thr).astype(int)
    tp = int(((pred == 1) & (labels == 1)).sum()); fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum()); tn = int(((pred == 0) & (labels == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return dict(thr=float(thr), tp=tp, fp=fp, fn=fn, tn=tn, precision=prec, recall=rec, f1=f1,
                fpr=fp / (fp + tn) if fp + tn else 0.0)


def temporal_iou(pred_frames, gt_frames) -> float:
    """Both: list of (start, end) frame intervals. IoU of the union masks."""
    p, g = merge_segments(pred_frames), merge_segments(gt_frames)
    inter = sum(max(0, min(pe, ge) - max(ps, gs)) for ps, pe in p for gs, ge in g)
    up = sum(e - s for s, e in p); ug = sum(e - s for s, e in g)
    union = up + ug - inter
    return inter / union if union > 0 else 0.0
