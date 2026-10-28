"""
Final evaluation (run ONCE per final configuration; the test split is never used for selection).

  python -m vad.evaluate --group tf --runs tf_s1 tf_s2 tf_s3
  python -m vad.evaluate --group tf --runs tf_s1 tf_s2 tf_s3 --vs base=base_s1,base_s2,base_s3 cliptf=cliptf_s1,cliptf_s2,cliptf_s3

Reports (artifacts/reports/eval_<group>.json + eval_<group>_videos.csv + png charts):
  VIDEO-LEVEL : ROC-AUC / AP of max snippet score vs video label
  FRAME-LEVEL : ROC-AUC over all test frames (literature-comparable; needs temporal annotations; annotated anomalous
                videos + all normal test videos)
  Operating point (threshold chosen on VALIDATION): P/R/F1/FPR, false alarms per hour of normal video
  Temporal IoU (annotated anomalous test videos)
  Calibration (scaler fitted on VALIDATION): ECE / Brier before & after, risk-coverage
  Mean +/- SD across seeds, paired bootstrap 95% CI of the seed-mean ensemble, optional paired comparison + Holm.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from vad import calibration as cal
from vad.config import load_config, paths
from vad.data.validate import parse_temporal
from vad.localize import extract_intervals, intervals_to_frames
from vad.metrics import (frame_arrays, frame_level_auc, pick_threshold_f1, prf, temporal_iou, video_level)
from vad.models import pick_device
from vad.stats import bootstrap_ci, holm, paired_bootstrap_diff
from vad.train import load_model, load_split_bags, predict_bags


def predict_split(run_name, cfg, split, device):
    P = paths(cfg)
    model, norm, mcfg, ck = load_model(P["runs"] / run_name / "best.pt", device)
    c2 = dict(cfg); c2["model"] = ck["cfg"]["model"]; c2["preprocess"] = ck["cfg"]["preprocess"]
    bags = load_split_bags(c2, split, norm)
    return bags, predict_bags(model, bags, device)


def get_annotations(cfg):
    p = Path(cfg["data"]["root_abs"]) / cfg["data"]["temporal_annotations"]
    if not p.exists():
        return None
    div = int(cfg["data"].get("gt_divisor", 1))
    ann = {}
    for k, segs in parse_temporal(p).items():
        segs = [(s_ // div, -(-e_ // div)) for s_, e_ in segs] if div != 1 else segs    # official frame numbers -> extracted-frame index
        ann[k] = segs; ann[Path(k).stem] = segs
    return ann


def frame_items(seg_list, bags, ann):
    """Per-video frame arrays; None for anomalous test videos lacking annotation."""
    items = []
    for i, vid in enumerate(bags.ids):
        meta = bags.metas[i]
        name = Path(bags.df.rel_path[i]).name
        if bags.labels[i] == 0:
            gt = []
        elif ann is not None and (name in ann or Path(name).stem in ann):
            gt = ann.get(name) or ann.get(Path(name).stem)
        else:
            items.append(None); continue
        items.append(frame_arrays(seg_list[i], meta["n_snippets"], meta["stride"], meta["n_frames_read"], gt))
    return items


def group_eval(cfg, runs, device, ann):
    ec = cfg["eval"]
    per_seed, test_segs, val_segs = [], [], []
    for r in runs:
        tb, ts = predict_split(r, cfg, "test", device)
        vb, vs = predict_split(r, cfg, "val", device)
        test_segs.append(ts); val_segs.append(vs)
    n_test = len(tb)
    ens = [np.mean([test_segs[s][i] for s in range(len(runs))], axis=0) for i in range(n_test)]
    ens_val = [np.mean([val_segs[s][i] for s in range(len(runs))], axis=0) for i in range(len(vb))]

    def vscore(segs): return np.array([s.max() for s in segs])
    y, yv = tb.labels, vb.labels

    # ---- per seed
    for s, r in enumerate(runs):
        vl = video_level(y, vscore(test_segs[s]))
        it = frame_items(test_segs[s], tb, ann)
        per_seed.append(dict(run=r, video_auc=vl["auc"], video_ap=vl["ap"],
                             frame_auc=frame_level_auc([i for i in it if i is not None])))
    agg = {k: dict(mean=float(np.nanmean([p[k] for p in per_seed])), sd=float(np.nanstd([p[k] for p in per_seed], ddof=1)) if len(runs) > 1 else None)
           for k in ["video_auc", "video_ap", "frame_auc"]}

    # ---- ensemble (seed-mean) metrics
    items = frame_items(ens, tb, ann)
    n_unannotated = sum(i is None for i in items)
    vs_t, vs_v = vscore(ens), vscore(ens_val)
    thr, f1v = pick_threshold_f1(yv, vs_v)                      # threshold from VALIDATION
    op = prf(y, vs_t, thr)
    normals_flagged_intervals, normal_hours = 0, 0.0
    ious = []
    for i in range(n_test):
        meta = tb.metas[i]
        from vad.seq import segments_to_snippets
        sn = segments_to_snippets(ens[i], meta["n_snippets"])
        flagged = vs_t[i] >= thr
        iv = extract_intervals(sn, ec["smooth_window"], ec["rel_threshold"], ec["min_len_snippets"], ec["merge_gap_snippets"])
        if y[i] == 0:
            normal_hours += meta["n_frames_read"] / meta["fps"] / 3600
            if flagged: normals_flagged_intervals += len(iv)
        elif ann is not None:
            name = Path(tb.df.rel_path[i]).name
            gt = ann.get(name) or ann.get(Path(name).stem)
            if gt:
                ious.append(temporal_iou(intervals_to_frames(iv, meta["stride"], meta["n_frames_read"]), gt))
    # ---- calibration
    sc = cal.fit_scaler(vs_v, yv, with_bias=False)
    sc_b = cal.fit_scaler(vs_v, yv, with_bias=True)
    p_raw, p_t, p_tb = vs_t, cal.apply_scaler(vs_t, sc), cal.apply_scaler(vs_t, sc_b)
    nb = ec["calib_bins"]
    calib = dict(scaler_T=sc, scaler_T_bias=sc_b,
                 raw=dict(ece=cal.ece(p_raw, y, nb), brier=cal.brier(p_raw, y)),
                 temperature=dict(ece=cal.ece(p_t, y, nb), brier=cal.brier(p_t, y)),
                 temperature_bias=dict(ece=cal.ece(p_tb, y, nb), brier=cal.brier(p_tb, y)),
                 note="MIL ranking scores are not probabilities; 'raw' ECE is shown only for reference.")
    cov, risk = cal.risk_coverage(p_t, y)
    seed_std = np.std([vscore(test_segs[s]) for s in range(len(runs))], axis=0) if len(runs) > 1 else np.zeros(n_test)
    routes = [cal.route(p_t[i], seed_std[i]) for i in range(n_test)]

    # ---- per-category (anomaly class vs ALL normal test videos)
    cls_arr = np.array(tb.cls)
    per_cat = {}
    for c in sorted(set(cls_arr) - {"Normal"}):
        m = (cls_arr == c) | (cls_arr == "Normal")
        idx = np.where(m)[0]
        per_cat[c] = dict(n_videos=int((cls_arr == c).sum()),
                          video_auc=video_level(y[idx], vs_t[idx])["auc"],
                          frame_auc=frame_level_auc([items[i] for i in idx if items[i] is not None]) if ann is not None else None)
    # ---- threshold sweep (false-alarm tracking)
    sweep = []
    for t in np.unique(np.round(np.quantile(vs_t, np.linspace(0, 1, 41)), 6)):
        r = prf(y, vs_t, t)
        sweep.append(dict(thr=float(t), tpr=r["recall"], fpr=r["fpr"], precision=r["precision"]))
    # ---- bootstrap CIs (ensemble)
    B = ec["bootstrap"]
    ann_items = items
    def m_video(idx, s=vs_t): return video_level(y[idx], s[idx])["auc"]
    def m_frame(idx, it=ann_items): return frame_level_auc([it[i] for i in idx if it[i] is not None])
    ci = dict(video_auc=bootstrap_ci(m_video, n_test, B), frame_auc=bootstrap_ci(m_frame, n_test, B) if ann is not None else None)

    res = dict(runs=runs, n_test=int(n_test), per_seed=per_seed, mean_sd_across_seeds=agg,
               ensemble=dict(video_auc=video_level(y, vs_t)["auc"], video_ap=video_level(y, vs_t)["ap"],
                             frame_auc=frame_level_auc([i for i in items if i is not None]),
                             n_videos_without_frame_gt=int(n_unannotated)),
               bootstrap_95ci=ci,
               operating_point=dict(source="threshold maximising F1 on validation", val_f1=f1v, **op,
                                    false_alarm_intervals_per_normal_hour=normals_flagged_intervals / normal_hours if normal_hours else None),
               temporal_iou=dict(mean=float(np.mean(ious)) if ious else None, n=len(ious),
                                 note="localisation params from config (not tunable on val: no val frame labels)"),
               calibration=calib, per_category=per_cat, threshold_sweep=sweep,
               risk_coverage=dict(coverage=cov[::max(1, len(cov)//50)].tolist(), error=risk[::max(1, len(cov)//50)].tolist()),
               routing_counts={k: routes.count(k) for k in set(routes)})
    arrays = dict(ens=ens, vs_t=vs_t, y=y, p_t=p_t, items=items, ids=tb.ids, seed_std=seed_std, test_bags=tb, route=routes)
    return res, arrays


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--group", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--vs", nargs="*", default=[], help="comparison models: name=run1,run2,run3 (paired bootstrap + Holm over ALL comparisons)")
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    P = paths(cfg); P["reports"].mkdir(parents=True, exist_ok=True)
    dev = pick_device(cfg["train"]["device"])
    ann = get_annotations(cfg)
    if ann is None:
        print("WARNING: no temporal annotation file -> frame-level AUC and temporal IoU skipped.")
    res, arr = group_eval(cfg, a.runs, dev, ann)
    if a.vs:
        y, B = arr["y"], cfg["eval"]["bootstrap"]
        comps, others = {}, {}
        for spec in a.vs:
            name, runs = spec.split("=", 1)
            r2, arr2 = group_eval(cfg, runs.split(","), dev, ann)
            others[name] = r2
            comps[f"{a.group}_minus_{name}:video_auc"] = paired_bootstrap_diff(
                lambda i, A=arr2: video_level(y[i], arr["vs_t"][i])["auc"], lambda i, A=arr2: video_level(y[i], A["vs_t"][i])["auc"], len(y), B)
            if ann is not None:
                comps[f"{a.group}_minus_{name}:frame_auc"] = paired_bootstrap_diff(
                    lambda i, A=arr2: frame_level_auc([arr["items"][k] for k in i if arr["items"][k] is not None]),
                    lambda i, A=arr2: frame_level_auc([A["items"][k] for k in i if A["items"][k] is not None]), len(y), B)
        adj = holm([c["p"] for c in comps.values()])
        for (k, c), (pa, rej) in zip(comps.items(), adj):
            c["holm_adjusted_p"], c["significant_after_holm"] = pa, rej
        res["comparisons"] = comps
        res["other_models"] = {k: {kk: v[kk] for kk in ["runs", "mean_sd_across_seeds", "ensemble", "bootstrap_95ci", "per_category"]} for k, v in others.items()}
    out = P["reports"] / f"eval_{a.group}.json"
    json.dump(res, open(out, "w"), indent=2, default=float)
    pd.DataFrame(dict(video_id=arr["ids"], label=arr["y"], video_score=arr["vs_t"], calibrated_p=arr["p_t"],
                      seed_std=arr["seed_std"], route=arr["route"])).to_csv(P["reports"] / f"eval_{a.group}_videos.csv", index=False)
    # per-video segment scores for the dashboard
    np.save(P["reports"] / f"eval_{a.group}_segscores.npy", np.array(arr["ens"], dtype=object), allow_pickle=True)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        rel = cal.reliability(arr["p_t"], arr["y"], cfg["eval"]["calib_bins"])
        fig, ax = plt.subplots(1, 2, figsize=(9, 4))
        ax[0].plot([0, 1], [0, 1], "k--"); ax[0].plot([r["conf"] for r in rel], [r["acc"] for r in rel], "o-")
        ax[0].set(xlabel="confidence", ylabel="fraction anomalous", title="Reliability (test)")
        rc = res["risk_coverage"]; ax[1].plot(rc["coverage"], rc["error"]); ax[1].set(xlabel="coverage", ylabel="error", title="Risk-coverage")
        fig.tight_layout(); fig.savefig(P["reports"] / f"eval_{a.group}_calibration.png", dpi=120)
    except Exception as e:
        print("plot skipped:", e)
    print(json.dumps({k: res[k] for k in ["mean_sd_across_seeds", "ensemble", "bootstrap_95ci", "operating_point", "temporal_iou"]}, indent=2, default=float))
    print("saved", out)


if __name__ == "__main__":
    main()
