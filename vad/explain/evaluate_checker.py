"""
Validate the groundedness checker against human labels; compute hallucination rates for the explanation ablation.
  python -m vad.explain.evaluate_checker --csv artifacts/explanations/explanations_to_label.csv
Label rule: consensus when both annotators agree; disagreements are dropped (counted) unless a column `final` is filled.
Threshold is tuned on one half of the VIDEOS and the checker is scored on the other half. CIs: bootstrap over rows.
"""
from __future__ import annotations
import argparse, json
import numpy as np, pandas as pd
from sklearn.metrics import cohen_kappa_score, roc_auc_score
from vad.config import load_config, paths
from vad.explain.groundedness import tune_threshold, checker_report


def boot(fn, n, B=1000, seed=0):
    r = np.random.RandomState(seed); v = [x for x in (fn(r.randint(0, n, n)) for _ in range(B)) if x == x]
    return dict(lo=float(np.quantile(v, .025)), hi=float(np.quantile(v, .975))) if v else None


def evaluate(df: pd.DataFrame, seed=0):
    df = df.dropna(subset=["annotator1", "annotator2"]).copy()
    df = df[(df.annotator1.astype(str) != "") & (df.annotator2.astype(str) != "")]
    a1, a2 = df.annotator1.astype(int), df.annotator2.astype(int)
    rep = dict(n_labelled=int(len(df)), cohen_kappa=float(cohen_kappa_score(a1, a2)), raw_agreement=float((a1 == a2).mean()))
    if "final" in df.columns and df["final"].notna().any():
        df["label"] = df["final"].fillna(a1.where(a1 == a2)).astype(float)
    else:
        df["label"] = a1.where(a1 == a2)
    rep["n_disagreements_dropped"] = int(df.label.isna().sum()); df = df.dropna(subset=["label"]); df["label"] = df.label.astype(int)
    rep["hallucination_rate_by_variant"] = {v: dict(rate=float(1 - g.label.mean()), n=int(len(g)), ci=boot(lambda i, g=g: 1 - g.label.to_numpy()[i].mean(), len(g)))
                                            for v, g in df.groupby("variant")}
    vids = np.array(sorted(df.video_id.unique())); rng = np.random.RandomState(seed); rng.shuffle(vids)
    tune_v = set(vids[: len(vids) // 2]); tune, test = df[df.video_id.isin(tune_v)], df[~df.video_id.isin(tune_v)]
    if len(tune) < 5 or len(test) < 5 or tune.label.nunique() < 2 or test.label.nunique() < 2:
        rep["checker"] = "not enough labelled rows with both classes to tune/test"; return rep
    rep["checker"] = {}
    for name, col in [("clip_only", "clip_percentile"), ("nli_only", "nli"), ("min(clip,nli) [proposed]", "groundedness")]:
        thr, f1 = tune_threshold(tune[col], tune.label)
        r = checker_report(test[col], test.label, thr)
        r["tune_f1"] = f1
        r["roc_auc_ci"] = boot(lambda i, c=col: roc_auc_score(test.label.to_numpy()[i], test[c].to_numpy()[i]) if len(set(test.label.to_numpy()[i])) == 2 else float("nan"), len(test))
        rep["checker"][name] = r
    thr = rep["checker"]["min(clip,nli) [proposed]"]["thr"]
    rv = df[df.variant == "rag_vlm"] if (df.variant == "rag_vlm").any() else df
    acc = rv[rv.groundedness >= thr]
    rep["with_evidence_grounding"] = dict(threshold=thr, acceptance_rate=float(len(acc) / max(1, len(rv))),
                                         hallucination_rate_before=float(1 - rv.label.mean()),
                                         hallucination_rate_after=float(1 - acc.label.mean()) if len(acc) else None,
                                         note="threshold tuned on the tune half; rates here use all rows of this variant")
    return rep


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--csv", required=True); ap.add_argument("--config", default="configs/default.yaml")
    a = ap.parse_args(); P = paths(load_config(a.config)); P["reports"].mkdir(parents=True, exist_ok=True)
    rep = evaluate(pd.read_csv(a.csv)); json.dump(rep, open(P["reports"] / "explanation_eval.json", "w"), indent=2, default=float)
    print(json.dumps(rep, indent=2, default=float))


if __name__ == "__main__":
    main()
