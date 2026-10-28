"""
Run:  streamlit run app/streamlit_app.py
Every control performs a real action (validation, extraction, training, evaluation, analysis) or says it is unavailable.
Paths come from configs/default.yaml and env VAD_DATA_ROOT / VAD_ARTIFACTS.
"""
import json, sys, tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import numpy as np, pandas as pd, streamlit as st

from app import jobs
from vad.config import load_config, paths

st.set_page_config(page_title="VAD: CNN-Transformer MIL", layout="wide")
cfg_path = st.sidebar.text_input("Config file", "configs/default.yaml")
overrides = st.sidebar.text_input("Overrides (key=value, space separated)", "").split()
try:
    cfg = load_config(cfg_path, overrides)
except Exception as e:
    st.error(f"Cannot load config: {e}"); st.stop()
P = paths(cfg)
st.sidebar.caption(f"data root: {cfg['data']['root_abs']}\n\nartifacts: {cfg['project']['artifacts_abs']}")
api_url = st.sidebar.text_input("FastAPI backend URL (optional)", "", help="e.g. http://localhost:8000 (start: uvicorn api.main:app). Empty = run inference in-process")
page = st.sidebar.radio("Page", ["1 Dataset", "2 Preprocessing", "3 Training", "4 Analyze video", "5 Evaluation", "6 Explanations", "7 Export", "8 Trends"])


def runs_available():
    return sorted([p.name for p in P["runs"].iterdir() if (p / "best.pt").exists()]) if P["runs"].exists() else []


def job_panel(name):
    s = jobs.status(cfg, name)
    st.write(f"**Job `{name}`:** {s['state']}")
    if s.get("log"):
        st.code(jobs.tail(s["log"]), language="text")
    c1, c2 = st.columns(2)
    if c1.button("Refresh log", key=f"r_{name}"): st.rerun()
    if s["state"] == "running" and c2.button("Stop job", key=f"s_{name}"):
        jobs.stop(cfg, name); st.rerun()


# ------------------------------------------------------------------ 1 Dataset
if page.startswith("1"):
    st.header("Dataset")
    root = Path(cfg["data"]["root_abs"])
    checks = {"videos dir": root / cfg["data"]["videos_dir"], "train split": root / cfg["data"]["train_split"],
              "test split": root / cfg["data"]["test_split"], "temporal annotations": root / cfg["data"]["temporal_annotations"]}
    st.table(pd.DataFrame([dict(item=k, path=str(v), exists=v.exists()) for k, v in checks.items()]))
    c1, c2, c3 = st.columns(3)
    quick = c1.checkbox("Quick test (first 20 videos/split)", True)
    if cfg["data"].get("source") == "frames":
        froot = st.text_input("Frames root (folder containing Train/ and Test/)", cfg["data"].get("frames_root") or "")
        if c2.button("Index frame dataset") and froot:
            jobs.start(cfg, "validate", ["vad.data.index_frames", "--config", cfg_path, "--root", froot, "--set", *overrides] + (["--limit", "20"] if quick else []))
    elif c2.button("Run validation"):
        jobs.start(cfg, "validate", ["vad.data.validate", "--config", cfg_path, "--set", *overrides] + (["--limit", "20"] if quick else []))
    if c3.button("Create fixed validation split"):
        jobs.start(cfg, "valsplit", ["vad.data.make_val_split", "--config", cfg_path, "--set", *overrides])
    job_panel("validate"); job_panel("valsplit")
    rep_full, rep_part = P["dataset"] / "validation_report.json", P["dataset"] / "validation_report_partial.json"
    rep = rep_full if rep_full.exists() and not (quick and rep_part.exists()) else rep_part
    if rep.exists():
        r = json.load(open(rep)); st.subheader("Validation report"); st.caption(r.get("note", ""))
        st.json({"blocking_problem_types": r["blocking_problem_types"], "warnings": r["warnings"], "stats": r["stats"]})
        with st.expander("All problems"): st.json(r["problems"])
    if P["manifest"].exists():
        df = pd.read_csv(P["manifest"]); st.subheader("Manifest")
        st.dataframe(df.groupby(["split", "cls"]).size().unstack(fill_value=0).T)
    if st.button("Run dataset analysis (imbalance / lengths / shift)"):
        jobs.start(cfg, "analysis", ["vad.analysis", "--config", cfg_path, "--set", *overrides])
    job_panel("analysis")
    ap_ = P["reports"] / "dataset_overview.png"
    if ap_.exists(): st.image(str(ap_))
    ar = P["reports"] / "dataset_analysis.json"
    if ar.exists(): st.json(json.load(open(ar)))

# ------------------------------------------------------------------ 2 Preprocessing
elif page.startswith("2"):
    st.header("Feature extraction (cached, resumable)")
    pc = cfg["preprocess"]
    st.write(f"backbone **{pc['backbone']}**, weights **{pc['weights']}**, {pc['clip_len']}-frame snippets, stride {pc['clip_stride']}")
    if pc["weights"] == "none": st.warning("weights=none: random backbone, smoke tests only")
    if P["manifest"].exists():
        df = pd.read_csv(P["manifest"]); df = df[df.status == "ok"]
        done = int(df.video_id.map(lambda v: (P["features"] / f"{v}.npy").exists() and (P["features"] / f"{v}.json").exists()).sum())
        st.progress(done / max(1, len(df)), text=f"{done}/{len(df)} videos have cached features")
        hours = df.duration_s.sum() / 3600
        st.caption(f"Total listed video: {hours:.1f} h. Feature size ~ {hours*3600*30/16*512*2/1e9:.2f} GB at 30 fps for 512-d fp16 (estimate).")
    else:
        st.info("No manifest yet. Run validation on page 1.")
    n = st.number_input("Limit (0 = all; use ~10 first)", 0, 100000, 10)
    if st.button("Start / resume extraction"):
        jobs.start(cfg, "extract", ["vad.data.features", "--config", cfg_path, "--limit", str(int(n)), "--set", *overrides])
    job_panel("extract")
    rp = P["reports"] / "extraction_report.json"
    if rp.exists(): st.json(json.load(open(rp)))

# ------------------------------------------------------------------ 3 Training
elif page.startswith("3"):
    st.header("Training")
    c1, c2, c3 = st.columns(3)
    name = c1.text_input("Run name", "tf_s42"); seed = c2.number_input("Seed", 0, 99999, 42)
    mtype = c3.selectbox("Model", ["transformer", "baseline"])
    extra = st.text_input("Extra overrides (e.g. train.lr=0.0003 model.n_layers=1)", "").split()
    a, b = st.columns(2)
    if a.button("Start new training"):
        jobs.start(cfg, f"train_{name}", ["vad.train", "--config", cfg_path, "--run-name", name, "--seed", str(int(seed)), "--set", *overrides, f"model.type={mtype}", *extra])
    if b.button("Resume training"):
        jobs.start(cfg, f"train_{name}", ["vad.train", "--config", cfg_path, "--run-name", name, "--seed", str(int(seed)), "--resume", "--set", *overrides, f"model.type={mtype}", *extra])
    job_panel(f"train_{name}")
    lg = P["runs"] / name / "log.jsonl"
    if lg.exists():
        d = pd.DataFrame([json.loads(l) for l in open(lg)])
        st.line_chart(d.set_index("epoch")[["train_loss", "val_auc"]])
        st.dataframe(d.tail(10))
    st.subheader("Checkpoints")
    rows = []
    for r in (P["runs"].iterdir() if P["runs"].exists() else []):
        if r.is_dir():
            s = r / "summary.json"
            rows.append(dict(run=r.name, best_pt=(r / "best.pt").exists(), last_pt=(r / "last.pt").exists(),
                             best_val_auc=json.load(open(s))["best_val_auc"] if s.exists() else None))
    st.dataframe(pd.DataFrame(rows))
    if st.button("Run validation-driven tuning table (long!)"):
        jobs.start(cfg, "tune", ["vad.tune", "--config", cfg_path, "--extra-set", *overrides])
    job_panel("tune")
    tt = P["reports"] / "tuning_table.csv"
    if tt.exists(): st.dataframe(pd.read_csv(tt))

# ------------------------------------------------------------------ 4 Analyze
elif page.startswith("4"):
    st.header("Analyze a video")
    rs = runs_available()
    if not rs: st.info("No trained checkpoint (runs/*/best.pt). Train first."); st.stop()
    run = st.selectbox("Checkpoint run", rs)
    if st.button("Fit operating point on validation split"):
        with st.spinner("fitting..."):
            from vad.infer import fit_operating_point
            from vad.models import pick_device
            try: st.json(fit_operating_point(cfg, run, pick_device(cfg["train"]["device"])))
            except Exception as e: st.error(str(e))
    up = st.file_uploader("Video", type=["mp4", "avi", "mkv", "mov"])
    if up is not None and st.button("Analyze"):
        tmp = Path(tempfile.mkdtemp()) / up.name; tmp.write_bytes(up.getbuffer())
        try:
            with st.spinner("extracting features + scoring..."):
                if api_url:
                    import requests
                    r = requests.post(f"{api_url.rstrip('/')}/analyze", params={"run": run}, files={"video": (up.name, up.getbuffer())}, timeout=3600)
                    r.raise_for_status(); st.session_state["res"] = r.json()
                else:
                    from vad.infer import Analyzer
                    st.session_state["res"] = Analyzer(cfg, run).analyze(tmp)
                st.session_state["video"] = str(tmp)
        except Exception as e:
            st.error(f"Analysis failed: {e}")
    res = st.session_state.get("res")
    if res:
        st.video(st.session_state["video"])
        c = st.columns(4)
        c[0].metric("video score (max)", f"{res['video_score']:.3f}")
        if "calibrated_p" in res:
            c[1].metric("calibrated p", f"{res['calibrated_p']:.2f}"); c[2].metric("flagged", str(res["flagged"])); c[3].metric("routing", res["route"])
        st.caption(res.get("note", ""))
        fps, stride = res["fps"], res["stride"]
        t = np.arange(len(res["snippet_scores"])) * stride / fps
        st.line_chart(pd.DataFrame({"score": res["snippet_scores"]}, index=pd.Index(t, name="seconds")))
        st.write("Predicted intervals:", res["intervals_fmt"] or "none")
        st.session_state["res_intervals"] = res

# ------------------------------------------------------------------ 5 Evaluation
elif page.startswith("5"):
    st.header("Evaluation (test split: run only for final configs)")
    rs = runs_available()
    group = st.text_input("Group name", "tf")
    sel = st.multiselect("Runs (seeds) of the proposed model", rs)
    osel = st.multiselect("Runs of comparison model (optional, paired bootstrap)", rs)
    oname = st.text_input("Comparison name", "base")
    if st.button("Run evaluation on TEST"):
        if not sel: st.error("Select at least one run")
        else:
            args = ["vad.evaluate", "--config", cfg_path, "--group", group, "--runs", *sel]
            if osel: args += ["--vs", f"{oname}=" + ",".join(osel)]
            jobs.start(cfg, f"eval_{group}", args + ["--set", *overrides])
    job_panel(f"eval_{group}")
    rp = P["reports"] / f"eval_{group}.json"
    if rp.exists():
        r = json.load(open(rp))
        c = st.columns(3); e = r["ensemble"]
        c[0].metric("video AUC (ensemble)", f"{e['video_auc']:.4f}"); c[1].metric("frame AUC (ensemble)", f"{e['frame_auc']:.4f}" if e["frame_auc"] == e["frame_auc"] else "n/a")
        t = r["temporal_iou"]["mean"]; c[2].metric("temporal IoU", f"{t:.3f}" if t is not None else "n/a")
        st.subheader("Mean ± SD across seeds"); st.json(r["mean_sd_across_seeds"])
        st.subheader("Paired-bootstrap 95% CI (seed-ensemble)"); st.json(r["bootstrap_95ci"])
        st.subheader("Operating point (threshold from validation)"); st.json(r["operating_point"])
        st.subheader("Calibration"); st.json(r["calibration"]); st.json(r["routing_counts"])
        if "comparisons" in r: st.subheader("Paired comparisons (Holm-corrected)"); st.json(r["comparisons"])
        png = P["reports"] / f"eval_{group}_calibration.png"
        if png.exists(): st.image(str(png))
        vc = P["reports"] / f"eval_{group}_videos.csv"
        if vc.exists(): st.dataframe(pd.read_csv(vc))

# ------------------------------------------------------------------ 6 Explanations
elif page.startswith("6"):
    st.header("Explanations (Extended tier)")
    try:
        import transformers  # noqa
        have_tf = True
    except ImportError:
        have_tf = False
    if not have_tf:
        st.warning("UNAVAILABLE: `transformers` is not installed (pip install transformers accelerate pillow). The detector works without it.")
    store_dir = P["artifacts"] / "rag_store"
    st.write("Retrieval store:", "found" if (store_dir / "meta.json").exists() else "not built")
    rs = runs_available()
    if rs and have_tf:
        run = st.selectbox("Run for index building", rs)
        if st.button("Build retrieval store from TRAIN videos"):
            jobs.start(cfg, "build_index", ["vad.explain.build_index", "--config", cfg_path, "--run", run, "--set", *overrides])
        job_panel("build_index")
    res, vid = st.session_state.get("res"), st.session_state.get("video")
    if have_tf and (store_dir / "meta.json").exists() and res and vid:
        use_vlm = st.checkbox("Use BLIP-2 (large download; unchecked = retrieval-only text)", False)
        if st.button("Explain top flagged snippet"):
            import cv2
            from PIL import Image
            from vad.explain.pipeline import Explainer
            from vad.explain.rag import Store
            from vad.models import pick_device
            j = int(np.argmax(res["snippet_scores"]))
            cap = cv2.VideoCapture(vid); frames = []
            for k in (0, 8, 15):
                cap.set(cv2.CAP_PROP_POS_FRAMES, j * res["stride"] + k); ok, fr = cap.read()
                if ok: frames.append(Image.fromarray(fr[:, :, ::-1]))
            cap.release()
            with st.spinner("loading models + explaining..."):
                out = Explainer(cfg, pick_device(cfg["train"]["device"]), Store.load(store_dir), use_vlm).explain(frames)
            st.image(frames[0]); st.write("**Explanation:**", out["explanation"]); st.caption(out["source"])
            st.json({k: out[k] for k in ["visual_support", "nli_entailment", "groundedness", "grounded", "caveat"]}); st.json(out["retrieved"])
            st.info("Groundedness is evidence of consistency, not proof that the event happened.")
    else:
        st.caption("Analyze a video on page 4 and build the store to enable explanations.")

# ------------------------------------------------------------------ 8 Trends
elif page.startswith("8"):
    st.header("Trends, false-alarm tracking, per-category performance (recorded test videos)")
    groups = sorted(p.stem[len("eval_"):] for p in P["reports"].glob("eval_*.json")) if P["reports"].exists() else []
    if not groups: st.info("Run an evaluation first (page 5)."); st.stop()
    g = st.selectbox("Evaluation group", groups)
    r = json.load(open(P["reports"] / f"eval_{g}.json"))
    st.subheader("Per-category AUC (anomaly class vs all normal test videos)")
    pc_ = pd.DataFrame(r["per_category"]).T
    st.bar_chart(pc_[["video_auc"] + (["frame_auc"] if "frame_auc" in pc_ else [])].astype(float)); st.dataframe(pc_)
    st.subheader("False-alarm tracking: threshold sweep")
    sw = pd.DataFrame(r["threshold_sweep"]); st.line_chart(sw.set_index("thr")[["tpr", "fpr", "precision"]])
    op = r["operating_point"]
    st.metric("false-alarm intervals per normal hour (at val threshold)", f"{op['false_alarm_intervals_per_normal_hour']:.1f}" if op["false_alarm_intervals_per_normal_hour"] is not None else "n/a")
    vc = P["reports"] / f"eval_{g}_videos.csv"
    if vc.exists():
        v = pd.read_csv(vc); st.subheader("Anomaly score distribution")
        h = pd.DataFrame({"normal": np.histogram(v[v.label == 0].video_score, bins=20, range=(0, 1))[0],
                          "anomalous": np.histogram(v[v.label == 1].video_score, bins=20, range=(0, 1))[0]}, index=np.linspace(0.025, 0.975, 20).round(3))
        st.bar_chart(h); st.subheader("Per-video scores (sorted)"); st.dataframe(v.sort_values("video_score", ascending=False))

# ------------------------------------------------------------------ 7 Export
else:
    st.header("Export")
    items = {}
    for r in (P["reports"].glob("*") if P["reports"].exists() else []):
        items[r.name] = r
    for r in (P["runs"].glob("*/config.json") if P["runs"].exists() else []):
        items[f"{r.parent.name}__config.json"] = r
    if not items: st.info("Nothing to export yet.")
    for n, p in sorted(items.items()):
        st.download_button(f"Download {n}", p.read_bytes(), file_name=n, key=n)
    st.caption("Reports: evaluation JSON/CSV/PNG, dataset analysis, tuning table; plus per-run experiment configs (incl. seed, manifest hash).")
