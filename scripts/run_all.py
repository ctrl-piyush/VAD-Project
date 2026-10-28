"""
ONE command for the whole study. Every stage is resumable/skippable, so an interruption never costs you finished work.

  python scripts/run_all.py --seeds 1 2 3                       # everything (hours: dominated by feature extraction)
  python scripts/run_all.py --seeds 1 2 3 --no-tune             # use default hyper-parameters (faster)
  python scripts/run_all.py --seeds 1 2 3 --with-explanations   # also build RAG store + explanation CSV for labelling
  python scripts/run_all.py --stages validate split features    # run only some stages
Stages (in order): validate split features clipfeatures analysis tune final evaluate explain report
"""
import argparse, json, subprocess, sys, time
from pathlib import Path
import pandas as pd, yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from vad.config import load_config, paths

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/default.yaml")
ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
ap.add_argument("--stages", nargs="*", default=None)
ap.add_argument("--frames-root", default=None, help="Kaggle frame dataset folder (contains Train/ and Test/); use with --config configs/kaggle_frames.yaml")
ap.add_argument("--no-tune", action="store_true"); ap.add_argument("--skip-clip", action="store_true")
ap.add_argument("--with-explanations", action="store_true"); ap.add_argument("--explain-n", type=int, default=50)
ap.add_argument("--set", nargs="*", default=[], help="global overrides, e.g. preprocess.batch_clips=4")
a = ap.parse_args()
ALL = ["validate", "split", "features", "clipfeatures", "analysis", "tune", "final", "evaluate", "explain", "report"]
stages = a.stages or ALL
cfg = load_config(a.config, a.set); P = paths(cfg)
G = ["--config", a.config] + (["--set", *a.set] if a.set else [])


def sh(mod_args, extra_set=None, allow_fail=False):
    cmd = [sys.executable, "-m", *mod_args, "--config", a.config]
    ov = [*a.set, *(extra_set or [])]
    if ov:
        cmd += ["--set", *ov]
    print("\n>>", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=REPO)
    if r.returncode != 0 and not allow_fail:
        sys.exit(f"STAGE FAILED: {' '.join(mod_args)} (fix the error, re-run the same command; finished work is kept)")
    return r.returncode


def best_tf_overrides():
    tt, tun = P["reports"] / "tuning_table.csv", yaml.safe_load(open(REPO / "configs/tuning.yaml"))
    if a.no_tune or not tt.exists():
        return [], "defaults"
    df = pd.read_csv(tt); df = df[df.name.str.startswith("tf_")]
    if df.empty: return [], "defaults"
    best = df.sort_values("best_val_auc", ascending=False).iloc[0]
    ov = [e["set"] for e in tun["experiments"] if e["name"] == best["name"]][0]
    return list(ov), f"{best['name']} (val AUC {best['best_val_auc']:.4f})"


t0 = time.time()
if "validate" in stages:
    if cfg["data"].get("source") == "frames":
        root = a.frames_root or cfg["data"].get("frames_root") or __import__("os").environ.get("VAD_FRAMES_ROOT")
        if not root: sys.exit("frames mode: pass --frames-root /path/to/folder_with_Train_and_Test")
        sh(["vad.data.index_frames", "--root", root])
    else:
        sh(["vad.data.validate"])
if "split" in stages: sh(["vad.data.make_val_split"])
if "features" in stages: sh(["vad.data.features"])
if "clipfeatures" in stages and not a.skip_clip:
    if sh(["vad.data.clip_features"], allow_fail=True) != 0:
        print("!! CLIP features failed (is `transformers` installed / internet available?). Continuing WITHOUT the Transformer+MIL(CLIP) ablation.")
        a.skip_clip = True
if "analysis" in stages: sh(["vad.analysis"], allow_fail=True)
if "tune" in stages and not a.no_tune:
    subprocess.run([sys.executable, "-m", "vad.tune", "--config", a.config, *(["--extra-set", *a.set] if a.set else [])], cwd=REPO, check=True)
if "final" in stages:
    ov, why = best_tf_overrides(); print(f"\n### Final config for the Transformer models: {why}; overrides={ov}")
    json.dump(dict(chosen=why, overrides=ov), open(P["reports"] / "final_config_choice.json", "w")) if P["reports"].exists() else None
    groups = [("base", ["model.type=baseline"] + [o for o in ov if o.startswith(("train.", "model.dropout"))]),
              ("tf", ov)]
    if not a.skip_clip: groups.append(("cliptf", ["preprocess.backbone=clip_vitb32", "model.type=transformer"] + ov))
    for name, extra in groups:
        for s in a.seeds:
            run = f"{name}_s{s}"
            if (P["runs"] / run / "summary.json").exists():
                print(f"skip {run} (finished)"); continue
            sh(["vad.train", "--run-name", run, "--seed", str(s), "--resume"], extra_set=extra)
        for s in a.seeds:
            if not (P["runs"] / f"{name}_s{s}" / "operating_point.json").exists():
                sh(["vad.infer", "fit", "--run", f"{name}_s{s}"], extra_set=extra)
if "evaluate" in stages:
    runs = lambda n: ",".join(f"{n}_s{s}" for s in a.seeds)
    vs = [f"base={runs('base')}"] + ([] if a.skip_clip else [f"cliptf={runs('cliptf')}"])
    sh(["vad.evaluate", "--group", "tf", "--runs", *[f"tf_s{s}" for s in a.seeds], "--vs", *vs])
if "explain" in stages and a.with_explanations:
    r0 = f"tf_s{a.seeds[0]}"
    sh(["vad.explain.build_index", "--run", r0], allow_fail=True)
    sh(["vad.explain.sample", "--run", r0, "--n", str(a.explain_n)], allow_fail=True)
    print("\nNEXT (manual): two annotators fill annotator1/annotator2 in artifacts/explanations/explanations_to_label.csv, then:\n"
          "  python -m vad.explain.evaluate_checker --csv artifacts/explanations/explanations_to_label.csv")
if "report" in stages:
    subprocess.run([sys.executable, "scripts/make_report.py", "--config", a.config], cwd=REPO)
print(f"\nDONE in {(time.time()-t0)/3600:.2f} h (this invocation).")
