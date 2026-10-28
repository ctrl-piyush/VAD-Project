"""End-to-end on SYNTHETIC cached features (no videos / no backbone): manifest -> split -> train -> resume -> evaluate.
These numbers are NOT experimental results; they only prove the code path works."""
import json, os, subprocess, sys
from pathlib import Path
import numpy as np, pandas as pd

REPO = Path(__file__).resolve().parents[1]
CLASSES = ["Abuse", "Arson", "Fighting"]


def build(tmp):
    rng = np.random.RandomState(0)
    art = tmp / "art"; feats = art / "features" / "r3d_18"; feats.mkdir(parents=True); (art / "dataset").mkdir()
    ann_dir = tmp / "data" / "annotations"; ann_dir.mkdir(parents=True); (tmp / "data" / "videos").mkdir()
    rows, temporal = [], []
    for split, n in [("train", 20), ("test", 8)]:
        for k in range(n * 2):
            anom = k < n; cls = CLASSES[k % 3] if anom else "Normal"
            vid = f"{cls}{split[:2]}{k:03d}_x264"
            ns = int(rng.randint(40, 120)); f = rng.randn(ns, 32).astype(np.float16)
            s = int(rng.randint(5, ns - 25)); e = s + 20
            if anom: f[s:e] += 2.0
            np.save(feats / f"{vid}.npy", f)
            json.dump(dict(video_id=vid, n_snippets=ns, dim=32, fps=25.0, n_frames_read=ns * 16 + 3, clip_len=16, stride=16), open(feats / f"{vid}.json", "w"))
            rows.append(dict(video_id=vid, rel_path=f"x/{vid}.mp4", split=split, cls=cls, label=int(anom), resolved_path="", status="ok",
                             readable=True, n_frames=ns * 16 + 3, fps=25.0, duration_s=(ns * 16 + 3) / 25))
            if split == "test": temporal.append(f"{vid}.mp4 {cls} {s*16} {e*16} -1 -1" if anom else f"{vid}.mp4 Normal -1 -1 -1 -1")
    pd.DataFrame(rows).to_csv(art / "dataset" / "manifest.csv", index=False)
    (ann_dir / "Temporal_Anomaly_Annotation_for_Testing_Videos.txt").write_text("\n".join(temporal))
    return art


def run(args, env):
    r = subprocess.run([sys.executable, "-m", *args], cwd=REPO, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:] + r.stdout[-1000:]
    return r.stdout


def test_end_to_end(tmp_path):
    art = build(tmp_path)
    env = dict(os.environ, VAD_DATA_ROOT=str(tmp_path / "data"), VAD_ARTIFACTS=str(art))
    small = ["train.epochs=10", "train.lr=0.001", "train.iters_per_epoch=8", "train.batch_size=8", "train.warmup_epochs=1", "model.d_model=32", "model.n_heads=2", "eval.bootstrap=50"]
    run(["vad.data.make_val_split"], env)
    for s in (1, 2):
        run(["vad.train", "--run-name", f"tf{s}", "--seed", str(s), "--set", *small], env)
    run(["vad.train", "--run-name", "b1", "--seed", "1", "--set", *small, "model.type=baseline"], env)
    run(["vad.train", "--run-name", "b2", "--seed", "2", "--set", *small, "model.type=baseline"], env)
    # resume: extend epochs, must continue from the saved epoch
    out = run(["vad.train", "--run-name", "tf1", "--seed", "1", "--resume", "--set", *small, "train.epochs=12"], env)
    assert "Resumed from epoch 10" in out
    run(["vad.evaluate", "--group", "g", "--runs", "tf1", "tf2", "--vs", "b=b1,b2", "--set", *small], env)
    r = json.load(open(art / "reports" / "eval_g.json"))
    assert r["ensemble"]["video_auc"] > 0.9            # strong synthetic signal
    assert r["ensemble"]["frame_auc"] > 0.7
    assert r["bootstrap_95ci"]["video_auc"]["lo"] <= r["ensemble"]["video_auc"] <= r["bootstrap_95ci"]["video_auc"]["hi"] + 1e-9
    assert "holm_adjusted_p" in r["comparisons"]["g_minus_b:video_auc"] and "Abuse" in r["per_category"] and len(r["threshold_sweep"]) > 5
    # test split is not used for checkpoint selection: val-only log
    log = [json.loads(l) for l in open(art / "runs" / "tf1" / "log.jsonl")]
    assert all("test" not in k for rec in log for k in rec)
