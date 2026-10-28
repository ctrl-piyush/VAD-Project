"""Frame-dataset adapter (Kaggle copy) tests on SYNTHETIC data."""
import json, os, subprocess, sys
from pathlib import Path
import cv2, numpy as np, pandas as pd, yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from vad.data.features import clips_from_video, clips_from_frames, preprocess_frame
from vad.data.index_frames import parse_name


def make_video(path, n, size=48, seed=0):
    rng = np.random.RandomState(seed)
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25, (size, size))
    for f in range(n):
        vw.write(rng.randint(0, 255, (size, size, 3), dtype=np.uint8))
    vw.release()


def test_parse_name():
    assert parse_name("Abuse001_x264_10") == ("Abuse001_x264", 10)
    assert parse_name("Normal_Videos_003_x264_1250") == ("Normal_Videos_003_x264", 1250)
    assert parse_name("noindex") is None


def test_video_clips_match_reference_and_frame_source(tmp_path):
    v = tmp_path / "v.mp4"; make_video(v, 120)
    cap = cv2.VideoCapture(str(v)); allf = []
    while True:
        ok, fr = cap.read()
        if not ok: break
        allf.append(fr)
    # (1) video clips == manual slicing of decoded frames
    got = np.concatenate([np.stack(c) for c, m in clips_from_video(str(v), 16, 16, 32, 32) if len(c)])
    ref = np.stack([np.stack([preprocess_frame(f, 32, 32) for f in allf[s:s + 16]]).transpose(3, 0, 1, 2) for s in range(0, 120 - 16 + 1, 16)])
    assert got.shape == ref.shape and np.allclose(got, ref, atol=1e-6)
    # (2) video with frame_every=10 + pre_resize == image-folder source built the same way (PNG is lossless)
    sub = allf[::10]; paths = []
    for i, f in enumerate(sub):
        p = tmp_path / f"v_{i*10}.png"; cv2.imwrite(str(p), cv2.resize(f, (24, 24), interpolation=cv2.INTER_AREA)); paths.append(str(p))
    a = [(c, m) for c, m in clips_from_video(str(v), 4, 4, 32, 32, frame_every=10, pre_resize=24)]
    b = [(c, m) for c, m in clips_from_frames(paths, 4, 4, 32, 32, fps=2.5)]
    A, B = np.concatenate([np.stack(c) for c, m in a if len(c)]), np.concatenate([np.stack(c) for c, m in b if len(c)])
    assert A.shape == B.shape and np.allclose(A, B, atol=1e-6)
    assert a[-1][1]["n_frames_read"] == b[-1][1]["n_frames_read"] == len(sub)
    assert abs(a[-1][1]["fps"] - 2.5) < 1e-6                      # 25 fps / every 10


def test_end_to_end_frames_pipeline(tmp_path):
    fx = tmp_path / "fx"
    subprocess.run([sys.executable, str(REPO / "scripts" / "make_synthetic_frames_fixture.py"), "--out", str(fx)], check=True, capture_output=True)
    ann_dir = tmp_path / "data" / "annotations"; ann_dir.mkdir(parents=True)
    (ann_dir / "Temporal_Anomaly_Annotation_for_Testing_Videos.txt").write_text((fx / "Temporal_Anomaly_Annotation_for_Testing_Videos.txt").read_text())
    art = tmp_path / "art"
    env = dict(os.environ, VAD_DATA_ROOT=str(tmp_path / "data"), VAD_ARTIFACTS=str(art))
    small = ["preprocess.weights=none", "preprocess.resize=56", "preprocess.crop=56", "preprocess.fp16=false", "train.epochs=6", "train.iters_per_epoch=6",
             "train.batch_size=8", "train.lr=0.001", "train.warmup_epochs=1", "model.d_model=32", "model.n_heads=2", "eval.bootstrap=30"]
    cfgf = ["--config", "configs/kaggle_frames.yaml"]

    def run(args, ok=True):
        r = subprocess.run([sys.executable, "-m", *args], cwd=REPO, env=env, capture_output=True, text=True)
        assert (r.returncode == 0) == ok, r.stderr[-1500:] + r.stdout[-800:]
        return r.stdout
    out = run(["vad.data.index_frames", *cfgf, "--root", str(fx)], ok=True)
    rep = json.load(open(art / "dataset" / "validation_report.json"))
    assert rep["counts"]["train"]["videos"] == 13 * 2 * 2 and rep["counts"]["test"]["anomalous"] == 13 and not rep["blocking_problem_types"]
    assert rep["warnings"]                                       # synthetic counts differ from the official 1610/290 -> must warn
    run(["vad.data.make_val_split", *cfgf])
    run(["vad.data.features", *cfgf, "--set", *small])
    run(["vad.data.features", *cfgf, "--set", *small])           # resume: nothing new
    assert json.load(open(art / "reports" / "extraction_report.json"))["processed"] == 0
    for s in (1, 2):
        run(["vad.train", *cfgf, "--run-name", f"tf{s}", "--seed", str(s), "--set", *small])
    run(["vad.evaluate", *cfgf, "--group", "g", "--runs", "tf1", "tf2", "--set", *small])
    r = json.load(open(art / "reports" / "eval_g.json"))
    assert r["ensemble"]["n_videos_without_frame_gt"] == 0 and r["ensemble"]["frame_auc"] == r["ensemble"]["frame_auc"]
    assert r["temporal_iou"]["n"] == 13


def test_gt_divisor_and_stem_keys(tmp_path, monkeypatch):
    from vad.config import load_config
    from vad.evaluate import get_annotations
    d = tmp_path / "d" / "annotations"; d.mkdir(parents=True)
    (d / "Temporal_Anomaly_Annotation_for_Testing_Videos.txt").write_text("Abuse028_x264.mp4  Abuse  165  240  -1  -1\nNormal_Videos_001_x264.mp4 Normal -1 -1 -1 -1\n")
    monkeypatch.setenv("VAD_DATA_ROOT", str(tmp_path / "d"))
    ann = get_annotations(load_config("configs/kaggle_frames.yaml"))
    assert ann["Abuse028_x264"] == [(16, 24)] and ann["Abuse028_x264.mp4"] == [(16, 24)] and ann["Normal_Videos_001_x264"] == []
    assert get_annotations(load_config("configs/default.yaml"))["Abuse028_x264"] == [(165, 240)]
    monkeypatch.setenv("VAD_CONFIG", "configs/kaggle_frames.yaml")
    assert load_config()["data"]["source"] == "frames"
