import numpy as np, pandas as pd, pytest, torch
from vad.seq import resample, segment_bounds, segments_to_snippets, snippet_frame_ranges
from vad.metrics import frame_arrays, frame_level_auc, video_level, pick_threshold_f1, prf, temporal_iou
from vad.localize import extract_intervals, smooth
from vad import calibration as cal
from vad.stats import bootstrap_ci, paired_bootstrap_diff, holm
from vad.models import build_model, mil_ranking_loss
from vad.config import load_config
from vad.data.bags import FeatureNorm
from vad.data.make_val_split import make_val_split, check_no_overlap


def test_resample_shapes_and_mean():
    f = np.random.randn(100, 8).astype(np.float32)
    r = resample(f, 32); b = segment_bounds(100, 32)
    assert r.shape == (32, 8) and np.allclose(r[3], f[b[3]:b[4]].mean(0), atol=1e-5)
    assert resample(f[:10], 32).shape == (32, 8)
    assert resample(f[:1], 32).shape == (32, 8)


def test_inverse_mapping_roundtrip():
    s = np.arange(32.0)
    assert segments_to_snippets(s, 100).shape == (100,)
    assert (np.diff(segments_to_snippets(s, 100)) >= 0).all()
    assert segments_to_snippets(s, 32).tolist() == s.tolist()
    assert segments_to_snippets(s, 10).shape == (10,)


def test_frame_level_auc_matches_bruteforce():
    rng = np.random.RandomState(0)
    items, ys, ss = [], [], []
    for _ in range(12):
        n_snip = rng.randint(5, 40); n_frames = n_snip * 16 + rng.randint(0, 16)
        seg = rng.rand(32)
        gt = [(rng.randint(0, n_frames // 2), n_frames // 2 + rng.randint(1, n_frames // 2))] if rng.rand() < 0.6 else []
        items.append(frame_arrays(seg, n_snip, 16, n_frames, gt))
        sn = segments_to_snippets(seg, n_snip)
        for (a, b) in snippet_frame_ranges(n_snip, 16, n_frames):
            for fr in range(a, b):
                ss.append(sn[min(fr // 16, n_snip - 1)] if fr // 16 < n_snip else sn[-1])
                ys.append(int(any(s <= fr < e for s, e in gt)))
    from sklearn.metrics import roc_auc_score
    assert abs(frame_level_auc(items) - roc_auc_score(ys, ss)) < 1e-9


def test_frame_auc_perfect_and_chance():
    gt = [(160, 320)]
    seg = np.zeros(32); seg[10:20] = 1.0                      # n_snip=32 -> identity mapping; frames 160..320 = snippets 10..19
    it = frame_arrays(seg, 32, 16, 512, gt)
    assert frame_level_auc([it]) == 1.0
    rng = np.random.RandomState(1)
    its = [frame_arrays(rng.rand(32), 32, 16, 512, gt) for _ in range(200)]
    assert abs(frame_level_auc(its) - 0.5) < 0.05


def test_video_level_and_threshold():
    y = np.array([0, 0, 1, 1]); s = np.array([.1, .2, .8, .9])
    assert video_level(y, s)["auc"] == 1.0
    thr, f1 = pick_threshold_f1(y, s); assert f1 == 1.0 and 0.2 < thr <= 0.8
    assert prf(y, s, 0.5)["fpr"] == 0.0


def test_iou_and_intervals():
    assert temporal_iou([(0, 10)], [(5, 15)]) == pytest.approx(5 / 15)
    sc = np.zeros(50); sc[20:25] = 1.0
    iv = extract_intervals(sc, window=1, rel_threshold=0.5)
    assert iv == [(20, 25)]
    assert extract_intervals(np.zeros(10)) == []


def test_calibration():
    rng = np.random.RandomState(0)
    z = rng.randn(4000) * 3; p_true = cal.sigmoid(z / 3)                 # model overconfident by factor 3
    y = (rng.rand(4000) < p_true).astype(int)
    p_over = cal.sigmoid(z)
    sc = cal.fit_scaler(p_over, y)
    assert 2.0 < sc["T"] < 4.5
    assert cal.ece(cal.apply_scaler(p_over, sc), y) < cal.ece(p_over, y)
    cov, risk = cal.risk_coverage(p_over, y); assert risk[0] <= risk[-1] + 0.05
    assert cal.route(0.5) == "human_review" and cal.route(0.95) == "alert" and cal.route(0.9, 0.5) == "human_review"


def test_bootstrap_and_holm():
    x = np.random.RandomState(0).rand(100)
    ci = bootstrap_ci(lambda i: x[i].mean(), 100, 300)
    assert ci["lo"] < x.mean() < ci["hi"]
    d = paired_bootstrap_diff(lambda i: x[i].mean() + 0.3, lambda i: x[i].mean(), 100, 300)
    assert d["excludes_zero"] and d["lo"] > 0.29
    adj = holm([0.01, 0.04, 0.5]); assert adj[0][1] and not adj[2][1]


@pytest.mark.parametrize("typ", ["baseline", "transformer"])
def test_model_forward_backward_and_padding(typ):
    m = dict(load_config()["model"]); m["type"] = typ
    net = build_model(m, 64).eval()
    x = torch.randn(1, 20, 64); xp = torch.cat([x, torch.randn(1, 12, 64)], 1)
    mk = torch.zeros(1, 32, dtype=torch.bool); mk[:, :20] = True
    assert (net(x, torch.ones(1, 20, dtype=torch.bool)) - net(xp, mk)[:, :20]).abs().max() < 1e-4
    net.train()
    X = torch.randn(6, 32, 64); M = torch.ones(6, 32, dtype=torch.bool); y = torch.tensor([1, 1, 1, 0, 0, 0])
    loss, _ = mil_ranking_loss(net(X, M), M, y); loss.backward()
    assert torch.isfinite(loss) and any(p.grad is not None and p.grad.abs().sum() > 0 for p in net.parameters())


def test_loss_prefers_correct_ranking():
    M = torch.ones(2, 4, dtype=torch.bool); y = torch.tensor([1, 0])
    good = torch.tensor([[-5., 5, -5, -5], [-5., -5, -5, -5]]); bad = torch.tensor([[-5., -5, -5, -5], [5., 5, -5, -5]])
    assert mil_ranking_loss(good, M, y)[0] < mil_ranking_loss(bad, M, y)[0]


def test_norm_fit_train_only():
    tr = np.random.randn(1, 50, 4) * 2 + 5
    n = FeatureNorm("zscore").fit(tr); out = n(tr)
    assert abs(out.mean()) < 1e-3
    n2 = FeatureNorm.from_state(n.state()); assert np.allclose(n2(tr), out, atol=1e-5)
    assert np.allclose(np.linalg.norm(FeatureNorm("l2")(tr), axis=-1), 1, atol=1e-5)


def test_val_split_no_leakage_and_stratified():
    rows = []
    for c in ["A", "B"]:
        for i in range(20): rows.append(dict(video_id=f"{c}{i}", split="train", cls=c, label=1, status="ok"))
    for i in range(30): rows.append(dict(video_id=f"N{i}", split="train", cls="Normal", label=0, status="ok"))
    for i in range(10): rows.append(dict(video_id=f"T{i}", split="test", cls="A", label=1, status="ok"))
    df = pd.DataFrame(rows)
    out = make_val_split(df, 0.1, 7); check_no_overlap(out)
    assert (out[out.split == "test"].video_id == df[df.split == "test"].video_id).all()
    assert set(out[out.split == "val"].cls) == {"A", "B", "Normal"}
    assert out.equals(make_val_split(out, 0.1, 7))            # idempotent / deterministic
