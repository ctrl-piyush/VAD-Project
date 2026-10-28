import numpy as np, cv2
from vad.data.clip_features import extract_clip_video, snippet_frames


def make_video(path, n, size=32):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25, (size, size))
    for f in range(n):
        vw.write(np.full((size, size, 3), f % 256, np.uint8))
    vw.release()


def test_snippet_grid_matches_cnn_grid(tmp_path):
    p = tmp_path / "v.mp4"; make_video(p, 100)
    per, fps, n_read = snippet_frames(str(p), 16, 16)
    assert n_read == 100 and len(per) == (100 - 16) // 16 + 1 and all(len(t) == 3 for t in per)
    embs, meta = extract_clip_video(lambda fr: np.stack([np.full(8, float(f.mean())) for f in fr]), str(p))
    assert embs.shape == (len(per), 8) and meta["n_frames_read"] == 100
    assert embs[0, 0] < embs[-1, 0]                      # later snippets show later (brighter) frames


def test_short_video_gets_one_snippet(tmp_path):
    p = tmp_path / "s.mp4"; make_video(p, 10)
    embs, _ = extract_clip_video(lambda fr: np.ones((len(fr), 4)), str(p))
    assert embs.shape == (1, 4)
