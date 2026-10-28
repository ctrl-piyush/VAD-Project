"""
SYNTHETIC stand-in for the Kaggle frame dataset (odins0n/ucf-crime-dataset): every 10th frame, 64x64 PNG,
<root>/{Train,Test}/<Class>/<VideoName>_<originalFrameIndex>.png, plus a temporal annotation file in ORIGINAL frame numbers.
Only for testing the code path. Results on it mean nothing.
  python scripts/make_synthetic_frames_fixture.py --out /tmp/fx_frames
"""
import argparse
from pathlib import Path
import cv2, numpy as np
CLASSES = ["Abuse", "Arrest", "Arson", "Assault", "Burglary", "Explosion", "Fighting", "RoadAccidents", "Robbery", "Shooting", "Shoplifting", "Stealing", "Vandalism"]


def write_video_frames(folder, vid, n_orig, anomaly, rng, every=10):
    base = rng.randint(60, 120)
    for f in range(0, n_orig, every):
        img = np.full((64, 64, 3), base, np.uint8) + rng.randint(0, 12, (64, 64, 3), dtype=np.uint8)
        if anomaly and anomaly[0] <= f < anomaly[1]:
            x, y = rng.randint(0, 40, 2); img[y:y + 24, x:x + 24] = 255 if (f // every) % 2 == 0 else 0
        cv2.imwrite(str(folder / f"{vid}_{f}.png"), img)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True)
    ap.add_argument("--per_class_train", type=int, default=2); ap.add_argument("--per_class_test", type=int, default=1)
    a = ap.parse_args(); rng = np.random.RandomState(0)
    root = Path(a.out); ann = []
    for split, per in [("Train", a.per_class_train), ("Test", a.per_class_test)]:
        for c in CLASSES:
            d = root / split / c; d.mkdir(parents=True, exist_ok=True)
            for k in range(per):
                vid = f"{c}{k + 1 + (100 if split == 'Test' else 0):03d}_x264"
                n = int(rng.randint(450, 700)); s = int(rng.randint(100, n - 250)); e = s + 200
                write_video_frames(d, vid, n, (s, e), rng)
                if split == "Test": ann.append(f"{vid}.mp4  {c}  {s}  {e}  -1  -1")
        d = root / split / "NormalVideos"; d.mkdir(parents=True, exist_ok=True)
        for k in range(len(CLASSES) * per):
            vid = f"Normal_Videos_{k + 1 + (900 if split == 'Test' else 0):03d}_x264"
            write_video_frames(d, vid, int(rng.randint(450, 700)), None, rng)
            if split == "Test": ann.append(f"{vid}.mp4  Normal  -1  -1  -1  -1")
    (root / "Temporal_Anomaly_Annotation_for_Testing_Videos.txt").write_text("\n".join(ann) + "\n")
    print("written", root)


if __name__ == "__main__":
    main()
