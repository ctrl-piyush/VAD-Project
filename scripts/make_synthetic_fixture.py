"""
Creates a TINY SYNTHETIC dataset in UCF-Crime layout, ONLY to test that the code runs end to end.
Anomalous videos contain a bright flashing block in a known frame range. Results on it say nothing about UCF-Crime.

  python scripts/make_synthetic_fixture.py --out /tmp/vad_fixture
  export VAD_DATA_ROOT=/tmp/vad_fixture/ucf_crime VAD_ARTIFACTS=/tmp/vad_fixture/artifacts
"""
import argparse
from pathlib import Path
import cv2
import numpy as np

CLASSES = ["Abuse", "Arrest", "Arson", "Assault", "Burglary", "Explosion", "Fighting",
           "RoadAccidents", "Robbery", "Shooting", "Shoplifting", "Stealing", "Vandalism"]


def write_video(path, n_frames, anomaly, rng, size=64, fps=25):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (size, size))
    base = rng.randint(60, 120)
    for f in range(n_frames):
        img = np.full((size, size, 3), base, np.uint8)
        img += rng.randint(0, 12, img.shape, dtype=np.uint8)
        if anomaly and anomaly[0] <= f < anomaly[1]:
            x = rng.randint(0, size - 24); y = rng.randint(0, size - 24)
            img[y:y + 24, x:x + 24] = 255 if f % 2 == 0 else 0
        vw.write(img)
    vw.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--per_class_train", type=int, default=3)
    ap.add_argument("--per_class_test", type=int, default=1)
    a = ap.parse_args()
    rng = np.random.RandomState(0)
    root = Path(a.out) / "ucf_crime"
    (root / "videos").mkdir(parents=True, exist_ok=True)
    (root / "annotations").mkdir(exist_ok=True)
    train, test, temporal = [], [], []
    for split, per, lst in [("train", a.per_class_train, train), ("test", a.per_class_test, test)]:
        for c in CLASSES:
            for k in range(per):
                n = int(rng.randint(80, 140))
                s = int(rng.randint(16, n - 48)); e = s + 32
                name = f"{c}{k + 1 + (100 if split == 'test' else 0):03d}_x264.mp4"
                d = root / "videos" / c; d.mkdir(exist_ok=True)
                write_video(d / name, n, (s, e), rng)
                lst.append(f"{c}/{name}")
                if split == "test":
                    temporal.append(f"{name}  {c}  {s}  {e}  -1  -1")
        for k in range(len(CLASSES) * per):
            n = int(rng.randint(80, 140))
            name = f"Normal_Videos_{k + 1 + (900 if split == 'test' else 0):03d}_x264.mp4"
            d = root / "videos" / ("Testing_Normal_Videos_Anomaly" if split == "test" else "Training_Normal_Videos_Anomaly")
            d.mkdir(exist_ok=True)
            write_video(d / name, n, None, rng)
            lst.append(f"{d.name}/{name}")
            if split == "test":
                temporal.append(f"{name}  Normal  -1  -1  -1  -1")
    (root / "annotations" / "Anomaly_Train.txt").write_text("\n".join(train) + "\n")
    (root / "annotations" / "Anomaly_Test.txt").write_text("\n".join(test) + "\n")
    (root / "annotations" / "Temporal_Anomaly_Annotation_for_Testing_Videos.txt").write_text("\n".join(temporal) + "\n")
    print("fixture written to", root, "| train", len(train), "test", len(test))


if __name__ == "__main__":
    main()
