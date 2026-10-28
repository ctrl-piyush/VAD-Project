"""
Final multi-seed protocol AFTER you chose a config on validation (see vad.tune).
  python scripts/run_final.py --name tf --seeds 1 2 3 [--set train.lr=0.0003 ...]
  python scripts/run_final.py --name base --seeds 1 2 3 --set model.type=baseline
Then evaluate once:
  python -m vad.evaluate --group tf --runs tf_s1 tf_s2 tf_s3 --other base --other-runs base_s1 base_s2 base_s3
"""
import argparse, subprocess, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True); ap.add_argument("--seeds", nargs="+", type=int, required=True)
ap.add_argument("--set", nargs="*", default=[]); ap.add_argument("--config", default="configs/default.yaml")
a = ap.parse_args()
for s in a.seeds:
    cmd = [sys.executable, "-m", "vad.train", "--config", a.config, "--run-name", f"{a.name}_s{s}", "--seed", str(s), "--resume"]
    if a.set: cmd += ["--set", *a.set]
    print(">>", " ".join(cmd)); subprocess.run(cmd, cwd=REPO, check=True)
    subprocess.run([sys.executable, "-m", "vad.infer", "fit", "--config", a.config, "--run", f"{a.name}_s{s}"] + (["--set", *a.set] if a.set else []), cwd=REPO, check=True)
