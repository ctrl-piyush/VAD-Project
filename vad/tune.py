"""
Validation-driven experiment table. Runs each experiment in configs/tuning.yaml (skips finished ones).
  python -m vad.tune [--only tf_default tf_lr3e-4] [--extra-set key=val ...]
Never touches the test split. Pick the winner by val AUC (look at the table, mind the noise: one seed!), then run
final multi-seed training with that config.
"""
from __future__ import annotations
import argparse, json, subprocess, sys
import pandas as pd, yaml
from vad.config import load_config, paths, REPO_ROOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml"); ap.add_argument("--tuning", default="configs/tuning.yaml")
    ap.add_argument("--only", nargs="*"); ap.add_argument("--extra-set", nargs="*", default=[])
    a = ap.parse_args()
    cfg = load_config(a.config, a.extra_set); P = paths(cfg); P["reports"].mkdir(parents=True, exist_ok=True)
    T = yaml.safe_load(open(REPO_ROOT / a.tuning))
    rows = []
    for e in T["experiments"]:
        if a.only and e["name"] not in a.only: continue
        run = f"tune_{e['name']}"
        summ = P["runs"] / run / "summary.json"
        if not summ.exists():
            cmd = [sys.executable, "-m", "vad.train", "--config", a.config, "--run-name", run, "--seed", str(T["seed"]),
                   "--set", *a.extra_set, *e["set"]] if (a.extra_set or e["set"]) else \
                  [sys.executable, "-m", "vad.train", "--config", a.config, "--run-name", run, "--seed", str(T["seed"])]
            print(">>", " ".join(cmd)); subprocess.run(cmd, check=False, cwd=REPO_ROOT)
        if summ.exists():
            lg = [json.loads(l) for l in open(P["runs"] / run / "log.jsonl")]
            rows.append(dict(name=e["name"], overrides=" ".join(e["set"]), best_val_auc=json.load(open(summ))["best_val_auc"],
                             best_epoch=max(lg, key=lambda r: r["val_auc"])["epoch"], epochs_run=len(lg)))
    df = pd.DataFrame(rows).sort_values("best_val_auc", ascending=False)
    df.to_csv(P["reports"] / "tuning_table.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
