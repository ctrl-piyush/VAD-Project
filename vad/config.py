from __future__ import annotations
import copy, os
from pathlib import Path
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def _set_dotted(cfg: dict, key: str, value):
    cur = cfg
    parts = key.split(".")
    for p in parts[:-1]:
        cur = cur[p]
    old = cur.get(parts[-1])
    if isinstance(old, bool):
        value = str(value).lower() in ("1", "true", "yes")
    elif isinstance(old, int):
        value = int(value)
    elif isinstance(old, float):
        value = float(value)
    cur[parts[-1]] = value


def _merge(a, b):
    out = dict(a)
    for k, v in b.items():
        out[k] = _merge(a[k], v) if isinstance(v, dict) and isinstance(a.get(k), dict) else v
    return out


def load_config(path="configs/default.yaml", overrides=None) -> dict:
    if str(path) == "configs/default.yaml" and os.environ.get("VAD_CONFIG"):
        path = os.environ["VAD_CONFIG"]                  # e.g. export VAD_CONFIG=configs/kaggle_frames.yaml
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    with open(p, "r") as f:
        cfg = yaml.safe_load(f)
    if "base" in cfg:                                   # e.g. base: default.yaml  (deep-merged under this file)
        base = load_config(p.parent / cfg.pop("base"))
        for k in ("root_abs", "artifacts_abs"):
            base["data" if k == "root_abs" else "project"].pop(k, None)
        cfg = _merge(base, cfg)
    for o in overrides or []:
        k, v = o.split("=", 1)
        _set_dotted(cfg, k.strip(), v.strip())
    root = Path(os.environ.get("VAD_DATA_ROOT", cfg["data"]["root"]))
    cfg["data"]["root_abs"] = str(root if root.is_absolute() else REPO_ROOT / root)
    art = Path(os.environ.get("VAD_ARTIFACTS", cfg["project"]["artifacts_dir"]))
    cfg["project"]["artifacts_abs"] = str(art if art.is_absolute() else REPO_ROOT / art)
    return copy.deepcopy(cfg)


def paths(cfg: dict) -> dict:
    a = Path(cfg["project"]["artifacts_abs"])
    bb = cfg["preprocess"]["backbone"]
    return dict(artifacts=a, dataset=a / "dataset", manifest=a / "dataset" / "manifest.csv",
                features=a / "features" / bb, runs=a / "runs", logs=a / "logs", reports=a / "reports")
