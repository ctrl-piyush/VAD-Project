"""
Train baseline / CNN-Transformer MIL on cached features. Selection uses VALIDATION video-level AUC only.

  python -m vad.train --run-name tf_s42 --seed 42
  python -m vad.train --run-name base_s42 --seed 42 --set model.type=baseline
  python -m vad.train --run-name tf_s42 --seed 42 --resume          # continue after interruption
Overrides: --set train.lr=3e-4 model.n_layers=1 ...
"""
from __future__ import annotations
import argparse, hashlib, json, os, random, subprocess, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from vad.config import load_config, paths
from vad.data.bags import Bags, FeatureNorm, balanced_batch_indices
from vad.metrics import video_level
from vad.models import build_model, mil_ranking_loss, video_scores, pick_device, count_params


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def git_rev():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def load_split_bags(cfg, split, norm=None):
    P = paths(cfg)
    df = pd.read_csv(P["manifest"])
    df = df[(df.split == split) & (df.status == "ok")]
    have = df.video_id.map(lambda v: (P["features"] / f"{v}.npy").exists() and (P["features"] / f"{v}.json").exists())
    if (~have).any():
        miss = df[~have].video_id.tolist()
        raise FileNotFoundError(f"{len(miss)} {split} videos have no cached features (e.g. {miss[:3]}). Run vad.data.features.")
    return Bags(df, P["features"], cfg["model"], norm)


@torch.no_grad()
def predict_bags(model, bags: Bags, device, bs=64):
    """Returns list of per-video segment-score arrays (length = pooled length)."""
    model.eval()
    out = []
    for i in range(0, len(bags), bs):
        idx = list(range(i, min(i + bs, len(bags))))
        X, M, _ = bags.collate(idx)
        lg = model(torch.from_numpy(X).to(device), torch.from_numpy(M).to(device))
        s = torch.sigmoid(lg).cpu().numpy()
        out += [s[k, :M[k].sum()] for k in range(len(idx))]
    return out


def val_metrics(model, bags, device):
    seg = predict_bags(model, bags, device)
    vs = np.array([s.max() for s in seg])
    return video_level(bags.labels, vs), vs


def save_ckpt(path, model, opt, sched, epoch, best, bad, norm, cfg, seed, extra=None):
    tmp = str(path) + ".tmp"
    torch.save(dict(model=model.state_dict(), opt=opt.state_dict() if opt else None,
                    sched=sched.state_dict() if sched else None, epoch=epoch, best=best, bad=bad,
                    norm=norm.state(), cfg=cfg, seed=seed, in_dim=model_in_dim(model), extra=extra or {}), tmp)
    os.replace(tmp, path)


def model_in_dim(model):
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            return m.in_features


def load_model(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ck["cfg"]["model"], ck["in_dim"]).to(device)
    model.load_state_dict(ck["model"])
    return model.eval(), FeatureNorm.from_state(ck["norm"]), ck["cfg"], ck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    seed = a.seed if a.seed is not None else cfg["project"]["seed"]
    tc, mc = cfg["train"], cfg["model"]
    P = paths(cfg)
    run = P["runs"] / a.run_name
    run.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    dev = pick_device(tc["device"])

    train = load_split_bags(cfg, "train")
    val = load_split_bags(cfg, "val")
    assert not (set(train.ids) & set(val.ids)), "train/val overlap"
    norm = FeatureNorm(mc["feature_norm"]).fit(np.concatenate([p for p in train.pooled])[None])   # TRAIN only
    train.set_norm(norm); val.set_norm(norm)

    model = build_model(mc, train.dim).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    total = tc["epochs"]; warm = tc["warmup_epochs"]
    def lr_at(ep):
        if ep < warm: return (ep + 1) / max(1, warm)
        if tc["scheduler"] == "cosine": return 0.5 * (1 + np.cos(np.pi * (ep - warm) / max(1, total - warm))) * 0.99 + 0.01
        return 1.0
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    scaler = torch.amp.GradScaler(enabled=tc["amp"] and dev.type == "cuda")

    start, best, bad = 0, -1.0, 0
    last = run / "last.pt"
    if a.resume and last.exists():
        ck = torch.load(last, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        start, best, bad = ck["epoch"] + 1, ck["best"], ck["bad"]
        print(f"Resumed from epoch {start}")
    elif a.resume:
        print("No last.pt found; starting fresh")

    manifest_hash = hashlib.md5(open(P["manifest"], "rb").read()).hexdigest()
    json.dump(dict(cfg=cfg, seed=seed, git=git_rev(), manifest_md5=manifest_hash, n_params=count_params(model),
                   device=str(dev), n_train=len(train), n_val=len(val), torch=torch.__version__),
              open(run / "config.json", "w"), indent=2, default=str)
    logf = open(run / "log.jsonl", "a")
    wb = None
    if tc.get("wandb"):
        try:
            import wandb
            wb = wandb.init(project=tc["wandb_project"], name=a.run_name, config=cfg, resume="allow", id=a.run_name.replace("/", "_"))
        except Exception as e:
            print(f"W&B disabled ({e!r}); continuing with local logs only.")
    print(f"{a.run_name}: {mc['type']} params={count_params(model):,} device={dev} train={len(train)} val={len(val)}")

    for ep in range(start, total):
        rng = np.random.RandomState(seed * 100003 + ep)       # per-epoch RNG => resume is reproducible
        model.train(); losses = []; t0 = time.time()
        for it in range(tc["iters_per_epoch"]):
            opt.zero_grad(set_to_none=True)
            for _ in range(tc["grad_accum"]):
                idx = balanced_batch_indices(train, tc["batch_size"], rng, tc["class_balanced"])
                X, M, y = train.collate(idx)
                X, M, y = torch.from_numpy(X).to(dev), torch.from_numpy(M).to(dev), torch.from_numpy(y).to(dev)
                with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=tc["amp"] and dev.type == "cuda"):
                    lg = model(X, M)
                loss, parts = mil_ranking_loss(lg.float(), M, y, tc["margin"], tc["lambda_smooth"], tc["lambda_sparse"])
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite loss at epoch {ep} it {it}")
                scaler.scale(loss / tc["grad_accum"]).backward()
            scaler.unscale_(opt)
            if tc["grad_clip"]:
                torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
            scaler.step(opt); scaler.update()
            losses.append(loss.item())
        sched.step()
        vm, _ = val_metrics(model, val, dev)
        rec = dict(epoch=ep, train_loss=float(np.mean(losses)), val_auc=vm["auc"], val_ap=vm["ap"],
                   lr=opt.param_groups[0]["lr"], sec=round(time.time() - t0, 1))
        logf.write(json.dumps(rec) + "\n"); logf.flush()
        if wb is not None: wb.log({k: v for k, v in rec.items() if k != 'epoch'}, step=ep)
        print(rec)
        if vm["auc"] > best:
            best, bad = vm["auc"], 0
            save_ckpt(run / "best.pt", model, None, None, ep, best, bad, norm, cfg, seed)
        else:
            bad += 1
        save_ckpt(last, model, opt, sched, ep, best, bad, norm, cfg, seed)
        if bad >= tc["patience"]:
            print(f"Early stop at epoch {ep} (best val AUC {best:.4f})")
            break
    json.dump(dict(best_val_auc=best), open(run / "summary.json", "w"))
    if wb is not None: wb.summary["best_val_auc"] = best; wb.finish()
    print(f"Done. best val video-AUC={best:.4f}  -> {run/'best.pt'}")


if __name__ == "__main__":
    main()
