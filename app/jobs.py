"""Background job runner for the dashboard: real subprocesses, logs on disk, no fake progress."""
from __future__ import annotations
import subprocess, sys, time, json
from pathlib import Path
from vad.config import REPO_ROOT, paths


def _meta(logdir: Path, name: str) -> Path:
    return logdir / f"{name}.job.json"


def start(cfg, name: str, module_args: list[str], env_extra=None) -> dict:
    import os
    P = paths(cfg); P["logs"].mkdir(parents=True, exist_ok=True)
    log = P["logs"] / f"{name}.log"
    env = dict(os.environ, **(env_extra or {}))
    env["PYTHONUNBUFFERED"] = "1"
    f = open(log, "a")
    f.write(f"\n=== {time.ctime()} :: {' '.join(module_args)}\n"); f.flush()
    p = subprocess.Popen([sys.executable, "-m", *module_args], cwd=REPO_ROOT, stdout=f, stderr=subprocess.STDOUT, env=env)
    info = dict(name=name, pid=p.pid, log=str(log), args=module_args, started=time.time())
    json.dump(info, open(_meta(P["logs"], name), "w"))
    return info


def status(cfg, name: str) -> dict:
    P = paths(cfg); m = _meta(P["logs"], name)
    if not m.exists():
        return dict(state="never started")
    info = json.load(open(m))
    try:
        import psutil
        alive = psutil.pid_exists(info["pid"]) and psutil.Process(info["pid"]).status() != psutil.STATUS_ZOMBIE
    except Exception:
        alive = False
    info["state"] = "running" if alive else "finished/stopped"
    return info


def stop(cfg, name: str) -> bool:
    import psutil
    info = status(cfg, name)
    if info.get("state") == "running":
        psutil.Process(info["pid"]).terminate(); return True
    return False


def tail(path, n=40) -> str:
    p = Path(path)
    if not p.exists(): return "(no log yet)"
    return "".join(p.read_text(errors="replace").splitlines(keepends=True)[-n:])
