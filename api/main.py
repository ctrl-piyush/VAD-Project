"""
FastAPI backend (synopsis 4.7).   uvicorn api.main:app --port 8000
  GET  /health
  GET  /runs                          trained checkpoints
  POST /analyze?run=<name>            multipart video upload -> scores, intervals, calibrated confidence, routing
  GET  /report/{group}                evaluation JSON written by vad.evaluate
Streamlit uses it when you type the URL in the sidebar ("FastAPI backend"); otherwise Streamlit runs inference in-process.
"""
import json, os, shutil, tempfile
from pathlib import Path
from fastapi import FastAPI, File, HTTPException, UploadFile

from vad.config import load_config, paths

CFG_PATH = os.environ.get("VAD_CONFIG", "configs/default.yaml")
app = FastAPI(title="VAD API")
_cache = {}


def cfg():
    return load_config(CFG_PATH)


@app.get("/health")
def health():
    return dict(status="ok", config=CFG_PATH)


@app.get("/runs")
def runs():
    P = paths(cfg())
    return sorted(p.name for p in P["runs"].iterdir() if (p / "best.pt").exists()) if P["runs"].exists() else []


@app.post("/analyze")
async def analyze(run: str, video: UploadFile = File(...)):
    c = cfg(); P = paths(c)
    if not (P["runs"] / run / "best.pt").exists():
        raise HTTPException(404, f"no checkpoint for run '{run}'")
    if run not in _cache:
        from vad.infer import Analyzer
        _cache[run] = Analyzer(c, run)
    tmp = Path(tempfile.mkdtemp()) / (video.filename or "upload.mp4")
    with open(tmp, "wb") as f:
        shutil.copyfileobj(video.file, f)
    try:
        return _cache[run].analyze(tmp)
    except Exception as e:
        raise HTTPException(422, f"analysis failed: {e}")


@app.get("/report/{group}")
def report(group: str):
    p = paths(cfg())["reports"] / f"eval_{group}.json"
    if not p.exists():
        raise HTTPException(404, "no such evaluation report")
    return json.load(open(p))
