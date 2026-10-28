"""Run: python scripts/check_hardware.py   -> paste the JSON back to me."""
import json, platform, shutil, sys

info = {"os": platform.platform(), "python": sys.version.split()[0]}
try:
    import psutil
    info["ram_total_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
    info["ram_available_gb"] = round(psutil.virtual_memory().available / 1e9, 1)
    info["cpu_cores"] = psutil.cpu_count(logical=True)
except ImportError:
    info["ram"] = "psutil not installed"
info["disk_free_gb_cwd"] = round(shutil.disk_usage(".").free / 1e9, 1)
try:
    import torch
    info["torch"] = torch.__version__
    info["cuda_available"] = torch.cuda.is_available()
    info["torch_cuda_build"] = torch.version.cuda
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        info["gpu"] = p.name
        info["vram_gb"] = round(p.total_memory / 1e9, 1)
        x = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
        info["gpu_matmul_ok"] = bool(torch.isfinite(x @ x).all().item())
    else:
        info["note"] = "No CUDA GPU visible to PyTorch; CPU-only paths will be used."
except ImportError:
    info["torch"] = "not installed"
try:
    import cv2
    info["opencv"] = cv2.__version__
except ImportError:
    info["opencv"] = "not installed"
print(json.dumps(info, indent=2))
