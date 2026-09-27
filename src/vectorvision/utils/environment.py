"""Stage 1a — verify the machine can run every later stage."""
from __future__ import annotations

import importlib
import platform
import shutil
import sys

from .config import ROOT, get_device

# (import name, pip name, needed from stage)
PACKAGES = [
    ("numpy", "numpy", 1), ("pandas", "pandas", 1), ("yaml", "pyyaml", 1),
    ("PIL", "pillow", 1), ("tifffile", "tifffile", 1), ("tqdm", "tqdm", 2),
    ("matplotlib", "matplotlib", 1), ("torch", "torch", 2), ("torchvision", "torchvision", 2),
    ("sklearn", "scikit-learn", 4), ("shap", "shap", 5), ("joblib", "joblib", 4),
    ("cv2", "opencv-python", 3), ("ee", "earthengine-api", 4), ("rasterio", "rasterio", 4),
    ("shapely", "shapely", 4), ("pymavlink", "pymavlink", 7), ("fastapi", "fastapi", 8),
    ("uvicorn", "uvicorn", 8),
]


def _line(status: str, label: str, detail: str = "") -> dict:
    mark = {"PASS": "[ OK ]", "WARN": "[WARN]", "FAIL": "[FAIL]"}[status]
    print(f"  {mark} {label:<28} {detail}")
    return {"status": status, "check": label, "detail": detail}


def check_environment(cfg: dict, check_ee: bool = True) -> list[dict]:
    out = []
    print("\nEnvironment")
    v = sys.version_info
    ok = (3, 10) <= (v.major, v.minor) <= (3, 12)
    out.append(_line("PASS" if ok else "WARN", "Python",
                     f"{platform.python_version()}"
                     + ("" if ok else "  (3.10-3.12 recommended; some geo packages lag newer versions)")))

    print("\nPackages")
    for mod, pipname, stage in PACKAGES:
        try:
            m = importlib.import_module(mod)
            ver = getattr(m, "__version__", "installed")
            out.append(_line("PASS", pipname, str(ver)))
        except Exception:
            status = "FAIL" if stage <= 2 else "WARN"
            out.append(_line(status, pipname, f"missing (needed from stage {stage}) -> pip install {pipname}"))

    print("\nCompute")
    dev = get_device(cfg["project"].get("device", "auto"))
    try:
        import torch
        if dev == "cuda":
            name = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            out.append(_line("PASS", "GPU", f"{name}, {mem:.1f} GB"))
        elif dev == "mps":
            out.append(_line("PASS", "GPU", "Apple MPS"))
        else:
            out.append(_line("WARN", "GPU", "none - training will be slow; run training stages on Colab"))
    except ImportError:
        out.append(_line("FAIL", "GPU", "torch not installed"))

    free = shutil.disk_usage(ROOT).free / 1e9
    out.append(_line("PASS" if free > 20 else "WARN", "Free disk", f"{free:.0f} GB (need ~20 GB for all datasets)"))

    if check_ee and cfg["earth_engine"].get("enabled", True):
        print("\nEarth Engine")
        try:
            import ee
            ee.Initialize(project=cfg["earth_engine"]["project"])
            val = ee.Number(1).add(1).getInfo()
            out.append(_line("PASS", "Earth Engine", f"project {cfg['earth_engine']['project']} (test = {val})"))
        except Exception as e:
            out.append(_line("WARN", "Earth Engine",
                             f"not authenticated: run  earthengine authenticate  ({str(e)[:70]})"))
    return out
