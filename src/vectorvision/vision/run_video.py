"""Stage 7 runner — connects the trained detector to real video files."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .video import (detect_sequence, evaluate_filter, ground_sample_distance)


def load_detector(model_dir: Path, dev: str):
    """Return (predict_fn, input_size). predict_fn maps uint8 RGB frames -> water probability."""
    import cv2
    import torch

    from ..satellite.unet import UNet
    ck_p = model_dir / "best.pt"
    if not ck_p.exists():
        raise SystemExit(f"No trained detector at {ck_p}. Run:  python main.py train-drone")
    ck = torch.load(ck_p, map_location=dev, weights_only=False)
    size = int(ck["config"]["image_size"])
    model = UNet(2, ck["config"]["base_width"], 3).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"detector from epoch {ck['epoch']+1}, {ck['params']:,} parameters, {size}px input")

    @torch.no_grad()
    def predict_fn(batch: np.ndarray) -> np.ndarray:
        out = []
        for img in batch:
            if img.shape[0] != size or img.shape[1] != size:
                img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
            out.append(img)
        x = torch.from_numpy(np.stack(out).astype(np.float32) / 255.0)
        x = x.permute(0, 3, 1, 2).to(dev)
        with torch.autocast(device_type="cuda", enabled=dev == "cuda"):
            prob = torch.softmax(model(x).float(), 1)[:, 1]
        return prob.cpu().numpy()

    return predict_fn, size, ck


def read_video(path: Path, every_n: int, max_frames: int, size: int):
    """Yield (frame_index, RGB frame) from an MP4, sampling every Nth frame."""
    import cv2
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open {path}. MP4 with H.264 works best.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or size
    frames, idx, kept = [], 0, 0
    while kept < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % every_n == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append((idx, cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)))
            kept += 1
        idx += 1
    cap.release()
    return frames, fps, width


def run_video(cfg: dict, video: Path, model_dir: Path, outputs: Path, dev: str) -> dict:
    v = cfg["video"]
    predict_fn, size, _ = load_detector(model_dir, dev)
    frames, fps, width = read_video(video, int(v["every_n"]), int(v["max_frames"]), size)
    if not frames:
        raise SystemExit("No frames were read from the video.")
    gsd = ground_sample_distance(float(v["altitude_m"]), float(v["hfov_deg"]), size)
    print(f"{len(frames)} frames sampled (every {v['every_n']}th of {fps:.0f} fps), "
          f"ground scale {gsd*100:.0f} cm per pixel at {v['altitude_m']} m")

    res = detect_sequence(frames, predict_fn, v, gsd, fps)
    res["video"] = str(video)
    res["altitude_m"] = float(v["altitude_m"])
    (outputs / "metrics").mkdir(parents=True, exist_ok=True)
    (outputs / "metrics" / "stage7_video.json").write_text(json.dumps(res, indent=2))
    _overlay(frames, predict_fn, res, float(v["prob_threshold"]), outputs / "graphs")
    return res


def run_simulation(cfg: dict, cache: Path, model_dir: Path, outputs: Path, dev: str,
                   n_images: int, n_steps: int) -> dict:
    """Measure the temporal filter on held-out FloodNet images."""
    v = cfg["video"]
    xt = cache / "X_test.npy"
    if not xt.exists():
        raise SystemExit("No FloodNet test cache. Run:  python main.py train-drone --cache-only")
    X, Y = np.load(xt), np.load(cache / "Y_test.npy")
    predict_fn, size, _ = load_detector(model_dir, dev)
    rep = evaluate_filter(X, Y, predict_fn, v, n_images=n_images, n_steps=n_steps,
                          seed=cfg["project"]["seed"])
    (outputs / "metrics").mkdir(parents=True, exist_ok=True)
    (outputs / "metrics" / "stage7_filter_eval.json").write_text(json.dumps(rep, indent=2))
    return rep


def _overlay(frames, predict_fn, res, thr: float, gdir: Path, n: int = 6):
    """Save a strip of frames with the detected water shaded."""
    try:
        import cv2
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    water = [i for i, f in enumerate(res["frames"]) if f["water"]]
    pick = water[:: max(1, len(water) // n)][:n] if water else list(range(min(n, len(frames))))
    if not pick:
        return
    gdir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, len(pick), figsize=(2.6 * len(pick), 3.0))
    ax = np.atleast_1d(ax)
    for a, i in zip(ax, pick):
        idx, img = frames[i]
        prob = predict_fn(img[None])[0]
        m = cv2.resize((prob >= thr).astype(np.uint8), (img.shape[1], img.shape[0]),
                       interpolation=cv2.INTER_NEAREST)
        vis = img.copy()
        vis[m > 0] = (0.45 * vis[m > 0] + 0.55 * np.array([20, 130, 145])).astype(np.uint8)
        a.imshow(vis)
        a.set_title(f"{res['frames'][i]['t_s']}s", fontsize=9)
        a.axis("off")
    fig.suptitle("Detected standing water (teal)", fontsize=10)
    fig.tight_layout()
    fig.savefig(gdir / "stage7_frames.png", dpi=130)
    plt.close(fig)
