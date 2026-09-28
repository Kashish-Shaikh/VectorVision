"""Stage 3 — test the satellite U-Net on the held-out TEST split.

The test split was never used for training or for choosing the best epoch, so this
is the only number that should go in the paper.

Results are reported two ways, because both matter:
  - raw labels:     the dataset exactly as published (water includes speckle)
  - cleaned labels: water blobs smaller than min_water_blob_px removed, the same
                    rule used in training. This measures real water bodies.
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np

from . import metrics as M
from .cache import _mask_to_index, _pairs, _remove_speckle
from .inspect_sen2lulc import discover, read_any


def _test_cache(root: Path, cache: Path, meta: dict, n: int, seed: int):
    xp, yp = cache / f"X_test_{n}.npy", cache / f"Y_test_raw_{n}.npy"
    if xp.exists() and yp.exists():
        print(f"test cache found ({n} tiles)")
        return np.load(xp), np.load(yp)
    from tqdm.auto import tqdm
    t0 = time.time()
    print("listing test files (done once)...")
    pairs = _pairs(discover(root)["layout"], "test")
    if not pairs:
        raise SystemExit("No test split found. Check Stage 1 lists a 'test' split.")
    sub = random.Random(seed).sample(pairs, min(n, len(pairs)))
    lut = np.zeros(max(meta["raw_ids"]) + 1, np.uint8)
    for i, r in enumerate(meta["raw_ids"]):
        lut[r] = i
    H, W = read_any(sub[0][0]).shape[:2]
    X = np.zeros((len(sub), H, W, 3), np.uint8)
    Y = np.zeros((len(sub), H, W), np.uint8)
    for i, (ip, mp) in enumerate(tqdm(sub, desc="test tiles", mininterval=5)):
        x = read_any(ip)
        if x.ndim == 2:
            x = np.stack([x] * 3, -1)
        X[i] = x[..., :3]
        Y[i] = _mask_to_index(read_any(mp), lut)
    np.save(xp, X); np.save(yp, Y)
    print(f"  {len(sub)} of {len(pairs)} test tiles cached in {time.time()-t0:.0f}s")
    return X, Y


def evaluate(cfg: dict, root: Path, outputs: Path, model_dir: Path, dev: str, n_tiles: int) -> dict:
    import torch
    from .unet import UNet

    ck = torch.load(model_dir / "best.pt", map_location=dev, weights_only=False)
    meta = ck["meta"]
    n_cls, water, fill, mb = meta["n_classes"], meta["water_index"], meta["fill_index"], meta["min_water_blob_px"]
    model = UNet(n_cls, ck["config"]["base_width"], ck["config"]["in_channels"]).to(dev)
    model.load_state_dict(ck["model"]); model.eval()
    print(f"model from epoch {ck['epoch']+1}, {ck['params']:,} parameters")

    X, Y_raw = _test_cache(root, outputs / "cache", meta, n_tiles, cfg["project"]["seed"] + 1)
    Y_clean = _remove_speckle(Y_raw, water, fill, mb)

    pred = np.zeros_like(Y_raw)
    wprob = np.zeros(Y_raw.shape, np.float16)
    with torch.no_grad():
        for i in range(0, len(X), 256):
            x = torch.from_numpy(X[i:i + 256]).permute(0, 3, 1, 2).float().div(255).to(dev)
            with torch.autocast(device_type="cuda", enabled=dev == "cuda"):
                p = torch.softmax(model(x).float(), 1)
            pred[i:i + 256] = p.argmax(1).cpu().numpy()
            wprob[i:i + 256] = p[:, water].cpu().numpy().astype(np.float16)

    stage1 = json.loads((outputs / "metrics" / "stage1_dataset_report.json").read_text())
    names = []
    for k, c in enumerate(stage1["classes"]):
        names.append(f"raw {c['raw_id']}" + (" (water)" if k == water else ""))

    report = {"test_tiles": int(len(X)), "model_epoch": int(ck["epoch"] + 1), "params": int(ck["params"]),
              "class_names": names, "water_index": water, "min_water_blob_px": mb}
    for tag, Y in (("raw_labels", Y_raw), ("cleaned_labels", Y_clean)):
        inter, pc, tc = M.per_tile_stats(pred, Y, n_cls)
        iou, prec, rec, f1 = M.scores(inter, pc, tc)
        conf = M.confusion(pred, Y, n_cls)
        report[tag] = {
            "pixel_accuracy": float(np.trace(conf) / conf.sum()),
            "mean_iou": float(iou.mean()),
            "per_class": [{"class": names[k], "iou": float(iou[k]), "precision": float(prec[k]),
                           "recall": float(rec[k]), "f1": float(f1[k]),
                           "share_pct": float(100 * tc[:, k].sum() / tc.sum())} for k in range(n_cls)],
            "ci95": M.bootstrap_ci(inter, pc, tc, water),
            "confusion": conf.tolist(),
        }
    report["water_threshold_sweep"] = M.threshold_sweep(wprob.astype(np.float32), Y_clean == water)
    report["tile_water_detection"] = M.tile_detection(pred == water, Y_clean == water, mb)

    out = outputs / "metrics" / "stage3_test_report.json"
    out.write_text(json.dumps(report, indent=2))
    _figures(report, X, Y_clean, pred, water, outputs / "graphs")
    return report


def _figures(rep, X, Y, pred, water, gdir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    gdir.mkdir(parents=True, exist_ok=True)
    names = rep["class_names"]

    # per-class IoU, raw vs cleaned
    fig, ax = plt.subplots(figsize=(8, 4))
    k = np.arange(len(names))
    ax.bar(k - 0.2, [c["iou"] for c in rep["raw_labels"]["per_class"]], 0.4, label="raw labels")
    ax.bar(k + 0.2, [c["iou"] for c in rep["cleaned_labels"]["per_class"]], 0.4, label="cleaned labels")
    ax.set_xticks(k); ax.set_xticklabels(names, rotation=20, ha="right"); ax.set_ylim(0, 1)
    ax.set_ylabel("IoU on test tiles"); ax.legend(frameon=False); ax.grid(axis="y", alpha=.3)
    fig.tight_layout(); fig.savefig(gdir / "stage3_per_class_iou.png", dpi=150); plt.close(fig)

    # confusion matrix (row-normalised), cleaned labels
    C = np.array(rep["cleaned_labels"]["confusion"], float)
    Cn = C / np.maximum(C.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    im = ax.imshow(Cn, cmap="Greys", vmin=0, vmax=1)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{Cn[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if Cn[i, j] > .5 else "black")
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); fig.colorbar(im, fraction=.046)
    fig.tight_layout(); fig.savefig(gdir / "stage3_confusion.png", dpi=150); plt.close(fig)

    # water precision / recall vs threshold
    sw = rep["water_threshold_sweep"]
    fig, ax = plt.subplots(figsize=(6.4, 4))
    th = [s["threshold"] for s in sw]
    for key in ("precision", "recall", "iou"):
        ax.plot(th, [s[key] for s in sw], marker="o", ms=3, label=key)
    ax.set_xlabel("water probability threshold"); ax.set_ylim(0, 1); ax.grid(alpha=.3); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(gdir / "stage3_water_threshold.png", dpi=150); plt.close(fig)

    # examples: tiles that contain real water
    idx = np.where((Y == water).reshape(len(Y), -1).sum(1) >= 20)[0][:6]
    if len(idx):
        fig, ax = plt.subplots(3, len(idx), figsize=(2.2 * len(idx), 6.8))
        ax = np.array(ax).reshape(3, -1)
        for j, i in enumerate(idx):
            ax[0, j].imshow(X[i]); ax[1, j].imshow(Y[i] == water, cmap="Greys")
            ax[2, j].imshow(pred[i] == water, cmap="Greys")
            for r in range(3):
                ax[r, j].axis("off")
        for r, t in enumerate(("image", "true water", "predicted water")):
            ax[r, 0].set_title(t, fontsize=9, loc="left")
        fig.tight_layout(); fig.savefig(gdir / "stage3_examples.png", dpi=130); plt.close(fig)
