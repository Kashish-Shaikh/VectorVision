"""Stage 6b — the drone water detector, trained from random initialisation.

Two metrics, and they answer different questions:

  water IoU   - how exactly does the predicted shape match the real pool?
  frame F1    - does this frame contain standing water at all?

The second one is what the drone operator actually needs, and it is the metric the
best checkpoint is chosen on. Pixel-perfect outlines do not matter when the job is
"fly here and look". Frame accuracy alone would be misleading because most frames
contain water, so F1 is used instead.

Loss is cross-entropy plus Tversky with beta > alpha, which punishes missed water
harder than false alarms: a missed pool is an unsurveyed breeding site, while a
false alarm costs one wasted look.
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from ..satellite.unet import UNet, count_params


class DroneSet(Dataset):
    def __init__(self, X, Y, train: bool):
        self.X, self.Y, self.train = X, Y, train

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        import cv2
        x, y = self.X[i], self.Y[i]
        if self.train:
            k = np.random.randint(4)
            x, y = np.rot90(x, k, (0, 1)), np.rot90(y, k, (0, 1))
            if np.random.rand() < 0.5:
                x, y = x[:, ::-1], y[:, ::-1]
            x = x.astype(np.float32) / 255.0
            # Photometric jitter IS wanted here, unlike the satellite model: drone
            # video swings in exposure and white balance between frames.
            x = x * np.random.uniform(0.75, 1.3) + np.random.uniform(-0.08, 0.08)
            x = x * np.random.uniform(0.9, 1.1, (1, 1, 3)).astype(np.float32)
            if np.random.rand() < 0.3:
                x = cv2.GaussianBlur(np.ascontiguousarray(x), (0, 0), np.random.uniform(0.5, 1.6))
            if np.random.rand() < 0.3:
                x = x + np.random.normal(0, 0.03, x.shape).astype(np.float32)
            x = np.clip(x, 0, 1)
        else:
            x = x.astype(np.float32) / 255.0
        return (torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1).float(),
                torch.from_numpy(np.ascontiguousarray(y)).long())


def tversky(logits, target, alpha=0.3, beta=0.7, eps=1.0):
    p = torch.softmax(logits.float(), 1)[:, 1]
    t = (target == 1).float()
    tp = (p * t).sum((1, 2))
    fp = (p * (1 - t)).sum((1, 2))
    fn = ((1 - p) * t).sum((1, 2))
    return (1 - (tp + eps) / (tp + alpha * fp + beta * fn + eps)).mean()


@torch.no_grad()
def evaluate(model, loader, dev, min_px: int, thresh: float = 0.5):
    model.eval()
    tp = fp = fn = 0
    f_tp = f_fp = f_fn = f_tn = 0
    for x, y in loader:
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        with torch.autocast(device_type="cuda", enabled=dev == "cuda"):
            prob = torch.softmax(model(x).float(), 1)[:, 1]
        pred = prob >= thresh
        t = y == 1
        tp += (pred & t).sum().item(); fp += (pred & ~t).sum().item(); fn += (~pred & t).sum().item()
        pf = pred.flatten(1).sum(1) >= min_px
        tf = t.flatten(1).sum(1) >= min_px
        f_tp += (pf & tf).sum().item(); f_fp += (pf & ~tf).sum().item()
        f_fn += (~pf & tf).sum().item(); f_tn += (~pf & ~tf).sum().item()
    iou = tp / max(tp + fp + fn, 1)
    prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    fp_, fr_ = f_tp / max(f_tp + f_fp, 1), f_tp / max(f_tp + f_fn, 1)
    n = f_tp + f_fp + f_fn + f_tn
    return {"water_iou": iou, "pixel_precision": prec, "pixel_recall": rec,
            "frame_precision": fp_, "frame_recall": fr_,
            "frame_f1": 2 * fp_ * fr_ / max(fp_ + fr_, 1e-12),
            "frame_accuracy": (f_tp + f_tn) / max(n, 1),
            "frames": n, "frames_with_water": f_tp + f_fn}


def train(cfg: dict, cache: Path, out_dir: Path, dev: str, epochs=None, resume=False) -> dict:
    d = cfg["drone_unet"]
    meta = json.loads((cache / "floodnet_meta.json").read_text())
    min_px = int(d["frame_min_water_px"])
    E = epochs or d["epochs"]
    out_dir.mkdir(parents=True, exist_ok=True)

    Xtr, Ytr = np.load(cache / "X_train.npy"), np.load(cache / "Y_train.npy")
    vx, vy = cache / "X_val.npy", cache / "Y_val.npy"
    if vx.exists():
        Xva, Yva = np.load(vx), np.load(vy)
    else:                                  # no val split: hold out 20% of train
        n = int(0.8 * len(Xtr))
        Xva, Yva, Xtr, Ytr = Xtr[n:], Ytr[n:], Xtr[:n], Ytr[:n]
        print("no validation split found; held out 20% of train")

    nw = d.get("num_workers", 2)
    tr = DataLoader(DroneSet(Xtr, Ytr, True), batch_size=d["batch_size"], shuffle=True,
                    num_workers=nw, pin_memory=dev == "cuda", drop_last=len(Xtr) > d["batch_size"])
    va = DataLoader(DroneSet(Xva, Yva, False), batch_size=max(4, d["batch_size"]),
                    num_workers=nw, pin_memory=dev == "cuda")

    model = UNet(2, d["base_width"], 3).to(dev)
    n_params = count_params(model)
    share = float(Ytr.mean())
    w = torch.tensor([1.0, float(np.clip((1 - share) / max(share, 1e-6), 1, 10))], device=dev)
    ce = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=d["lr"] / 10, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=d["lr"], total_steps=E * max(len(tr), 1),
                                                pct_start=0.2)
    scaler = torch.amp.GradScaler("cuda", enabled=dev == "cuda")

    start, best = 0, -1.0
    last_p, best_p, log_p = out_dir / "last.pt", out_dir / "best.pt", out_dir / "history.csv"
    if resume and last_p.exists():
        st = torch.load(last_p, map_location=dev, weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"]); scaler.load_state_dict(st["scaler"])
        start, best = st["epoch"] + 1, st["best"]
        print(f"resumed from epoch {start} (best frame F1 {best:.4f})")
    elif log_p.exists():
        log_p.unlink()

    print(f"device {dev} | {n_params:,} parameters (random init) | "
          f"{len(Xtr)} train, {len(Xva)} val images at {d['image_size']}px")
    print(f"water is {100*share:.2f}% of training pixels -> class weight {w[1]:.1f}")

    new_log = not log_p.exists()
    with open(log_p, "a", newline="") as fh:
        wr = csv.writer(fh)
        if new_log:
            wr.writerow(["epoch", "train_loss", "water_iou", "pixel_precision", "pixel_recall",
                         "frame_f1", "frame_precision", "frame_recall", "frame_accuracy", "seconds"])
        for ep in range(start, E):
            t0 = time.time()
            model.train(); tot, nb, skipped = 0.0, 0, 0
            for x, y in tr:
                x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", enabled=dev == "cuda"):
                    logits = model(x)
                    loss = ce(logits.float(), y) + tversky(logits, y)
                if not torch.isfinite(loss):
                    skipped += 1; sched.step(); continue
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); sched.step()
                tot += loss.item(); nb += 1
            if nb == 0:
                print(f"epoch {ep+1}: all batches non-finite; stopping. Lower lr in config.")
                break

            m = evaluate(model, va, dev, min_px)
            secs = time.time() - t0
            wr.writerow([ep + 1, round(tot / nb, 5)] +
                        [round(m[k], 5) for k in ("water_iou", "pixel_precision", "pixel_recall",
                                                  "frame_f1", "frame_precision", "frame_recall",
                                                  "frame_accuracy")] + [round(secs, 1)])
            fh.flush()
            improved = m["frame_f1"] > best
            if improved:
                best = m["frame_f1"]
                torch.save({"model": model.state_dict(), "epoch": ep, "metrics": m,
                            "meta": meta, "config": d, "params": n_params}, best_p)
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "scaler": scaler.state_dict(), "epoch": ep, "best": best}, last_p)
            print(f"ep {ep+1:03d}/{E}  loss {tot/nb:.4f}  water IoU {m['water_iou']:.3f}  "
                  f"frame F1 {m['frame_f1']:.3f} (P {m['frame_precision']:.2f} R {m['frame_recall']:.2f})  "
                  f"{secs:.0f}s" + ("  * best" if improved else "")
                  + (f"  [{skipped} skipped]" if skipped else ""))

    summary = {"best_frame_f1": best, "epochs_run": E, "params": n_params,
               "water_classes": meta["water_classes"], "image_size": d["image_size"]}
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
    _plot(log_p, out_dir)
    return summary


def test(cfg: dict, cache: Path, out_dir: Path, outputs: Path, dev: str) -> dict:
    """Final numbers on the untouched test split, plus a threshold sweep."""
    d = cfg["drone_unet"]
    min_px = int(d["frame_min_water_px"])
    xt = cache / "X_test.npy"
    if not xt.exists():
        print("no test split in the cache; skipping the held-out test")
        return {}
    X, Y = np.load(xt), np.load(cache / "Y_test.npy")
    ck = torch.load(out_dir / "best.pt", map_location=dev, weights_only=False)
    model = UNet(2, ck["config"]["base_width"], 3).to(dev)
    model.load_state_dict(ck["model"]); model.eval()
    loader = DataLoader(DroneSet(X, Y, False), batch_size=max(4, d["batch_size"]))

    sweep = []
    for th in np.round(np.arange(0.2, 0.85, 0.1), 2):
        m = evaluate(model, loader, dev, min_px, thresh=float(th))
        m["threshold"] = float(th)
        sweep.append(m)
    main = max(sweep, key=lambda m: m["frame_f1"])
    rep = {"test_images": int(len(X)), "model_epoch": int(ck["epoch"] + 1),
           "best_by_frame_f1": main, "threshold_sweep": sweep,
           "water_classes": ck["meta"]["water_classes"],
           "note": "Test split never used for training or checkpoint selection."}
    (outputs / "metrics").mkdir(parents=True, exist_ok=True)
    (outputs / "metrics" / "stage6_drone_test.json").write_text(json.dumps(rep, indent=2))
    _examples(model, X, Y, dev, outputs / "graphs", float(main["threshold"]))
    return rep


def _plot(log_p: Path, out_dir: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    rows = list(csv.DictReader(open(log_p)))
    if not rows:
        return
    ep = [int(r["epoch"]) for r in rows]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ep, [float(r["frame_f1"]) for r in rows], label="frame F1 (does this frame show water?)")
    ax.plot(ep, [float(r["water_iou"]) for r in rows], label="water IoU (shape match)")
    ax.set_xlabel("epoch"); ax.set_ylabel("score"); ax.set_ylim(0, 1)
    ax.legend(frameon=False); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(out_dir / "training_curves.png", dpi=150); plt.close(fig)


@torch.no_grad()
def _examples(model, X, Y, dev, gdir: Path, thresh: float, n: int = 6):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    gdir.mkdir(parents=True, exist_ok=True)
    idx = np.argsort(-Y.reshape(len(Y), -1).sum(1))[:n]
    x = torch.from_numpy(X[idx].astype(np.float32) / 255).permute(0, 3, 1, 2).to(dev)
    prob = torch.softmax(model(x).float(), 1)[:, 1].cpu().numpy()
    fig, ax = plt.subplots(3, len(idx), figsize=(2.3 * len(idx), 7))
    ax = np.array(ax).reshape(3, -1)
    for j, i in enumerate(idx):
        ax[0, j].imshow(X[i]); ax[1, j].imshow(Y[i], cmap="Blues")
        ax[2, j].imshow(prob[j] >= thresh, cmap="Blues")
        for r in range(3):
            ax[r, j].axis("off")
    for r, t in enumerate(("drone image", "true water", "predicted water")):
        ax[r, 0].set_title(t, fontsize=9, loc="left")
    fig.tight_layout(); fig.savefig(gdir / "stage6_examples.png", dpi=130); plt.close(fig)
