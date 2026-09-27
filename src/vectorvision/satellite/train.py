"""Stage 2b — train the satellite U-Net from random initialisation.

Safeguards learned from earlier runs:
  - gradient clipping (norm 1.0): an earlier run collapsed to NaN without it
  - non-finite losses are skipped instead of poisoning the weights
  - training stops early if the loss becomes NaN for a whole epoch
  - every epoch is logged to CSV and the last state is saved, so a Kaggle/Colab
    timeout never loses more than one epoch (resume with --resume)
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

from .unet import UNet, count_params


class TileSet(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray, train: bool):
        self.X, self.Y, self.train = X, Y, train

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        x, y = self.X[i], self.Y[i]
        if self.train:
            # Rotations and flips only. No colour changes: colour is exactly what
            # separates water from land, so jittering it would destroy the signal.
            k = np.random.randint(4)
            x, y = np.rot90(x, k, (0, 1)), np.rot90(y, k, (0, 1))
            if np.random.rand() < 0.5:
                x, y = x[:, ::-1], y[:, ::-1]
        x = torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1).float().div(255.0)
        return x, torch.from_numpy(np.ascontiguousarray(y)).long()


def dice_loss(logits, target, n_cls, eps=1.0):
    p = F.softmax(logits.float(), 1)
    t = F.one_hot(target, n_cls).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    return 1 - ((2 * (p * t).sum(dims) + eps) / (p.sum(dims) + t.sum(dims) + eps)).mean()


@torch.no_grad()
def evaluate(model, loader, n_cls, dev):
    """Returns per-class IoU, precision, recall and the confusion matrix."""
    model.eval()
    conf = torch.zeros(n_cls, n_cls, dtype=torch.long, device=dev)
    for x, y in loader:
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        with torch.autocast(device_type="cuda", enabled=dev == "cuda"):
            pred = model(x).argmax(1)
        idx = y.reshape(-1) * n_cls + pred.reshape(-1)
        conf += torch.bincount(idx, minlength=n_cls * n_cls).reshape(n_cls, n_cls)
    conf = conf.cpu().numpy().astype(np.float64)
    tp = np.diag(conf)
    iou = tp / np.maximum(conf.sum(0) + conf.sum(1) - tp, 1)
    prec = tp / np.maximum(conf.sum(0), 1)
    rec = tp / np.maximum(conf.sum(1), 1)
    return iou, prec, rec, conf


def train(cfg: dict, cache: Path, out_dir: Path, dev: str, epochs: int | None = None,
          resume: bool = False) -> dict:
    c = cfg["satellite_unet"]
    meta = json.loads((cache / "lulc_meta.json").read_text())
    n_cls, water = meta["n_classes"], meta["water_index"]
    E = epochs or c["epochs"]
    out_dir.mkdir(parents=True, exist_ok=True)

    Xtr, Ytr = np.load(cache / "X_train.npy"), np.load(cache / "Y_train.npy")
    Xva, Yva = np.load(cache / "X_val.npy"), np.load(cache / "Y_val.npy")
    nw = c.get("num_workers", 2)
    tr = DataLoader(TileSet(Xtr, Ytr, True), batch_size=c["batch_size"], shuffle=True,
                    num_workers=nw, pin_memory=dev == "cuda", drop_last=True, persistent_workers=nw > 0)
    va = DataLoader(TileSet(Xva, Yva, False), batch_size=256, shuffle=False,
                    num_workers=nw, pin_memory=dev == "cuda")

    model = UNet(n_cls, c["base_width"], c["in_channels"]).to(dev)
    n_params = count_params(model)
    w = torch.tensor(meta["class_weights"], dtype=torch.float32, device=dev)
    ce = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=c["lr"] / 10, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=c["lr"], total_steps=E * len(tr), pct_start=0.2)
    scaler = torch.amp.GradScaler("cuda", enabled=dev == "cuda")

    start, best = 0, -1.0
    last_p, best_p, log_p = out_dir / "last.pt", out_dir / "best.pt", out_dir / "history.csv"
    if resume and last_p.exists():
        st = torch.load(last_p, map_location=dev)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"]); scaler.load_state_dict(st["scaler"])
        start, best = st["epoch"] + 1, st["best"]
        print(f"resumed from epoch {start} (best mean IoU so far {best:.4f})")
    elif log_p.exists():
        log_p.unlink()

    print(f"device {dev} | {n_params:,} parameters (random init) | {len(Xtr)} train, {len(Xva)} val tiles | "
          f"{E} epochs x {len(tr)} steps")
    print(f"water = class {water}; class weights {np.round(meta['class_weights'], 2).tolist()}")

    new_log = not log_p.exists()
    with open(log_p, "a", newline="") as fh:
        wr = csv.writer(fh)
        if new_log:
            wr.writerow(["epoch", "train_loss", "mean_iou", "water_iou", "water_precision", "water_recall",
                         "lr", "seconds"] + [f"iou_c{k}" for k in range(n_cls)])
        for ep in range(start, E):
            t0 = time.time()
            model.train(); tot, nb, skipped = 0.0, 0, 0
            for x, y in tr:
                x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", enabled=dev == "cuda"):
                    logits = model(x)
                    loss = ce(logits.float(), y) + dice_loss(logits, y, n_cls)
                if not torch.isfinite(loss):
                    skipped += 1
                    sched.step()
                    continue
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); sched.step()
                tot += loss.item(); nb += 1
            if nb == 0:
                print(f"epoch {ep+1}: every batch was non-finite - stopping. Lower lr in config.")
                break

            iou, prec, rec, conf = evaluate(model, va, n_cls, dev)
            miou, secs = float(iou.mean()), time.time() - t0
            wr.writerow([ep + 1, round(tot / nb, 5), round(miou, 5), round(float(iou[water]), 5),
                         round(float(prec[water]), 5), round(float(rec[water]), 5),
                         f"{sched.get_last_lr()[0]:.2e}", round(secs, 1)] + [round(float(v), 5) for v in iou])
            fh.flush()
            improved = miou > best
            if improved:
                best = miou
                torch.save({"model": model.state_dict(), "epoch": ep, "iou": iou.tolist(),
                            "confusion": conf.tolist(), "meta": meta, "params": n_params,
                            "config": c}, best_p)
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "scaler": scaler.state_dict(), "epoch": ep, "best": best}, last_p)
            print(f"ep {ep+1:03d}/{E}  loss {tot/nb:.4f}  mIoU {miou:.4f}  water IoU {iou[water]:.4f} "
                  f"(P {prec[water]:.2f} R {rec[water]:.2f})  {secs:.0f}s" + ("  * best" if improved else "")
                  + (f"  [{skipped} bad batches skipped]" if skipped else ""))

    _plot(log_p, out_dir, water)
    summary = {"best_mean_iou": best, "epochs_run": E, "params": n_params,
               "checkpoint": str(best_p), "history": str(log_p)}
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def _plot(log_p: Path, out_dir: Path, water: int) -> None:
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
    ax.plot(ep, [float(r["mean_iou"]) for r in rows], label="mean IoU (all classes)")
    ax.plot(ep, [float(r["water_iou"]) for r in rows], label=f"water IoU (class {water})")
    ax.set_xlabel("epoch"); ax.set_ylabel("IoU on validation tiles"); ax.set_ylim(0, 1)
    ax.legend(frameon=False); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_dir / "training_curves.png", dpi=150); plt.close(fig)
