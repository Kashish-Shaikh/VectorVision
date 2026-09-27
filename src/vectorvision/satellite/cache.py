"""Stage 2a — build the training cache for the satellite U-Net.

Reads the dataset ONCE and saves compact .npy arrays, because reading hundreds of
thousands of small files every epoch is far too slow (especially on Kaggle/Drive).

Everything it needs comes from Stage 1's report, so it never guesses:
  - which raw mask value is water
  - the list of raw class ids

Design decisions, each learned from an earlier run:
  1. Validation keeps the NATURAL water share. Enriching it would flatter the scores.
  2. Training is enriched with water-rich tiles, because water is ~1% of pixels.
  3. Water blobs smaller than `min_water_blob_px` are removed from the labels in BOTH
     sets. In our earlier run 79% of water pixels were isolated specks, which are
     label noise at 10 m resolution, not ponds.
  4. Class weights come from the natural (validation) distribution. Computing them
     from the enriched set cancels out the enrichment.
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np

from .inspect_sen2lulc import discover, read_any


def _load_stage1(outputs: Path) -> dict:
    rp = outputs / "metrics" / "stage1_dataset_report.json"
    if not rp.exists():
        raise SystemExit("Stage 1 report not found. Run:  python main.py verify --skip-ee")
    rep = json.loads(rp.read_text())
    if rep.get("water_raw_id") is None:
        raise SystemExit("Stage 1 did not identify a water class. Fix Stage 1 before training.")
    return rep


def _pairs(lay: list, split: str) -> list[tuple[Path, Path]]:
    imgs = [f for L in lay if L["split"] == split and L["kind"] == "image" for f in L["files"]]
    msks = [f for L in lay if L["split"] == split and L["kind"] == "mask" for f in L["files"]]
    im = {f.stem: f for f in imgs}
    mk = {f.stem: f for f in msks}
    keys = sorted(im.keys() & mk.keys())
    return [(im[k], mk[k]) for k in keys]


def _mask_to_index(m: np.ndarray, lut: np.ndarray) -> np.ndarray:
    if m.ndim == 3:            # colour-coded masks are not expected for Sen-2 LULC
        m = m[..., 0]
    return lut[np.clip(m, 0, len(lut) - 1)]


def _remove_speckle(y: np.ndarray, water: int, fill: int, min_px: int) -> np.ndarray:
    """Relabel water blobs smaller than min_px as the most common class."""
    import cv2
    k3 = np.ones((3, 3), np.uint8)
    out = y.copy()
    for i in range(len(y)):
        w = (y[i] == water).astype(np.uint8)
        if not w.any():
            continue
        w = cv2.morphologyEx(w, cv2.MORPH_OPEN, k3)
        n, lab, st, _ = cv2.connectedComponentsWithStats(w, 8)
        keep = np.zeros_like(w)
        for c in range(1, n):
            if st[c, cv2.CC_STAT_AREA] >= min_px:
                keep[lab == c] = 1
        out[i][(y[i] == water) & (keep == 0)] = fill
    return out


def build_cache(cfg: dict, root: Path, outputs: Path, force: bool = False) -> Path:
    c = cfg["satellite_unet"]
    cache = outputs / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    meta_p = cache / "lulc_meta.json"
    if meta_p.exists() and not force:
        print(f"cache already built -> {cache}  (use --rebuild-cache to redo)")
        return cache

    rep = _load_stage1(outputs)
    raw_ids = [int(k["raw_id"]) for k in rep["classes"]]
    lut = np.zeros(max(raw_ids) + 1, np.uint8)
    for i, r in enumerate(raw_ids):
        lut[r] = i
    water = raw_ids.index(int(rep["water_raw_id"]))
    most_common = int(np.argmax([k["pixel_share_pct"] for k in rep["classes"]]))
    n_cls = len(raw_ids)
    print(f"classes: {n_cls} raw ids {raw_ids} -> index 0..{n_cls-1};  water = index {water}")

    rng = random.Random(cfg["project"]["seed"])
    t0 = time.time()
    print("listing files (slow on network drives, done once)...")
    lay = discover(root)["layout"]          # one listing for both splits
    tr_pairs = _pairs(lay, "train")
    va_pairs = _pairs(lay, "val")
    print(f"  train pairs {len(tr_pairs)}, val pairs {len(va_pairs)}  ({time.time()-t0:.0f}s)")

    # ---- score water share of every training mask (cached, it is the slow part)
    score_p = cache / "train_water_share.json"
    if score_p.exists():
        share = json.loads(score_p.read_text())
    else:
        from tqdm.auto import tqdm
        share = {}
        for ip, mp in tqdm(tr_pairs, desc="scoring water", mininterval=5):
            m = read_any(mp)
            if m.ndim == 3:
                m = m[..., 0]
            share[mp.stem] = float((m == rep["water_raw_id"]).mean())
        score_p.write_text(json.dumps(share))

    # ---- choose the training subset
    N = min(c["train_subset"], len(tr_pairs))
    rich = [p for p in tr_pairs if share.get(p[1].stem, 0) > c["water_rich_min"]]
    rest = [p for p in tr_pairs if share.get(p[1].stem, 0) <= c["water_rich_min"]]
    n_rich = min(len(rich), int(N * c["water_rich_share"]))
    subset = rng.sample(rich, n_rich) + rng.sample(rest, N - n_rich)
    rng.shuffle(subset)
    va_sub = rng.sample(va_pairs, min(c["val_tiles"], len(va_pairs)))
    print(f"train subset {len(subset)} ({n_rich} water-rich) | val {len(va_sub)} (natural)")

    def load(pairs, desc):
        from tqdm.auto import tqdm
        x0 = read_any(pairs[0][0])
        H, W = x0.shape[:2]
        X = np.zeros((len(pairs), H, W, 3), np.uint8)
        Y = np.zeros((len(pairs), H, W), np.uint8)
        for i, (ip, mp) in enumerate(tqdm(pairs, desc=desc, mininterval=5)):
            x = read_any(ip)
            if x.ndim == 2:
                x = np.stack([x] * 3, -1)
            X[i] = x[..., :3]
            Y[i] = _mask_to_index(read_any(mp), lut)
        return X, Y

    Xtr, Ytr = load(subset, "train tiles")
    Xva, Yva = load(va_sub, "val tiles")

    raw_tr, raw_va = float((Ytr == water).mean()), float((Yva == water).mean())
    mb = c["min_water_blob_px"]
    if mb and mb > 1:
        Ytr = _remove_speckle(Ytr, water, most_common, mb)
        Yva = _remove_speckle(Yva, water, most_common, mb)
    print(f"water share  train {100*raw_tr:.2f}% -> {100*(Ytr==water).mean():.2f}% | "
          f"val {100*raw_va:.2f}% -> {100*(Yva==water).mean():.2f}%  (after removing blobs < {mb} px)")

    counts = np.bincount(Yva.ravel(), minlength=n_cls).astype(np.float64)
    weights = np.clip(counts.sum() / (n_cls * np.maximum(counts, 1)), 0.3, 10.0)

    np.save(cache / "X_train.npy", Xtr); np.save(cache / "Y_train.npy", Ytr)
    np.save(cache / "X_val.npy", Xva);   np.save(cache / "Y_val.npy", Yva)
    meta = {"n_classes": n_cls, "raw_ids": raw_ids, "water_index": water, "fill_index": most_common,
            "min_water_blob_px": mb, "class_weights": weights.round(4).tolist(),
            "train_tiles": len(Xtr), "val_tiles": len(Xva), "water_rich_tiles": n_rich,
            "water_share": {"train_raw": raw_tr, "train_clean": float((Ytr == water).mean()),
                            "val_raw": raw_va, "val_clean": float((Yva == water).mean())},
            "built_seconds": round(time.time() - t0)}
    meta_p.write_text(json.dumps(meta, indent=2))
    print(f"class weights (natural distribution): {weights.round(2).tolist()}")
    print(f"cache written to {cache} in {time.time()-t0:.0f}s")
    return cache
