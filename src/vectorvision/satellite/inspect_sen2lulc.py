"""Stage 1b — inspect the Sen-2 LULC dataset without assuming its layout.

Answers, from the files themselves:
  image format, mask format, dimensions, bands, dtype and value range,
  number of classes and their raw IDs, split organisation, sample counts,
  which class is water (by spectral signature), and how noisy the water labels are.
"""
from __future__ import annotations

import json
import random
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

IMG_EXT = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}


# ------------------------------------------------------------------ reading
def read_any(path: Path) -> np.ndarray:
    """PIL for ordinary images, tifffile for multi-band GeoTIFFs."""
    try:
        from PIL import Image
        with Image.open(path) as im:
            return np.array(im)
    except Exception:
        import tifffile
        arr = tifffile.imread(str(path))
        # tifffile may return (bands, H, W); normalise to (H, W, bands)
        if arr.ndim == 3 and arr.shape[0] < arr.shape[-1] and arr.shape[0] <= 16:
            arr = np.moveaxis(arr, 0, -1)
        return arr


# ------------------------------------------------------------------ discovery
def maybe_extract(root: Path) -> None:
    zips = list(root.glob("*.zip"))
    has_dirs = any(p.is_dir() for p in root.iterdir()) if root.exists() else False
    if zips and not has_dirs:
        z = zips[0]
        print(f"  extracting {z.name} ({z.stat().st_size / 1e9:.2f} GB) - a few minutes...")
        with zipfile.ZipFile(z) as zf:
            zf.extractall(root)
        print("  extracted")


def discover(root: Path) -> dict:
    """Find every folder that directly holds image files and classify it."""
    folders = defaultdict(list)
    for f in root.rglob("*"):
        if f.is_file() and f.suffix.lower() in IMG_EXT:
            folders[f.parent].append(f)
    layout = []
    for d, files in folders.items():
        name = str(d.relative_to(root)).lower().replace("\\", "/")
        kind = "mask" if ("mask" in name or "label" in name or "gt" in name.split("/")) else "image"
        split = next((s for s in ("train", "val", "test") if s in name), "unknown")
        layout.append({"dir": d, "kind": kind, "split": split, "files": files,
                       "exts": dict(Counter(x.suffix.lower() for x in files))})
    return {"layout": layout}


# ------------------------------------------------------------------ inspection
def inspect_sen2lulc(root: Path, out_dir: Path, n_samples: int = 300,
                     do_extract: bool = True, seed: int = 42) -> dict:
    print("\nSen-2 LULC dataset")
    if not root.exists():
        print(f"  [FAIL] folder not found: {root}")
        print("         Download 'SEN-2 LULC.zip' (Mendeley f4ky6ks248) and put it in that folder.")
        return {"status": "FAIL", "reason": "dataset folder missing", "expected_at": str(root)}
    if do_extract:
        maybe_extract(root)

    lay = discover(root)["layout"]
    if not lay:
        print(f"  [FAIL] no image files anywhere under {root}")
        return {"status": "FAIL", "reason": "no image files found"}

    print("  folders found:")
    for L in sorted(lay, key=lambda x: (x["split"], x["kind"])):
        rel = L["dir"].relative_to(root)
        print(f"    {L['split']:<7} {L['kind']:<5} {len(L['files']):>7} files  {L['exts']}  {rel}")

    # pair images with masks per split, by filename stem
    splits = {}
    for s in ("train", "val", "test", "unknown"):
        imgs = [f for L in lay if L["split"] == s and L["kind"] == "image" for f in L["files"]]
        msks = [f for L in lay if L["split"] == s and L["kind"] == "mask" for f in L["files"]]
        if not imgs and not msks:
            continue
        im = {f.stem: f for f in imgs}
        mk = {f.stem: f for f in msks}
        common = sorted(im.keys() & mk.keys())
        splits[s] = {"images": len(imgs), "masks": len(msks), "pairs": len(common),
                     "unpaired": len(im.keys() ^ mk.keys()),
                     "_pairs": [(im[k], mk[k]) for k in common]}
    print("  pairing:")
    for s, v in splits.items():
        print(f"    {s:<7} images {v['images']:>7} | masks {v['masks']:>7} | "
              f"pairs {v['pairs']:>7} | unpaired {v['unpaired']}")

    train = splits.get("train") or next(iter(splits.values()))
    if not train["_pairs"]:
        print("  [FAIL] no image/mask pairs could be matched by filename")
        return {"status": "FAIL", "reason": "no matched pairs"}

    rng = random.Random(seed)
    sample = rng.sample(train["_pairs"], min(n_samples, len(train["_pairs"])))

    shapes, dtypes, vmin, vmax = Counter(), Counter(), [], []
    mshapes, mdtypes = Counter(), Counter()
    raw_counts = Counter()
    rgb_sum, rgb_n = defaultdict(lambda: np.zeros(3)), Counter()
    water_blobs = []

    x0 = read_any(sample[0][0])
    n_bands = 1 if x0.ndim == 2 else x0.shape[-1]

    for ip, mp in sample:
        x, y = read_any(ip), read_any(mp)
        shapes[x.shape] += 1; dtypes[str(x.dtype)] += 1
        vmin.append(float(x.min())); vmax.append(float(x.max()))
        mshapes[y.shape] += 1; mdtypes[str(y.dtype)] += 1
        if y.ndim == 3:            # colour-coded mask: collapse to one id per colour
            y = y[..., 0].astype(np.int64) * 65536 + y[..., 1].astype(np.int64) * 256 + y[..., 2]
        vals, cnt = np.unique(y, return_counts=True)
        raw_counts.update(dict(zip(vals.tolist(), cnt.tolist())))
        if x.ndim == 3 and x.shape[-1] >= 3 and x.shape[:2] == y.shape[:2]:
            for v in vals.tolist():
                sel = y == v
                rgb_sum[v] += x[sel][:, :3].astype(np.float64).sum(0)
                rgb_n[v] += int(sel.sum())

    total = sum(raw_counts.values())
    classes = []
    for v in sorted(raw_counts):
        mean = (rgb_sum[v] / max(rgb_n[v], 1)).round(1).tolist() if rgb_n[v] else None
        classes.append({"raw_id": v, "pixel_share_pct": round(100 * raw_counts[v] / total, 2),
                        "mean_rgb": mean})

    # Water: the only surface where blue exceeds red (it absorbs red, scatters blue).
    water = None
    cand = [c for c in classes if c["mean_rgb"]]
    if cand:
        best = max(cand, key=lambda c: c["mean_rgb"][2] - c["mean_rgb"][0])
        if best["mean_rgb"][2] - best["mean_rgb"][0] > 3:
            water = best["raw_id"]

    # Label noise: how fragmented are the water labels?
    if water is not None:
        try:
            import cv2
            for ip, mp in sample:
                y = read_any(mp)
                if y.ndim == 3:
                    continue
                m = (y == water).astype(np.uint8)
                if m.any():
                    n, _, st, _ = cv2.connectedComponentsWithStats(m, 8)
                    water_blobs.extend(st[1:, cv2.CC_STAT_AREA].tolist())
        except ImportError:
            pass

    print(f"\n  images : {dict(shapes)}  dtype {dict(dtypes)}  bands {n_bands}  "
          f"value range {min(vmin):.0f}-{max(vmax):.0f}")
    print(f"  masks  : {dict(mshapes)}  dtype {dict(mdtypes)}")
    print(f"  classes: {len(classes)} raw ids")
    print(f"    {'raw id':>7} {'share':>8}   mean R,G,B")
    for c in classes:
        tag = "   <-- water (blue > red)" if c["raw_id"] == water else ""
        print(f"    {c['raw_id']:>7} {c['pixel_share_pct']:>7.2f}%   {c['mean_rgb']}{tag}")

    noise = None
    if water_blobs:
        a = np.array(water_blobs)
        noise = {"components": int(len(a)), "median_px": float(np.median(a)),
                 "share_under_6px_pct": round(100 * float((a < 6).mean()), 1)}
        print(f"\n  water label fragmentation: median blob {noise['median_px']:.0f} px, "
              f"{noise['share_under_6px_pct']}% of blobs under 6 px")
        if noise["share_under_6px_pct"] > 50:
            print("    -> most water labels are single-pixel speckle; Stage 2 filters them before training")

    preprocessed = (len(shapes) == 1 and max(vmax) <= 255 and "uint8" in dtypes)
    report = {
        "status": "PASS" if water is not None else "WARN",
        "root": str(root),
        "splits": {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in splits.items()},
        "image": {"shapes": {str(k): v for k, v in shapes.items()}, "dtypes": dict(dtypes),
                  "bands": n_bands, "value_min": min(vmin), "value_max": max(vmax),
                  "image_exts": sorted({e for L in lay if L["kind"] == "image" for e in L["exts"]})},
        "mask": {"shapes": {str(k): v for k, v in mshapes.items()}, "dtypes": dict(mdtypes),
                 "mask_exts": sorted({e for L in lay if L["kind"] == "mask" for e in L["exts"]})},
        "classes": classes, "n_classes": len(classes), "water_raw_id": water,
        "water_label_noise": noise,
        "appears_preprocessed": preprocessed,
        "note_bands": "3 bands = visible RGB only (B4,B3,B2). No NIR/SWIR, so NDWI cannot be "
                      "computed from this dataset; live Sentinel-2 via Earth Engine supplies it.",
        "class_names_confirmed": False,
        "class_names_note": "Only water is identified from data. Confirm the other six names "
                            "visually with outputs/graphs/stage1_samples.png.",
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "stage1_dataset_report.json").write_text(json.dumps(report, indent=2))
    _figure(sample[:6], classes, water, out_dir.parent / "graphs" / "stage1_samples.png")
    return report


def _figure(pairs, classes, water, path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(2, len(pairs), figsize=(2.6 * len(pairs), 5.6))
    for j, (ip, mp) in enumerate(pairs):
        x, y = read_any(ip), read_any(mp)
        ax[0, j].imshow(x[..., :3] if x.ndim == 3 else x, cmap=None if x.ndim == 3 else "gray")
        ax[1, j].imshow(y if y.ndim == 2 else y[..., :3], cmap="tab10", interpolation="nearest")
        ax[0, j].set_title(ip.stem, fontsize=8)
        for a in ax[:, j]:
            a.axis("off")
    fig.suptitle(f"Sen-2 LULC samples (top: image, bottom: mask). Water raw id = {water}")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
