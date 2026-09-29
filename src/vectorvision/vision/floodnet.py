"""Stage 6a — FloodNet: real drone images with pixel labels.

FloodNet was collected with a DJI Mavic Pro after Hurricane Harvey: 2,343 images,
labelled pixel by pixel. Its ten classes are:

    0 background        1 building flooded   2 building non-flooded
    3 road flooded      4 road non-flooded   5 water
    6 tree              7 vehicle            8 pool
    9 grass

For breeding-site detection only standing water matters, so the ten classes are
merged into one binary target. Which classes count is set in config.yaml, because
it is a judgement call that should be visible, not hidden in code:

    water (5) and pool (8)  - unambiguous standing water
    road flooded (3)        - standing water on a road surface
    building flooded (1)    - mostly roof and wall pixels, so it is OFF by default

Why this replaces the synthetic images: a model trained on images I generated can
only be tested on images I generated, which proves nothing about real flights.
"""
from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

IMG_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
CLASS_NAMES = ["background", "building-flooded", "building-non-flooded", "road-flooded",
               "road-non-flooded", "water", "tree", "vehicle", "pool", "grass"]


def discover(root: Path) -> dict:
    """Find image and mask folders without assuming the archive's layout."""
    buckets = {}
    for f in root.rglob("*"):
        if not f.is_file() or f.suffix.lower() not in IMG_EXT:
            continue
        name = str(f.parent).lower().replace("\\", "/")
        kind = "mask" if any(k in name for k in ("label", "mask", "gt", "annot")) else "image"
        split = next((s for s in ("train", "val", "test") if s in name), "unknown")
        buckets.setdefault((split, kind), []).append(f)
    return {k: sorted(v) for k, v in buckets.items()}


def pair_split(found: dict, split: str) -> list[tuple[Path, Path]]:
    imgs = found.get((split, "image"), [])
    msks = found.get((split, "mask"), [])
    # FloodNet masks are usually named <id>_lab.png for image <id>.jpg
    mk = {}
    for m in msks:
        stem = m.stem
        for suffix in ("_lab", "_label", "_mask", "_gt"):
            stem = stem.replace(suffix, "")
        mk[stem] = m
    return [(i, mk[i.stem]) for i in imgs if i.stem in mk]


def summarise(root: Path) -> dict:
    found = discover(root)
    out = {}
    print("FloodNet folders found:")
    for (split, kind), files in sorted(found.items()):
        print(f"  {split:<8}{kind:<6}{len(files):>6} files   {files[0].parent}")
    for split in ("train", "val", "test", "unknown"):
        pairs = pair_split(found, split)
        if pairs:
            out[split] = len(pairs)
            print(f"  -> {split}: {len(pairs)} image/mask pairs")
    if not out:
        raise SystemExit("No image/mask pairs found. Check the dataset folder in config.yaml.")
    return out


def build_cache(cfg: dict, root: Path, cache: Path, force: bool = False) -> Path:
    d = cfg["drone_unet"]
    size = int(d["image_size"])
    water_ids = set(int(i) for i in d["water_class_ids"])
    cache.mkdir(parents=True, exist_ok=True)
    meta_p = cache / "floodnet_meta.json"
    if meta_p.exists() and not force:
        print(f"cache already built -> {cache}  (use --rebuild-cache to redo)")
        return cache

    import cv2
    from tqdm.auto import tqdm

    t0 = time.time()
    found = discover(root)
    counts = Counter()
    splits = {}
    for split in ("train", "val", "test"):
        pairs = pair_split(found, split)
        if not pairs:
            continue
        X = np.zeros((len(pairs), size, size, 3), np.uint8)
        Y = np.zeros((len(pairs), size, size), np.uint8)
        for i, (ip, mp) in enumerate(tqdm(pairs, desc=f"{split} images", mininterval=5)):
            img = cv2.cvtColor(cv2.imread(str(ip)), cv2.COLOR_BGR2RGB)
            m = cv2.imread(str(mp), cv2.IMREAD_UNCHANGED)
            if m.ndim == 3:
                m = m[..., 0]
            counts.update(dict(zip(range(10), np.bincount(m.ravel(), minlength=10)[:10].tolist())))
            X[i] = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
            binm = np.isin(m, list(water_ids)).astype(np.uint8)
            Y[i] = cv2.resize(binm, (size, size), interpolation=cv2.INTER_NEAREST)
        np.save(cache / f"X_{split}.npy", X)
        np.save(cache / f"Y_{split}.npy", Y)
        splits[split] = {"pairs": len(pairs), "water_pixel_pct": round(100 * float(Y.mean()), 2),
                         "frames_with_water_pct": round(100 * float((Y.reshape(len(Y), -1).sum(1) > 0).mean()), 1)}
        print(f"  {split}: {len(pairs)} pairs, water {splits[split]['water_pixel_pct']}% of pixels, "
              f"{splits[split]['frames_with_water_pct']}% of frames contain water")

    total = sum(counts.values()) or 1
    class_share = {CLASS_NAMES[k]: round(100 * v / total, 2) for k, v in sorted(counts.items()) if k < 10}
    print("\n  original class shares:", class_share)
    print(f"  merged into 'standing water': {[CLASS_NAMES[i] for i in sorted(water_ids)]}")

    meta = {"image_size": size, "water_class_ids": sorted(water_ids),
            "water_classes": [CLASS_NAMES[i] for i in sorted(water_ids)],
            "original_class_share_pct": class_share, "splits": splits,
            "built_seconds": round(time.time() - t0),
            "note": "Binary target: 1 = standing water, 0 = everything else."}
    meta_p.write_text(json.dumps(meta, indent=2))
    print(f"\ncache written to {cache} in {time.time()-t0:.0f}s")
    return cache
