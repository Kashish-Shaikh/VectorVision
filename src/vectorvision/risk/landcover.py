"""Stage 4b — connect the satellite U-Net to the risk model.

Until now the two models ran side by side: the U-Net classified land cover, and the
risk model never saw the result. This closes that loop, which is what the proposal's
pipeline actually describes.

For every point, a 64 x 64 Sentinel-2 chip is pulled from Earth Engine, the U-Net
predicts its land cover, and the fraction of each class becomes a feature. So instead
of only knowing "NDVI is 0.4 here", the risk model learns "this cell is 60% farmland,
20% built-up, 5% water".

One honest wrinkle: the U-Net was trained on Sen-2 LULC's own true-colour rendering,
and Earth Engine returns raw reflectance. The two have to be matched by a stretch, and
the match is approximate. `check_domain_shift` compares the predicted class mix against
the dataset's own class shares, so if the rendering is badly off you find out rather
than quietly feeding the model nonsense.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

CHIP = 64              # the tile size the U-Net was trained on (640 m at 10 m/px)
REFL_STRETCH = 0.30    # reflectance mapped to white; Sen-2 LULC looks like a 0-0.3 stretch


def fetch_chips(ee, image, pts: np.ndarray, chunk: int = 12, chip: int = CHIP) -> tuple:
    """Pull a square of pixels around each point.

    `neighborhoodToArray` turns each pixel's surroundings into an array, so one
    sampleRegions call returns a whole chip per point. Chunks are small because each
    point carries chip x chip x 3 numbers and the response has a size limit.
    """
    radius = chip // 2
    kern = ee.Kernel.square(radius)
    arr = image.select(["B4", "B3", "B2"]).neighborhoodToArray(kern)

    out, kept = [], []
    for i in range(0, len(pts), chunk):
        part = pts[i:i + chunk]
        feats = [ee.Feature(ee.Geometry.Point([float(lon), float(lat)]), {"pid": int(i + j)})
                 for j, (lat, lon) in enumerate(part)]
        fc = ee.FeatureCollection(feats)
        try:
            got = arr.sampleRegions(collection=fc, scale=10, geometries=False).getInfo()
        except Exception as e:
            print(f"\n  chunk at {i} failed ({str(e)[:90]}); skipping")
            continue
        for f in got["features"]:
            p = f["properties"]
            try:
                b = np.array([p["B4"], p["B3"], p["B2"]], dtype=np.float32)   # (3,H,W)
            except Exception:
                continue
            if b.ndim != 3 or b.shape[1] < chip or b.shape[2] < chip:
                continue
            img = np.transpose(b[:, :chip, :chip], (1, 2, 0))                 # (H,W,3) RGB
            img = np.clip(img / REFL_STRETCH, 0, 1) * 255.0
            out.append(img.astype(np.uint8))
            kept.append(int(p["pid"]))
        print(f"  chips {min(i + chunk, len(pts)):,}/{len(pts):,}, kept {len(out):,}", end="\r")
    print()
    return (np.stack(out) if out else np.zeros((0, chip, chip, 3), np.uint8)), np.array(kept, int)


def predict_fractions(chips: np.ndarray, model_dir: Path, dev: str, batch: int = 64):
    """Fraction of each land-cover class in every chip, from our own U-Net."""
    import torch

    from ..satellite.unet import UNet
    ck = torch.load(model_dir / "best.pt", map_location=dev, weights_only=False)
    meta = ck["meta"]
    n_cls = meta["n_classes"]
    model = UNet(n_cls, ck["config"]["base_width"], ck["config"]["in_channels"]).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()

    fr = np.zeros((len(chips), n_cls), np.float32)
    with torch.no_grad():
        for i in range(0, len(chips), batch):
            x = torch.from_numpy(chips[i:i + batch].astype(np.float32) / 255.0)
            x = x.permute(0, 3, 1, 2).to(dev)
            with torch.autocast(device_type="cuda", enabled=dev == "cuda"):
                pred = model(x).argmax(1).cpu().numpy()
            for j, pm in enumerate(pred):
                fr[i + j] = np.bincount(pm.ravel(), minlength=n_cls) / pm.size
    return fr, meta


def check_domain_shift(fractions: np.ndarray, meta: dict, stage1_report: Path) -> dict:
    """Does the predicted class mix resemble the dataset the U-Net was trained on?

    A large gap means the Earth Engine rendering does not look like Sen-2 LULC, so the
    fractions would be unreliable. Better to see that number than to assume.
    """
    rep = json.loads(stage1_report.read_text()) if stage1_report.exists() else None
    pred_share = fractions.mean(0)
    out = {"predicted_share_pct": [round(100 * float(v), 2) for v in pred_share]}
    if rep:
        train_share = np.array([c["pixel_share_pct"] for c in rep["classes"]]) / 100.0
        if len(train_share) == len(pred_share):
            diff = float(np.abs(pred_share - train_share).sum() / 2)   # total variation
            out["training_share_pct"] = [round(100 * float(v), 2) for v in train_share]
            out["total_variation_distance"] = round(diff, 3)
            out["verdict"] = ("close enough to trust" if diff < 0.25 else
                              "noticeably different: treat the land-cover features with caution")
    return out


def add_to_table(cfg: dict, table: Path, model_dir: Path, outputs: Path, dev: str) -> dict:
    """Append land-cover fraction columns to the Stage 4 training table."""
    import pandas as pd

    from .features import build_stack, country_geometry, init_ee

    df = pd.read_csv(table)
    if any(c.startswith("lulc_") for c in df.columns):
        print("land-cover columns already present (delete them to redo)")
        return {"skipped": True}

    pts = df[["lat", "lon"]].to_numpy(float)
    ee = init_ee(cfg)
    region = country_geometry(ee, cfg["study_area"]["country"])
    stack, _ = build_stack(ee, cfg, region)

    print(f"fetching {len(pts):,} image chips ({CHIP}x{CHIP} px, {CHIP*10} m across)...")
    chips, kept = fetch_chips(ee, stack, pts)
    if len(chips) == 0:
        raise SystemExit("No chips returned. Check the season window and cloud filter.")

    print(f"running the land-cover U-Net on {len(chips):,} chips...")
    fr, meta = predict_fractions(chips, model_dir, dev)
    shift = check_domain_shift(fr, meta, outputs / "metrics" / "stage1_dataset_report.json")

    names = [f"lulc_c{k}" for k in range(meta["n_classes"])]
    names[meta["water_index"]] = "lulc_water"
    for n in names:
        df[n] = np.nan
    df.loc[kept, names] = fr
    before = len(df)
    df = df.dropna(subset=names)
    df.to_csv(table, index=False)

    print(f"\nland-cover features added: {names}")
    print(f"  rows kept {len(df):,} of {before:,}")
    print(f"  predicted class mix:  {shift['predicted_share_pct']}")
    if "training_share_pct" in shift:
        print(f"  training dataset mix: {shift['training_share_pct']}")
        print(f"  difference: {shift['total_variation_distance']} -> {shift['verdict']}")
    meta_out = {"columns": names, "chip_px": CHIP, "chip_m": CHIP * 10,
                "reflectance_stretch": REFL_STRETCH, "rows": len(df), **shift}
    (outputs / "metrics" / "stage4b_landcover.json").write_text(json.dumps(meta_out, indent=2))
    return meta_out


def fractions_for_points(cfg: dict, ee, stack, pts: np.ndarray, model_dir: Path,
                         dev: str, n_cls: int, water_index: int):
    """Same features for the Stage 8 grid, so training and prediction match."""
    chips, kept = fetch_chips(ee, stack, pts)
    if len(chips) == 0:
        return None, None, None
    fr, meta = predict_fractions(chips, model_dir, dev)
    names = [f"lulc_c{k}" for k in range(meta["n_classes"])]
    names[meta["water_index"]] = "lulc_water"
    return fr, kept, names
