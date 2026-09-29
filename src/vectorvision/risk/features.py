"""Stage 4b — environmental features from Earth Engine.

One feature stack is defined here and used twice: to describe the occurrence and
background points (Stage 4, for training), and later to score every 100 m cell of
the study district (Stage 8, for prediction). Using the same code for both is what
keeps training and prediction consistent.

Never add a band from an asset that might be missing. Earth Engine builds a lazy
graph, so one unreachable asset breaks every later call until the stack is rebuilt.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

# Bands the risk model will see. Order is fixed so training and prediction match.
FEATURES = ["NDVI", "NDWI", "MNDWI", "B2", "B3", "B4", "B8", "B11",
            "rain_season_mm", "rain_30d_mm", "temp_c", "dew_c",
            "elev_m", "slope_deg", "hollow_m", "dist_water_m", "water_occurrence"]


def init_ee(cfg: dict):
    import ee
    try:
        ee.Initialize(project=cfg["earth_engine"]["project"])
    except Exception:
        ee.Authenticate()          # on Kaggle this prints a link and asks for a code
        ee.Initialize(project=cfg["earth_engine"]["project"])
    ee.Number(1).add(1).getInfo()
    return ee


def country_geometry(ee, name: str = "India"):
    fc = ee.FeatureCollection("FAO/GAUL/2015/level0").filter(ee.Filter.eq("ADM0_NAME", name))
    return fc.geometry()


def district_geometry(ee, state: str, district: str):
    adm = ee.FeatureCollection("FAO/GAUL/2015/level2").filter(ee.Filter.eq("ADM1_NAME", state))
    names = adm.aggregate_array("ADM2_NAME").getInfo()
    match = [n for n in names if district.lower()[:5] in n.lower()]
    if not match:
        raise SystemExit(f"No district like '{district}' in {state}. Options: {sorted(names)}")
    return adm.filter(ee.Filter.eq("ADM2_NAME", match[0])).geometry(), match[0]


def build_stack(ee, cfg: dict, aoi):
    """Satellite indices, climate and terrain, clipped to the area of interest."""
    s = cfg["study_area"]

    def mask_clouds(img):
        scl = img.select("SCL")
        ok = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10)).And(scl.neq(11))
        return img.updateMask(ok).divide(10000)

    coll = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(aoi).filterDate(s["season_start"], s["season_end"])
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 40)).map(mask_clouds))
    n_scenes = coll.size().getInfo()
    comp = coll.median()
    s2 = comp.addBands([
        comp.normalizedDifference(["B8", "B4"]).rename("NDVI"),
        comp.normalizedDifference(["B3", "B8"]).rename("NDWI"),     # water absorbs NIR
        comp.normalizedDifference(["B3", "B11"]).rename("MNDWI"),   # and SWIR even more
    ])

    y0, y1 = s["climate_years"]
    m0 = int(s["season_start"][5:7])
    m1 = int(s["season_end"][5:7])
    rain = (ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
            .filterDate(f"{y0}-01-01", f"{y1}-12-31")
            .filter(ee.Filter.calendarRange(6, m1, "month"))
            .sum().divide(y1 - y0 + 1).rename("rain_season_mm"))
    rain30 = (ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
              .filterDate(s["season_end"][:8] + "01", s["season_end"])
              .sum().rename("rain_30d_mm"))
    era = (ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR")
           .filterDate(f"{y0}-01-01", f"{y1}-12-31")
           .filter(ee.Filter.calendarRange(m0, m1, "month")).mean())
    temp = era.select("temperature_2m").subtract(273.15).rename("temp_c")
    dew = era.select("dewpoint_temperature_2m").subtract(273.15).rename("dew_c")

    dem = ee.Image("USGS/SRTMGL1_003")
    slope = ee.Terrain.slope(dem).rename("slope_deg")
    # A local dip relative to its 500 m surroundings: where water gathers.
    hollow = dem.focal_mean(500, "circle", "meters").subtract(dem).rename("hollow_m")
    jrc = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence").unmask(0)
    dist = (jrc.gt(50).fastDistanceTransform(512).sqrt().multiply(30)
            .rename("dist_water_m"))

    stack = (s2.select(["NDVI", "NDWI", "MNDWI", "B2", "B3", "B4", "B8", "B11"])
             .addBands([rain, rain30, temp, dew, dem.rename("elev_m"), slope, hollow, dist,
                        jrc.rename("water_occurrence")])
             .select(FEATURES))
    return stack.clip(aoi), n_scenes


def sample_points(ee, stack, pts: np.ndarray, labels: np.ndarray, scale: int = 100,
                  chunk: int = 400) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read the feature stack at each point, in chunks (one big request would fail)."""
    X, Y, keep = [], [], []
    n = len(pts)
    for i in range(0, n, chunk):
        part = pts[i:i + chunk]
        lab = labels[i:i + chunk]
        feats = [ee.Feature(ee.Geometry.Point([float(lon), float(lat)]), {"pid": int(i + j)})
                 for j, (lat, lon) in enumerate(part)]
        fc = ee.FeatureCollection(feats)
        for attempt in range(3):
            try:
                got = stack.sampleRegions(collection=fc, scale=scale, geometries=False).getInfo()
                break
            except Exception as e:
                if attempt == 2:
                    raise SystemExit(f"Earth Engine sampling failed: {str(e)[:200]}")
                time.sleep(3 * (attempt + 1))
        for f in got["features"]:
            p = f["properties"]
            if any(p.get(k) is None for k in FEATURES):
                continue                      # masked pixel (cloud, water edge): drop it
            X.append([float(p[k]) for k in FEATURES])
            j = int(p["pid"])
            Y.append(int(labels[j]))
            keep.append(j)
        print(f"  sampled {min(i + chunk, n):,}/{n:,} points, kept {len(X):,}", end="\r")
    print()
    return np.array(X, float), np.array(Y, int), np.array(keep, int)


def background_inside(ee, region, n: int, seed: int) -> np.ndarray:
    """Random points inside the country POLYGON, not its bounding box.

    Sampling a lat/lon rectangle around India puts points in the Arabian Sea, the
    Bay of Bengal and neighbouring countries. Those fall outside the clipped feature
    stack and get dropped, which removes background points for a reason that has
    nothing to do with mosquitoes and biases the comparison.
    """
    fc = ee.FeatureCollection.randomPoints(region=region, points=int(n), seed=int(seed))
    coords = fc.geometry().coordinates().getInfo()
    return np.array([[c[1], c[0]] for c in coords], float)     # -> (lat, lon)


def drop_far_from(pts: np.ndarray, presence: np.ndarray, min_km: float) -> np.ndarray:
    """Remove background points that sit within min_km of a presence record."""
    if len(pts) == 0 or len(presence) == 0:
        return pts
    dlat = min_km / 111.32
    occupied = set()
    for lat, lon in presence:
        dlon = min_km / (111.32 * max(math.cos(math.radians(lat)), 0.2))
        occupied.add((int(lat / dlat), int(lon / dlon)))
    keep = []
    for lat, lon in pts:
        dlon = min_km / (111.32 * max(math.cos(math.radians(lat)), 0.2))
        if (int(lat / dlat), int(lon / dlon)) not in occupied:
            keep.append((lat, lon))
    return np.array(keep, float)


def relative_humidity(temp_c: np.ndarray, dew_c: np.ndarray) -> np.ndarray:
    """Magnus formula: saturation vapour pressure at dew point / at air temperature."""
    e = lambda t: 6.112 * np.exp(17.67 * t / (t + 243.5))
    return np.clip(100 * e(dew_c) / e(temp_c), 0, 100)


def build_training_table(cfg: dict, out_dir: Path, force: bool = False) -> dict:
    from .occurrence import fetch_occurrences

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_p = out_dir / "risk_training_table.csv"
    if csv_p.exists() and not force:
        print(f"training table already built -> {csv_p}  (use --rebuild to redo)")
        return json.loads((out_dir / "risk_training_meta.json").read_text())

    o, s = cfg["occurrence"], cfg["study_area"]
    presence, occ_meta = fetch_occurrences(cfg, out_dir, force=force)
    if len(presence) < 50:
        raise SystemExit(f"Only {len(presence)} presence points. Too few to train on.")

    ee = init_ee(cfg)
    country = country_geometry(ee, s["country"])

    # Ask for extra background points, because some will land on cloud-masked pixels.
    # They are trimmed back to the exact ratio AFTER sampling, so the ratio is honest.
    n_wanted = int(len(presence) * o["background_ratio"])
    over = float(o.get("background_oversample", 2.5))
    background = background_inside(ee, country, int(n_wanted * over), cfg["project"]["seed"])
    background = drop_far_from(background, presence, o["background_min_km"])
    print(f"background points inside the country: {len(background):,} "
          f"(asking for {over:g}x the {n_wanted:,} needed, trimmed after sampling)")

    pts = np.vstack([presence, background])
    labels = np.concatenate([np.ones(len(presence), int), np.zeros(len(background), int)])

    print("building the Earth Engine feature stack over the country...")
    stack, n_scenes = build_stack(ee, cfg, country)
    print(f"  Sentinel-2 scenes in the season window: {n_scenes:,}")
    print(f"sampling {len(pts):,} points at {s['grid_m']} m (chunked)...")
    X, Y, kept = sample_points(ee, stack, pts, labels, scale=s["grid_m"])
    if len(X) < 100:
        raise SystemExit("Almost every point was dropped. Check the season window and cloud filter.")

    # How many of each class survived? Very different rates would mean the drops
    # themselves carry information, which would bias the model.
    keep_pres = float((Y == 1).sum()) / max(len(presence), 1)
    keep_back = float((Y == 0).sum()) / max(len(background), 1)
    print(f"  kept {100*keep_pres:.0f}% of presence points and {100*keep_back:.0f}% of background points")
    if abs(keep_pres - keep_back) > 0.15:
        print("  WARNING: the two classes were dropped at very different rates. "
              "Check the cloud filter and season window before trusting the model.")

    # Trim background down to the requested ratio, chosen at random.
    n_keep_bg = min(int((Y == 1).sum() * o["background_ratio"]), int((Y == 0).sum()))
    rng = np.random.default_rng(cfg["project"]["seed"])
    bg_idx = np.where(Y == 0)[0]
    sel = np.sort(np.concatenate([np.where(Y == 1)[0], rng.choice(bg_idx, n_keep_bg, replace=False)]))
    X, Y, kept = X[sel], Y[sel], kept[sel]
    print(f"  final table: {(Y == 1).sum():,} presence and {(Y == 0).sum():,} background rows")

    hum = relative_humidity(X[:, FEATURES.index("temp_c")], X[:, FEATURES.index("dew_c")])
    cols = FEATURES + ["humidity_pct"]
    data = np.column_stack([X, hum])
    coords = pts[kept]

    import csv as _csv
    with open(csv_p, "w", newline="") as fh:
        w = _csv.writer(fh)
        w.writerow(["lat", "lon", "presence"] + cols)
        for (lat, lon), lab, row in zip(coords, Y, data):
            w.writerow([f"{lat:.6f}", f"{lon:.6f}", int(lab)] + [f"{v:.5f}" for v in row])

    meta = {
        "occurrence": occ_meta,
        "presence_sampled": int((Y == 1).sum()), "background_sampled": int((Y == 0).sum()),
        "points_requested": int(len(pts)), "points_kept": int(len(X)),
        "dropped_masked_pixels": int(len(pts) - len(X)),
        "kept_rate_presence": round(keep_pres, 3), "kept_rate_background": round(keep_back, 3),
        "features": cols, "scale_m": s["grid_m"], "s2_scenes": n_scenes,
        "season": f"{s['season_start']} to {s['season_end']}",
        "climate_years": s["climate_years"],
        "label_meaning": "1 = Anopheles recorded here (real field record). "
                         "0 = background point, NOT a confirmed absence.",
        "limitations": [
            "Presence-only data: zeros are background, not verified absences.",
            "Survey bias: records cluster near roads, towns and institutes.",
            "Records span years while features come from one season window.",
            "Vector presence is not the same as an active breeding site; the drone confirms that.",
        ],
    }
    (out_dir / "risk_training_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\ntraining table written: {csv_p}")
    print(f"  {meta['presence_sampled']:,} presence and {meta['background_sampled']:,} background rows, "
          f"{len(cols)} features")
    return meta
