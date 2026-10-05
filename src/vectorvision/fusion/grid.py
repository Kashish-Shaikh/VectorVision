"""Stage 8a — score the study district on a grid, coarse first, then fine.

This mirrors the two-stage logic of the proposal, for a practical reason as much as
a conceptual one. Gondiya is about 5,200 km², which is roughly 520,000 cells of
100 m × 100 m. Sampling every one through Earth Engine is not feasible, and it would
be wasted effort: most of the district is obviously unsuitable.

  Tier 1 (macro)  every 1 km cell across the whole district, about 5,200 of them
  Tier 2 (micro)  100 m cells inside only the highest-scoring 1 km zones

So the coarse pass decides where to look closely, exactly as the satellite pass
decides where to fly the drone. Coverage is stated honestly in the output: the fine
grid covers the top zones, not the whole district.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from ..risk.features import FEATURES, build_stack, district_geometry, init_ee, relative_humidity, sample_points


def _grid_points(ee, geom, spacing_m: float, max_points: int) -> np.ndarray:
    """Regular lattice of cell centres clipped to the district polygon."""
    b = geom.bounds().coordinates().getInfo()[0]
    lons = [c[0] for c in b]; lats = [c[1] for c in b]
    lat0, lat1, lon0, lon1 = min(lats), max(lats), min(lons), max(lons)
    dlat = spacing_m / 111_320.0
    dlon = spacing_m / (111_320.0 * max(math.cos(math.radians((lat0 + lat1) / 2)), 0.2))
    la = np.arange(lat0 + dlat / 2, lat1, dlat)
    lo = np.arange(lon0 + dlon / 2, lon1, dlon)
    pts = np.array([(a, o) for a in la for o in lo], float)
    if len(pts) > max_points:                       # thin evenly, never randomly biased
        step = int(np.ceil(len(pts) / max_points))
        pts = pts[::step]
    return pts


def _score(bundle: dict, X: np.ndarray) -> np.ndarray:
    return bundle["model"].predict_proba(X)[:, 1]


def _explain_batch(bundle: dict, X: np.ndarray, top_k: int = 4) -> list[list[dict]]:
    """Per-cell reasons: move one feature to its median and see how the score shifts.

    Exact for a forest, and it answers the question a health worker actually asks,
    which is 'why here?' rather than 'how important is this feature in general?'
    """
    model, feats, med = bundle["model"], bundle["features"], np.array(bundle["medians"])
    full = model.predict_proba(X)[:, 1]
    effects = np.zeros((len(X), len(feats)))
    for j in range(len(feats)):
        swapped = X.copy()
        swapped[:, j] = med[j]
        effects[:, j] = full - model.predict_proba(swapped)[:, 1]
    out = []
    for i in range(len(X)):
        order = np.argsort(-np.abs(effects[i]))[:top_k]
        out.append([{"feature": feats[j], "effect": round(float(effects[i, j]), 4),
                     "value": round(float(X[i, j]), 3),
                     "direction": "raises" if effects[i, j] > 0 else "lowers"} for j in order])
    return out


PLAIN = {
    "NDWI": "surface water signal", "MNDWI": "standing water signal",
    "NDVI": "vegetation density", "slope_deg": "ground slope",
    "hollow_m": "local dip where water collects", "dist_water_m": "distance to permanent water",
    "rain_season_mm": "seasonal rainfall", "rain_30d_mm": "rainfall in the last 30 days",
    "temp_c": "average temperature", "dew_c": "dew point", "humidity_pct": "humidity",
    "elev_m": "elevation", "water_occurrence": "how often this place holds water",
}


def phrase(reason: dict) -> str:
    """Value and effect in separate clauses, so the sentence cannot be misread."""
    from .priority import phrase_reason
    return phrase_reason(reason)


def build_risk_map(cfg: dict, model_path: Path, out_dir: Path,
                   lulc_model: Path | None = None, dev: str = "cpu") -> dict:
    import joblib

    f = cfg["fusion"]
    s = cfg["study_area"]
    bundle = joblib.load(model_path)
    want = list(bundle["features"])
    needs_lulc = any(w.startswith("lulc_") for w in want)
    print(f"model expects {len(want)} features" + (" (including land cover)" if needs_lulc else ""))

    ee = init_ee(cfg)
    geom, name = district_geometry(ee, s["state"], s["district"])
    area_km2 = geom.area().divide(1e6).getInfo()
    print(f"district: {name}, {area_km2:,.0f} km2")
    stack, n_scenes = build_stack(ee, cfg, geom)
    print(f"  Sentinel-2 scenes in the season window: {n_scenes:,}")

    def score_points(pts, label):
        """Build exactly the feature matrix the trained model was given, in its order.

        Assembling by name rather than by position means a model trained with land
        cover cannot be fed a matrix without it, which would otherwise fail silently
        and produce confident nonsense.
        """
        print(f"sampling {len(pts):,} {label} points...")
        X, _, kept = sample_points(ee, stack, pts, np.zeros(len(pts), int), scale=int(s["grid_m"]))
        if len(X) == 0:
            raise SystemExit(f"No {label} points returned data. Check the season window.")
        cols = {name: X[:, i] for i, name in enumerate(FEATURES)}
        cols["humidity_pct"] = relative_humidity(cols["temp_c"], cols["dew_c"])
        coords = pts[kept]

        if needs_lulc:
            from ..risk.landcover import fractions_for_points
            print(f"  land-cover chips for {len(coords):,} {label} points...")
            fr, lk, names = fractions_for_points(cfg, ee, stack, coords, lulc_model, dev,
                                                 0, 0)
            if fr is None:
                raise SystemExit("Land-cover chips failed, but the model needs them.")
            keep = np.zeros(len(coords), bool); keep[lk] = True
            for k, n in enumerate(names):
                v = np.full(len(coords), np.nan)
                v[lk] = fr[:, k]
                cols[n] = v
            cols = {k: v[keep] for k, v in cols.items()}
            coords = coords[keep]

        missing = [w for w in want if w not in cols]
        if missing:
            raise SystemExit(f"The model needs features this stage cannot build: {missing}")
        return np.column_stack([cols[w] for w in want]), coords

    # ---- Tier 1: coarse pass over the whole district
    coarse_pts = _grid_points(ee, geom, f["coarse_spacing_m"], f["max_coarse_points"])
    Xc, coords_c = score_points(coarse_pts, "coarse")
    sc = _score(bundle, Xc)
    order = np.argsort(-sc)[:int(f["top_zones"])]
    print(f"  coarse scores: median {np.median(sc):.3f}, top {sc[order[0]]:.3f}; "
          f"taking the best {len(order)} zones for the fine pass")

    # ---- Tier 2: fine pass inside the best zones only
    dlat = f["coarse_spacing_m"] / 111_320.0
    fine_pts = []
    per_zone = max(1, int(f["coarse_spacing_m"] / s["grid_m"]))
    for i in order:
        lat, lon = coords_c[i]
        dlon = f["coarse_spacing_m"] / (111_320.0 * max(math.cos(math.radians(lat)), 0.2))
        gy = np.linspace(lat - dlat / 2, lat + dlat / 2, per_zone)
        gx = np.linspace(lon - dlon / 2, lon + dlon / 2, per_zone)
        fine_pts += [(a, o) for a in gy for o in gx]
    fine_pts = np.array(fine_pts, float)
    if len(fine_pts) > f["max_fine_points"]:
        fine_pts = fine_pts[:: int(np.ceil(len(fine_pts) / f["max_fine_points"]))]
    Xf, coords_f = score_points(fine_pts, "fine")
    sf = _score(bundle, Xf)
    reasons = _explain_batch(bundle, Xf)

    q = np.quantile(sf, [0.5, 0.75, 0.9])
    band = lambda v: ("critical" if v >= q[2] else "high" if v >= q[1]
                      else "moderate" if v >= q[0] else "low")
    fi = {k: i for i, k in enumerate(want)}
    cells = []
    for i in range(len(Xf)):
        row = Xf[i]
        cells.append({
            "lat": round(float(coords_f[i][0]), 5), "lon": round(float(coords_f[i][1]), 5),
            "suitability": round(float(sf[i]), 4), "severity": band(sf[i]),
            "ndwi": round(float(row[fi["NDWI"]]), 3),
            "slope": round(float(row[fi["slope_deg"]]), 2),
            "elev": round(float(row[fi["elev_m"]]), 1),
            "rain": round(float(row[fi["rain_season_mm"]]), 1),
            "temp": round(float(row[fi["temp_c"]]), 1),
            "dist_water": round(float(row[fi["dist_water_m"]]), 1),
            "humidity": round(float(row[fi["humidity_pct"]]), 1),
            "reasons": [{**r, "text": phrase(r)} for r in reasons[i]],
        })

    cell_area_km2 = (s["grid_m"] / 1000.0) ** 2
    meta = {
        "district": name, "state": s["state"], "district_area_km2": round(area_km2),
        "season": f"{s['season_start']} to {s['season_end']}", "s2_scenes": n_scenes,
        "coarse": {"spacing_m": f["coarse_spacing_m"], "points_scored": int(len(sc)),
                   "zones_selected": int(len(order))},
        "fine": {"spacing_m": s["grid_m"], "cells_scored": int(len(cells)),
                 "area_covered_km2": round(len(cells) * cell_area_km2, 1),
                 "district_coverage_pct": round(100 * len(cells) * cell_area_km2 / area_km2, 2)},
        "score_meaning": "Relative habitat suitability from the Stage 5 model: how much the "
                         "conditions here resemble places where Anopheles vectors were recorded. "
                         "It is NOT a probability that larvae are present.",
        "severity_bands": {"moderate": round(float(q[0]), 4), "high": round(float(q[1]), 4),
                           "critical": round(float(q[2]), 4),
                           "note": "Bands are quantiles of the scored cells, so they rank within "
                                   "this district rather than against an absolute scale."},
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "risk_map.json").write_text(json.dumps({"meta": meta, "cells": cells}))
    print(f"\nfine grid: {len(cells):,} cells covering {meta['fine']['area_covered_km2']} km2 "
          f"({meta['fine']['district_coverage_pct']}% of the district)")
    return {"meta": meta, "cells": cells}
