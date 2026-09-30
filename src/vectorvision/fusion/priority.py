"""Stage 8b — the Breeding Site Priority Index.

How the index avoids arbitrary weights
--------------------------------------
Each ranked site is described by three measured quantities, and the index is their
product. Nothing here is a tuned coefficient:

    PBI = S x C x A

    S  habitat suitability, the Stage 5 model's score for that cell (0-1)
    C  drone confirmation, the detector's mean confidence for that pool, scaled by
       how many frames it survived (0-1)
    A  larval capacity, pool area divided by a reference area, capped at 1

A product is the right shape because the three conditions must ALL hold for a site to
matter. A large confirmed pool in unsuitable terrain, or suitable terrain with no
water, should both rank low, and multiplying achieves that without anyone choosing
how much each factor is "worth". A weighted sum would not: it lets a high score on
one factor compensate for a zero on another.

The reference area is a stated physical constant (a pool of this size is treated as
full larval capacity), not a fitted parameter. It only sets the scale of A, and
because ranking is invariant to it, changing it cannot reorder two sites that share
the same S and C.

What the index is and is not
----------------------------
It is a RANKING for deciding where a health worker goes first. It is not a
probability that larvae are present. S comes from a presence-background model with an
artificial 1:5 sampling ratio, so it measures relative suitability, not real-world
prevalence. Only a field visit confirms larvae.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

PLAIN = {
    "NDWI": "surface water signal", "MNDWI": "standing water signal",
    "NDVI": "vegetation density", "slope_deg": "ground slope",
    "hollow_m": "local dip where water collects", "dist_water_m": "distance to permanent water",
    "rain_season_mm": "seasonal rainfall", "rain_30d_mm": "rainfall in the last 30 days",
    "temp_c": "average temperature", "dew_c": "dew point", "humidity_pct": "humidity",
    "elev_m": "elevation", "water_occurrence": "how often this place holds water",
}
UNITS = {"rain_season_mm": " mm", "rain_30d_mm": " mm", "temp_c": " C", "dew_c": " C",
         "humidity_pct": "%", "elev_m": " m", "slope_deg": " deg", "dist_water_m": " m",
         "hollow_m": " m"}


def phrase_reason(r: dict) -> str:
    """Keep the measured value and its effect in separate clauses.

    'standing water signal (-0.288) raises the score' reads as a contradiction; the
    number is the reading, not the effect. Splitting them removes the ambiguity.
    """
    name = PLAIN.get(r["feature"], r["feature"].replace("_", " "))
    val = r.get("value")
    unit = UNITS.get(r["feature"], "")
    verb = "raises" if r.get("direction", "raises") == "raises" else "lowers"
    if val is None:
        return f"{name} {verb} the score"
    return f"{name} here is {val}{unit}, which {verb} the score"


# ------------------------------------------------------------------ index
def confirmation(confidence: float, frames_seen: int, full_at_frames: int = 12) -> float:
    """Drone evidence: how sure the detector was, tempered by how long it persisted.

    A pool glimpsed in three frames is weaker evidence than the same pool tracked
    across thirty, even at equal confidence.
    """
    c = float(np.clip(confidence, 0.0, 1.0))
    persist = float(np.clip(frames_seen / max(full_at_frames, 1), 0.0, 1.0))
    return c * (0.5 + 0.5 * persist)          # persistence can halve, never inflate


def capacity(area_m2: float, reference_m2: float = 50.0) -> float:
    """Larval capacity rises with water surface area and saturates."""
    return float(np.clip(area_m2 / max(reference_m2, 1e-6), 0.0, 1.0))


def priority_index(suitability: float, confidence: float, frames_seen: int,
                   area_m2: float, reference_m2: float = 50.0) -> dict:
    S = float(np.clip(suitability, 0.0, 1.0))
    C = confirmation(confidence, frames_seen)
    A = capacity(area_m2, reference_m2)
    pbi = S * C * A
    band = ("critical" if pbi > 0.40 else "high" if pbi > 0.20
            else "moderate" if pbi > 0.08 else "low")
    return {"priority_index": round(pbi, 4), "band": band,
            "factors": {"habitat_suitability": round(S, 3),
                        "drone_confirmation": round(C, 3),
                        "larval_capacity": round(A, 3)}}


# ------------------------------------------------------------------ geometry
def project_to_ground(lat: float, lon: float, alt_m: float, heading_deg: float | None,
                      fx: float, fy: float, hfov_deg: float, aspect: float):
    """Where on the ground is a detection that sits at (fx, fy) in the frame?

    Assumes the camera points straight down over level ground. Tilt introduces error
    that grows with angle, so the gimbal must stay level during survey passes.
    """
    if lat is None or alt_m is None or alt_m < 1:
        return None
    half_w = alt_m * math.tan(math.radians(hfov_deg / 2))
    half_h = half_w / max(aspect, 1e-6)
    east = (fx - 0.5) * 2 * half_w
    north = -(fy - 0.5) * 2 * half_h
    if heading_deg is not None:
        t = math.radians(heading_deg)
        east, north = (east * math.cos(t) + north * math.sin(t),
                       -east * math.sin(t) + north * math.cos(t))
    return (round(lat + north / 111_320.0, 6),
            round(lon + east / (111_320.0 * math.cos(math.radians(lat))), 6))


def offset_from_cell(lat: float, lon: float, fx: float, fy: float,
                    alt_m: float, hfov_deg: float, aspect: float = 1.0):
    """Approximate ground position when the flight had no telemetry.

    Without a GPS fix every pool would collapse onto the survey centre, which
    destroys the route and gives every pool the same suitability. The frame position
    still says where a pool sat relative to the camera, so it is offset by that much
    from the centre. Heading is unknown, so the offset is only as good as the
    assumption that the drone was pointing north; positions are marked approximate.
    """
    return project_to_ground(lat, lon, alt_m, None, fx, fy, hfov_deg, aspect)


def merge_nearby(targets: list[dict], min_sep_m: float) -> list[dict]:
    """One water body seen as several blobs is still one site to visit.

    Detections closer together than the separation distance are merged, keeping the
    largest area and the strongest evidence, so a health worker gets a list of places
    rather than a list of pixels.
    """
    kept: list[dict] = []
    for t in sorted(targets, key=lambda x: -x["area_m2"]):
        dup = False
        for k in kept:
            dlat = (k["lat"] - t["lat"]) * 111_320.0
            dlon = (k["lon"] - t["lon"]) * 111_320.0 * math.cos(math.radians(k["lat"]))
            if math.hypot(dlat, dlon) < min_sep_m:
                k["merged_detections"] = k.get("merged_detections", 1) + 1
                k["area_m2"] = round(k["area_m2"] + t["area_m2"], 2)
                k["frames_seen"] = max(k["frames_seen"], t["frames_seen"])
                k["confidence"] = round(max(k["confidence"], t["confidence"]), 3)
                dup = True
                break
        if not dup:
            kept.append(dict(t, merged_detections=1))
    return kept


def nearest_cell(cells: list[dict], lat: float, lon: float):
    """The scored grid cell a pool falls in, for its suitability and reasons."""
    best, bd = None, float("inf")
    for c in cells:
        d = (c["lat"] - lat) ** 2 + (c["lon"] - lon) ** 2
        if d < bd:
            bd, best = d, c
    return best, math.sqrt(bd) * 111.32          # km


def route_order(targets: list[dict], start: tuple | None = None) -> list[int]:
    """Nearest-neighbour ordering, so the drone flies a sensible circuit.

    Not the optimal tour, but a short one computed instantly, which is the right
    trade for a battery-limited survey with a handful of targets.
    """
    left = list(range(len(targets)))
    cur = start or (targets[0]["lat"], targets[0]["lon"])
    order = []
    while left:
        j = min(left, key=lambda i: (targets[i]["lat"] - cur[0]) ** 2 + (targets[i]["lon"] - cur[1]) ** 2)
        order.append(j)
        cur = (targets[j]["lat"], targets[j]["lon"])
        left.remove(j)
    return order


def route_length_km(targets: list[dict], order: list[int]) -> float:
    total = 0.0
    for a, b in zip(order, order[1:]):
        dlat = (targets[b]["lat"] - targets[a]["lat"]) * 111.32
        dlon = ((targets[b]["lon"] - targets[a]["lon"]) * 111.32
                * math.cos(math.radians(targets[a]["lat"])))
        total += math.hypot(dlat, dlon)
    return total


# ------------------------------------------------------------------ fusion
def fuse(risk_map: dict, video: dict, cfg: dict, survey_cell=None) -> dict:
    """Combine the district risk map with one flight's confirmed pools."""
    f = cfg["fusion"]
    v = cfg["video"]
    cells = risk_map["cells"]
    ref = float(f["reference_pool_m2"])
    aspect = float(f.get("frame_aspect", 1.0))

    targets = []
    for pool in video.get("pools", []):
        fix = pool.get("gps")                    # present when telemetry was connected
        gp = None
        if fix:
            gp = project_to_ground(fix.get("lat"), fix.get("lon"), fix.get("alt_m"),
                                   fix.get("hdg_deg"), pool["frame_x"], pool["frame_y"],
                                   float(v["hfov_deg"]), aspect)
        source = "telemetry"
        if gp is None and survey_cell:
            gp = offset_from_cell(survey_cell["lat"], survey_cell["lon"],
                                  pool["frame_x"], pool["frame_y"],
                                  float(v["altitude_m"]), float(v["hfov_deg"]), aspect)
            source = "survey cell + frame offset (approximate)"
        if gp is None:
            continue                              # no location: cannot be a GPS target

        cell, dist_km = nearest_cell(cells, gp[0], gp[1])
        S = cell["suitability"] if cell and dist_km < f["max_cell_distance_km"] else f["default_suitability"]
        idx = priority_index(S, pool["confidence"], pool["frames_seen"], pool["area_m2"], ref)
        reasons = [phrase_reason(r) for r in (cell["reasons"] if cell else [])][:3]
        targets.append({
            "lat": gp[0], "lon": gp[1], "location_source": source,
            "area_m2": pool["area_m2"], "confidence": pool["confidence"],
            "frames_seen": pool["frames_seen"],
            "t_start_s": pool.get("t_start_s"), "t_end_s": pool.get("t_end_s"),
            "matched_cell_km": round(dist_km, 3) if cell else None,
            **idx, "reasons": reasons,
        })

    n_raw = len(targets)
    targets = merge_nearby(targets, float(f.get("merge_distance_m", 25)))
    # Merging changes the area, so the index and the wording must be recomputed from
    # the merged figures, or the printed area and the stated reason disagree.
    big = float(f.get("large_body_m2", 500))
    for t in targets:
        t.update(priority_index(t["factors"]["habitat_suitability"], t["confidence"],
                                t["frames_seen"], t["area_m2"], ref))
        seen = f"drone saw {t['area_m2']} m2 of standing water across {t['frames_seen']} frames"
        if t.get("merged_detections", 1) > 1:
            seen += f" ({t['merged_detections']} detections merged)"
        t["reasons"] = list(t["reasons"]) + [seen]
        # A lake is not a puddle. Anopheles larvae favour small, shallow, sunlit
        # water; large permanent bodies usually hold fish and wave action. The index
        # is left alone (no invented penalty) but the site is flagged for the operator.
        t["likely_permanent_water"] = bool(t["area_m2"] >= big)
        if t["likely_permanent_water"]:
            t["reasons"].append("large water body: likely permanent, check before treating "
                                "as a breeding puddle")
    targets.sort(key=lambda t: -t["priority_index"])
    order = route_order(targets) if targets else []
    for rank, i in enumerate(order, 1):
        targets[i]["visit_order"] = rank

    bands = {b: sum(1 for t in targets if t["band"] == b)
             for b in ("critical", "high", "moderate", "low")}
    sui = [t["factors"]["habitat_suitability"] for t in targets]
    notes = []
    if targets and max(sui) - min(sui) < 0.01:
        notes.append("Every target shares one suitability value, so the flight stayed inside "
                     "a single scored cell. Fly a wider pass, or use telemetry, to let the "
                     "satellite layer separate them.")
    if targets and max(sui) < 0.2:
        notes.append("Suitability is low everywhere in this flight: the drone found water in "
                     "terrain the satellite model rates as poor habitat. That is the index "
                     "working, not a bug, but it is worth flying one of the top-ranked zones.")
    n_big = sum(1 for t in targets if t.get("likely_permanent_water"))
    if n_big:
        notes.append(f"{n_big} target(s) are larger than {int(big)} m2 and are probably "
                     "permanent water, not breeding puddles. Anopheles prefer small sunlit "
                     "pools, so check these before treating them as priority sites.")
    if any(t["location_source"].startswith("survey cell") for t in targets):
        notes.append("Positions are approximate: no telemetry, so they were placed by frame "
                     "offset from the survey centre assuming a north-facing camera.")

    return {
        "district": risk_map["meta"]["district"],
        "detections_before_merge": n_raw,
        "targets_found": len(targets),
        "bands": bands, "notes": notes,
        "route_km": round(route_length_km(targets, order), 3) if len(order) > 1 else 0.0,
        "reference_pool_m2": ref,
        "targets": targets,
        "index_definition": "PBI = habitat suitability x drone confirmation x larval capacity. "
                            "A product, because all three must hold; no fitted weights.",
        "caveat": "A ranking for prioritising visits, not a probability that larvae are present.",
    }


def top_zones(risk_map: dict, n: int = 10) -> list[dict]:
    """Where to fly next, from the satellite side alone, before any drone data."""
    cells = sorted(risk_map["cells"], key=lambda c: -c["suitability"])[:n]
    return [{"lat": c["lat"], "lon": c["lon"], "suitability": c["suitability"],
             "severity": c["severity"],
             "reasons": [phrase_reason(r) for r in c["reasons"][:3]]}
            for c in cells]
