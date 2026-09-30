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
            gp, source = (survey_cell["lat"], survey_cell["lon"]), "survey cell"
        if gp is None:
            continue                              # no location: cannot be a GPS target

        cell, dist_km = nearest_cell(cells, gp[0], gp[1])
        S = cell["suitability"] if cell and dist_km < f["max_cell_distance_km"] else f["default_suitability"]
        idx = priority_index(S, pool["confidence"], pool["frames_seen"], pool["area_m2"], ref)
        reasons = [r["text"] for r in (cell["reasons"] if cell else [])][:3]
        reasons.append(f"drone saw {pool['area_m2']} m2 of standing water across "
                       f"{pool['frames_seen']} frames")
        targets.append({
            "lat": gp[0], "lon": gp[1], "location_source": source,
            "area_m2": pool["area_m2"], "confidence": pool["confidence"],
            "frames_seen": pool["frames_seen"],
            "t_start_s": pool.get("t_start_s"), "t_end_s": pool.get("t_end_s"),
            "matched_cell_km": round(dist_km, 3) if cell else None,
            **idx, "reasons": reasons,
        })

    targets.sort(key=lambda t: -t["priority_index"])
    order = route_order(targets) if targets else []
    for rank, i in enumerate(order, 1):
        targets[i]["visit_order"] = rank

    return {
        "district": risk_map["meta"]["district"],
        "targets_found": len(targets),
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
             "severity": c["severity"], "reasons": [r["text"] for r in c["reasons"][:3]]}
            for c in cells]
