"""Stage 9b — turn ranked targets into a flight the drone can actually fly.

Produces a QGroundControl .plan file with a waypoint over each target and a servo
command to release the payload, plus a feasibility check: can this aircraft reach
every target on one battery, and is the plan legal and safe?

Nothing here commands an autonomous flight. The proposal's drone is flown manually,
so the plan is a route for the pilot, and the drop is water only.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

# MAVLink command numbers used in a QGroundControl plan
NAV_WAYPOINT = 16
DO_SET_SERVO = 183
NAV_RTL = 20


def haversine_km(a: tuple, b: tuple) -> float:
    lat1, lon1 = a
    lat2, lon2 = b
    dlat = (lat2 - lat1) * 111.32
    dlon = (lon2 - lon1) * 111.32 * math.cos(math.radians((lat1 + lat2) / 2))
    return math.hypot(dlat, dlon)


def feasibility(targets: list[dict], home: tuple, cfg_d: dict) -> dict:
    """Can one battery cover this route, with reserve left?

    Endurance is stated in the config from the pilot's own measurement, not guessed
    here: it depends on the battery, the payload and the day.
    """
    speed = float(cfg_d["cruise_speed_ms"])
    hover_s = float(cfg_d["hover_per_target_s"])
    endurance = float(cfg_d["endurance_min"]) * 60.0
    reserve = float(cfg_d.get("reserve_fraction", 0.25))

    pts = [home] + [(t["lat"], t["lon"]) for t in targets] + [home]
    legs = [haversine_km(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]
    dist_km = sum(legs)
    fly_s = dist_km * 1000.0 / max(speed, 0.1)
    total_s = fly_s + hover_s * len(targets)
    usable = endurance * (1 - reserve)

    n_ok = 0
    acc = 0.0
    for i, t in enumerate(targets):
        acc += legs[i] * 1000.0 / max(speed, 0.1) + hover_s
        back = haversine_km((t["lat"], t["lon"]), home) * 1000.0 / max(speed, 0.1)
        if acc + back <= usable:
            n_ok = i + 1

    return {"targets": len(targets), "route_km": round(dist_km, 3),
            "flight_time_min": round(total_s / 60, 1),
            "usable_endurance_min": round(usable / 60, 1),
            "reachable_targets": n_ok,
            "fits_one_battery": n_ok >= len(targets),
            "note": ("All targets fit one battery with reserve." if n_ok >= len(targets)
                     else f"Only {n_ok} of {len(targets)} targets fit. Split into sorties "
                          "or move the launch point closer.")}


def safety_check(targets: list[dict], home: tuple, cfg_d: dict) -> list[dict]:
    """Checks that must pass before any outdoor flight."""
    max_km = float(cfg_d["max_range_km"])
    checks = []

    far = [t for t in targets if haversine_km((t["lat"], t["lon"]), home) > max_km]
    checks.append({"check": "within visual line of sight",
                   "pass": not far,
                   "detail": f"{len(far)} target(s) beyond {max_km} km from launch"
                             if far else f"all targets within {max_km} km"})

    perm = [t for t in targets if t.get("likely_permanent_water")]
    checks.append({"check": "targets are puddles, not permanent water",
                   "pass": not perm,
                   "detail": f"{len(perm)} target(s) look like permanent water; "
                             "confirm on foot first" if perm else "no large water bodies"})

    approx = [t for t in targets if str(t.get("location_source", "")).startswith("survey cell")]
    checks.append({"check": "positions come from telemetry",
                   "pass": not approx,
                   "detail": f"{len(approx)} target(s) are approximate frame offsets; "
                             "connect the telemetry radio" if approx else "all from telemetry"})

    checks.append({"check": "payload is water only",
                   "pass": str(cfg_d.get("payload", "water")).lower() == "water",
                   "detail": f"payload set to '{cfg_d.get('payload', 'water')}'"})

    checks.append({"check": "drone registered and permissions held",
                   "pass": bool(cfg_d.get("registered", False)),
                   "detail": "DGCA Micro category: registration required, and a payload "
                             "drop needs specific permission. Set registered: true in "
                             "config once both are in place."})
    return checks


def build_plan(targets: list[dict], home: tuple, cfg_d: dict, lead_m: float) -> dict:
    """QGroundControl .plan: fly to each target, release, close, return.

    The release point is placed `lead_m` BEFORE the target along the approach, because
    the capsule keeps the drone's forward speed while it falls.
    """
    alt = float(cfg_d["release_height_m"])
    servo = int(cfg_d["servo_channel"])
    open_us = int(cfg_d["servo_open_us"])
    shut_us = int(cfg_d["servo_closed_us"])

    items, seq = [], 1

    def add(cmd, params):
        nonlocal seq
        items.append({"type": "SimpleItem", "command": cmd, "frame": 3 if cmd == NAV_WAYPOINT else 2,
                      "autoContinue": True, "doJumpId": seq, "params": params})
        seq += 1

    prev = home
    for t in targets:
        tgt = (t["lat"], t["lon"])
        # back off along the approach line so the capsule's forward travel is cancelled
        d = haversine_km(prev, tgt) * 1000.0
        if d > 1e-6:
            f = max(0.0, (d - lead_m) / d)
            rel = (prev[0] + (tgt[0] - prev[0]) * f, prev[1] + (tgt[1] - prev[1]) * f)
        else:
            rel = tgt
        add(NAV_WAYPOINT, [0, 0, 0, None, round(rel[0], 7), round(rel[1], 7), alt])
        add(DO_SET_SERVO, [servo, open_us, 0, 0, 0, 0, 0])
        add(DO_SET_SERVO, [servo, shut_us, 0, 0, 0, 0, 0])
        add(NAV_WAYPOINT, [0, 0, 0, None, round(tgt[0], 7), round(tgt[1], 7), alt])
        prev = tgt
    add(NAV_RTL, [0, 0, 0, 0, 0, 0, 0])

    return {"fileType": "Plan", "version": 1, "groundStation": "QGroundControl",
            "mission": {"version": 2, "firmwareType": 12, "vehicleType": 2,
                        "cruiseSpeed": float(cfg_d["cruise_speed_ms"]),
                        "hoverSpeed": float(cfg_d.get("hover_speed_ms", 3)),
                        "items": items,
                        "plannedHomePosition": [home[0], home[1], 0]},
            "geoFence": {"version": 2, "circles": [], "polygons": []},
            "rallyPoints": {"version": 2, "points": []}}


def plan_mission(cfg: dict, targets_path: Path, out_dir: Path, home: tuple | None) -> dict:
    from .drop_sim import error_budget, simulate_drops, sweep_conditions

    d = cfg["mission"]
    data = json.loads(targets_path.read_text())
    targets = data.get("targets", [])[: int(cfg["fusion"]["top_targets"])]
    if not targets:
        raise SystemExit("No targets in targets.json. Run Stage 8 with a flight video first.")
    if home is None:
        home = (targets[0]["lat"], targets[0]["lon"])

    drops = simulate_drops(d, n=int(d.get("trials", 2000)), seed=cfg["project"]["seed"])
    sweep = sweep_conditions(d, n=400, seed=cfg["project"]["seed"])
    budget = error_budget(d, n=600, seed=cfg["project"]["seed"])
    feas = feasibility(targets, home, d)
    checks = safety_check(targets, home, d)
    plan = build_plan(targets, home, d, drops["release_lead_m"])

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "mission.plan").write_text(json.dumps(plan, indent=2))
    report = {"home": list(home), "targets_in_plan": len(targets),
              "drop_simulation": drops, "condition_sweep": sweep,
              "error_budget": budget, "feasibility": feas, "safety_checks": checks,
              "note": "Simulated water-only drop. No autonomous flight is commanded; the "
                      "plan is a route for a manual pilot."}
    (out_dir / "mission_report.json").write_text(json.dumps(report, indent=2))
    return report
