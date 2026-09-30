"""Stage 9a — will the payload land within 1 metre of the puddle?

The proposal sets a 1 m targeting requirement. That is a physics question, and it can
be answered before any chemical, or even any flight, by simulating the drop.

A released capsule keeps the drone's forward speed, so it does not fall straight down:
it travels forward while it falls. The release point must therefore be BEFORE the
target by that distance. The simulation computes that lead distance, then asks how far
the landing spot scatters once real-world error is added:

  wind            pushes the capsule sideways during the fall
  GPS position    the drone does not know exactly where it is
  release latency the servo does not open the instant the command is sent
  altitude error  the barometer drifts

Drag matters. A water capsule is light and slow, so treating it as a vacuum projectile
would underestimate the fall time and mis-place the release point.
"""
from __future__ import annotations

import math

import numpy as np

G = 9.81
RHO_AIR = 1.225          # kg/m3 at sea level


def fall_with_drag(height_m: float, v_forward: float, mass_kg: float, diameter_m: float,
                   cd: float = 0.47, wind_ms: float = 0.0, dt: float = 0.001) -> dict:
    """Integrate the fall, so drag is included rather than assumed away.

    cd = 0.47 is the standard drag coefficient of a sphere, which is the right shape
    for a water capsule.
    """
    area = math.pi * (diameter_m / 2) ** 2
    k = 0.5 * RHO_AIR * cd * area / max(mass_kg, 1e-6)

    x, y, z = 0.0, 0.0, float(height_m)          # x forward, y crosswind, z up
    vx, vy, vz = float(v_forward), 0.0, 0.0
    t = 0.0
    while z > 0 and t < 30:
        # air-relative velocity: wind blows across the flight path
        ax_rel, ay_rel, az_rel = vx, vy - wind_ms, vz
        speed = math.sqrt(ax_rel ** 2 + ay_rel ** 2 + az_rel ** 2) + 1e-12
        ax = -k * speed * ax_rel
        ay = -k * speed * ay_rel
        az = -G - k * speed * az_rel
        vx += ax * dt; vy += ay * dt; vz += az * dt
        x += vx * dt; y += vy * dt; z += vz * dt
        t += dt
    return {"fall_time_s": round(t, 3), "forward_travel_m": round(x, 3),
            "crosswind_drift_m": round(y, 3), "impact_speed_ms": round(abs(vz), 2)}


def release_lead(height_m: float, v_forward: float, mass_kg: float, diameter_m: float,
                 cd: float = 0.47) -> float:
    """How far before the target the servo must open, in still air."""
    return fall_with_drag(height_m, v_forward, mass_kg, diameter_m, cd)["forward_travel_m"]


def simulate_drops(cfg_d: dict, n: int = 2000, seed: int = 42) -> dict:
    """Monte Carlo: scatter of landing points once real errors are included."""
    rng = np.random.default_rng(seed)
    h = float(cfg_d["release_height_m"])
    v = float(cfg_d["forward_speed_ms"])
    m = float(cfg_d["payload_mass_kg"])
    d = float(cfg_d["payload_diameter_m"])
    cd = float(cfg_d.get("drag_coefficient", 0.47))

    lead = release_lead(h, v, m, d, cd)          # aim point, computed in still air
    errs = []
    for _ in range(n):
        wind = rng.normal(0, float(cfg_d["wind_sigma_ms"]))
        h_err = rng.normal(0, float(cfg_d["altitude_sigma_m"]))
        gps_e = rng.normal(0, float(cfg_d["gps_sigma_m"]), 2)
        lat_s = max(0.0, rng.normal(float(cfg_d["release_latency_s"]),
                                    float(cfg_d["latency_sigma_s"])))
        v_err = rng.normal(v, float(cfg_d["speed_sigma_ms"]))

        r = fall_with_drag(max(h + h_err, 1.0), v_err, m, d, cd, wind_ms=wind, dt=0.002)
        along = r["forward_travel_m"] + v_err * lat_s - lead     # overshoot if positive
        across = r["crosswind_drift_m"]
        errs.append(math.hypot(along + gps_e[0], across + gps_e[1]))

    e = np.array(errs)
    return {"release_lead_m": round(lead, 2),
            "fall_time_s": fall_with_drag(h, v, m, d, cd)["fall_time_s"],
            "impact_speed_ms": fall_with_drag(h, v, m, d, cd)["impact_speed_ms"],
            "trials": n,
            "mean_error_m": round(float(e.mean()), 2),
            "median_error_m": round(float(np.median(e)), 2),
            "cep50_m": round(float(np.percentile(e, 50)), 2),
            "cep90_m": round(float(np.percentile(e, 90)), 2),
            "within_1m_pct": round(100 * float((e <= 1.0).mean()), 1),
            "within_2m_pct": round(100 * float((e <= 2.0).mean()), 1),
            "meets_1m_requirement": bool(np.percentile(e, 50) <= 1.0)}


def sweep_conditions(cfg_d: dict, heights=(10, 15, 20, 25, 30), speeds=(0, 2, 4, 6),
                     n: int = 600, seed: int = 42) -> list[dict]:
    """Which height and speed actually meet the 1 m requirement?"""
    out = []
    for h in heights:
        for v in speeds:
            c = dict(cfg_d, release_height_m=h, forward_speed_ms=v)
            r = simulate_drops(c, n=n, seed=seed)
            out.append({"height_m": h, "speed_ms": v, "lead_m": r["release_lead_m"],
                        "cep50_m": r["cep50_m"], "within_1m_pct": r["within_1m_pct"]})
    return out


def error_budget(cfg_d: dict, n: int = 800, seed: int = 42) -> list[dict]:
    """Turn each error source off in turn, to see which one dominates.

    This says where to spend effort: a better GPS, a calmer day, or a faster servo.
    """
    base = simulate_drops(cfg_d, n=n, seed=seed)["cep50_m"]
    rows = [{"source": "all sources", "cep50_m": base, "improvement_m": 0.0}]
    for key, label in (("wind_sigma_ms", "no wind"), ("gps_sigma_m", "perfect GPS"),
                       ("latency_sigma_s", "no latency jitter"),
                       ("altitude_sigma_m", "exact altitude"),
                       ("speed_sigma_ms", "exact speed")):
        c = dict(cfg_d); c[key] = 0.0
        r = simulate_drops(c, n=n, seed=seed)["cep50_m"]
        rows.append({"source": label, "cep50_m": r, "improvement_m": round(base - r, 2)})
    return sorted(rows, key=lambda x: -x["improvement_m"])
