"""Tests for the parts of Vector Vision that can be wrong silently.

These check arithmetic and contracts, not model accuracy. A model that scores badly
tells you so; a ground-scale conversion that is wrong by a factor of four, or a
feature matrix in the wrong column order, produces confident numbers that look fine.
Those are what is tested here.

Run:  python -m pytest tests/ -v        (or: python tests/test_pipeline.py)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vectorvision.fusion.priority import (band_of, capacity, confirmation, merge_nearby,
                                          priority_index, project_to_ground, route_length_km,
                                          route_order)
from vectorvision.risk.occurrence import background_points, thin_points
from vectorvision.risk.schema import SchemaError, risk_score_100, spread_out, validate
from vectorvision.vision.video import (ground_sample_distance, persistence_vote, pixels_to_m2,
                                       PoolTracker)


# ───────────────────────────────────────────────── geometry
def test_ground_sample_distance():
    """At 30 m with an 80 deg camera the frame is 2*30*tan(40) = 50.3 m wide."""
    g = ground_sample_distance(30, 80, 512)
    assert abs(g * 512 - 2 * 30 * math.tan(math.radians(40))) < 0.01
    assert abs(g - 0.0983) < 0.001


def test_area_scales_with_height_squared():
    """Double the height, quadruple the ground area of the same pixel count."""
    a15 = pixels_to_m2(1000, ground_sample_distance(15, 80, 512))
    a30 = pixels_to_m2(1000, ground_sample_distance(30, 80, 512))
    assert abs(a30 / a15 - 4.0) < 0.01


def test_projection_matches_geometry():
    """A detection at the right edge lands half a frame width east."""
    lat, lon, alt = 21.45, 80.19, 30.0
    gp = project_to_ground(lat, lon, alt, 0.0, 1.0, 0.5, 80, 1.0)
    east_m = (gp[1] - lon) * 111320 * math.cos(math.radians(lat))
    assert abs(east_m - alt * math.tan(math.radians(40))) < 0.2


def test_projection_centre_is_the_drone():
    gp = project_to_ground(21.45, 80.19, 30, 0, 0.5, 0.5, 80, 1.0)
    assert abs(gp[0] - 21.45) < 1e-6 and abs(gp[1] - 80.19) < 1e-6


def test_heading_rotates_the_offset():
    """Facing east, a detection at the right of the frame lies to the south."""
    gp = project_to_ground(21.45, 80.19, 30, 90, 1.0, 0.5, 80, 1.0)
    assert gp[0] < 21.45


# ───────────────────────────────────────────────── temporal filter
def test_persistence_rejects_single_frame_flash():
    raw = np.array([0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0], bool)
    out = persistence_vote(raw, 5, 3)
    assert not out[2], "a one-frame flash must not survive"
    assert out[6:9].all(), "a sustained run must survive"


def test_persistence_handles_empty_and_short():
    assert len(persistence_vote(np.array([], bool), 5, 3)) == 0
    assert persistence_vote(np.array([1], bool), 1, 1)[0]


def test_tracker_merges_one_pool_and_drops_flicker():
    t = PoolTracker(0.15, 5)
    for i in range(10):
        t.update(i, [{"cx": 0.3 + 0.01 * i, "cy": 0.5, "area_px": 400, "conf": 0.9}])
    t.update(3, [{"cx": 0.9, "cy": 0.1, "area_px": 300, "conf": 0.8}])
    assert t.n_raw == 2
    assert len(t.confirmed(3, 0.1, 30)) == 1


def test_tracker_uses_per_sighting_scale():
    """Area must use the height recorded while the pool was in view."""
    t = PoolTracker(0.15, 5)
    g = ground_sample_distance(15, 80, 512)
    for i in range(5):
        t.update(i, [{"cx": .5, "cy": .5, "area_px": 1000, "conf": .9, "gsd_m": g}])
    pool = t.confirmed(3, ground_sample_distance(30, 80, 512), 30)[0]
    assert abs(pool["area_m2"] - pixels_to_m2(1000, g)) < 0.5


# ───────────────────────────────────────────────── priority index
def test_product_beats_weighted_sum_on_the_case_that_matters():
    """A big pool on unsuitable ground must not outrank a real breeding site."""
    dry = priority_index(0.05, 0.95, 20, 60.0)["priority_index"]
    good = priority_index(0.85, 0.80, 10, 8.0)["priority_index"]
    assert good > dry
    w_dry = (0.05 + confirmation(0.95, 20) + capacity(60.0)) / 3
    w_good = (0.85 + confirmation(0.80, 10) + capacity(8.0)) / 3
    assert w_dry > w_good, "the weighted sum should get this backwards"


def test_ranking_is_invariant_to_the_reference_area():
    sites = [(0.9, 0.9, 15, 12.0), (0.6, 0.95, 20, 45.0), (0.8, 0.7, 8, 30.0)]
    orders = []
    for ref in (30.0, 50.0, 100.0):
        sc = [priority_index(*s, reference_m2=ref)["priority_index"] for s in sites]
        orders.append(tuple(np.argsort(-np.array(sc))))
    assert len(set(orders)) == 1


def test_any_zero_factor_zeroes_the_index():
    assert priority_index(0.0, 0.9, 20, 60.0)["priority_index"] == 0.0
    assert priority_index(0.9, 0.9, 20, 0.0)["priority_index"] == 0.0


def test_bands_are_configurable():
    assert band_of(0.5) == "critical"
    assert band_of(0.5, {"critical": 0.9, "high": 0.6, "moderate": 0.3}) == "moderate"


def test_scores_are_reported_out_of_100():
    r = priority_index(0.8, 0.9, 20, 60.0)
    assert r["risk_score"] == 80
    assert 0 <= r["priority_score"] <= 100


def test_merge_combines_fragments_of_one_body():
    t = [{"lat": 21.45, "lon": 80.19, "area_m2": 60, "frames_seen": 20, "confidence": .9},
         {"lat": 21.45005, "lon": 80.19003, "area_m2": 25, "frames_seen": 12, "confidence": .8},
         {"lat": 21.4530, "lon": 80.1930, "area_m2": 40, "frames_seen": 15, "confidence": .85}]
    m = merge_nearby(t, 25)
    assert len(m) == 2
    assert max(x["area_m2"] for x in m) == 85


# ───────────────────────────────────────────────── occurrence sampling
def test_thinning_collapses_nearby_records():
    pts = np.array([[21.10, 80.10], [21.1005, 80.1005], [21.1008, 80.1002], [25.0, 82.0]])
    assert len(thin_points(pts, 1.0)) == 2
    assert len(thin_points(np.zeros((0, 2)), 1.0)) == 0


def test_background_keeps_clear_of_presence():
    pres = np.array([[21.0, 80.0], [22.0, 81.0]])
    bg = background_points(pres, (20.0, 24.0, 79.0, 83.0), 200, 1.0, 42)
    assert len(bg) > 0
    for lat, lon in bg[:50]:
        d = min(math.hypot((lat - a) * 111.32, (lon - b) * 103) for a, b in pres)
        assert d > 0.4


# ───────────────────────────────────────────────── feature schema
class _Bundle(dict):
    pass


def _bundle(feats):
    return {"features": feats, "model": None, "medians": [0] * len(feats)}


def test_schema_rejects_missing_features():
    b = _bundle(["temp_c", "rain_30d_mm", "slope_deg"])
    try:
        validate({"temp_c": [27.0], "slope_deg": [2.0]}, b)
    except SchemaError as e:
        assert "rain_30d_mm" in str(e)
    else:
        raise AssertionError("missing feature was not caught")


def test_schema_enforces_model_column_order():
    """Columns given in any order must come back in the model's order."""
    b = _bundle(["temp_c", "slope_deg"])
    X = validate({"slope_deg": [2.0, 3.0], "temp_c": [27.0, 28.0]}, b)
    assert X.shape == (2, 2)
    assert X[0, 0] == 27.0 and X[0, 1] == 2.0


def test_schema_rejects_nan():
    b = _bundle(["temp_c"])
    try:
        validate({"temp_c": [27.0, float("nan")]}, b)
    except SchemaError as e:
        assert "non-finite" in str(e)
    else:
        raise AssertionError("NaN was not caught")


def test_schema_catches_kelvin():
    b = _bundle(["temp_c"])
    try:
        validate({"temp_c": [300.0]}, b, strict_range=True)
    except SchemaError as e:
        assert "temp_c" in str(e)
    else:
        raise AssertionError("out-of-range temperature was not caught")


def test_risk_score_conversion():
    assert risk_score_100(0.837) == 84
    assert risk_score_100(0.0) == 0 and risk_score_100(1.0) == 100


def test_survey_list_is_spread_out():
    """Eight adjacent cells of one wetland must not fill the list."""
    cells = [{"lat": 21.45 + i * 0.0009, "lon": 80.19, "suitability": 0.96 - i * 0.001}
             for i in range(8)]
    cells += [{"lat": 21.30, "lon": 80.05, "suitability": 0.80},
              {"lat": 21.10, "lon": 80.40, "suitability": 0.75}]
    picks = spread_out(cells, 2.0, 5)
    assert len(picks) == 3, "clustered cells should collapse to one entry"
    assert picks[0]["suitability"] == 0.96


# ───────────────────────────────────────────────── routing
def test_route_visits_every_target_once():
    t = [{"lat": 21.45 + i * 0.001, "lon": 80.19} for i in range(5)]
    order = route_order(t)
    assert sorted(order) == list(range(5))
    assert route_length_km(t, order) > 0


if __name__ == "__main__":
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith("test_") and callable(f)]
    passed, failed = 0, []
    for name, fn in fns:
        try:
            fn()
            passed += 1
            print(f"  PASS  {name}")
        except Exception as e:
            failed.append((name, e))
            print(f"  FAIL  {name}: {e}")
    print(f"\n{passed} passed, {len(failed)} failed, {len(fns)} total")
    sys.exit(1 if failed else 0)
