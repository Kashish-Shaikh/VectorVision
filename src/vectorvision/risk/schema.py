"""Feature schema — the contract between training and prediction.

A Random Forest does not complain when it is handed the wrong columns. Give it
temperature where it expects rainfall and it returns a confident number that means
nothing. The only defence is to check the columns by name before every prediction,
which is what this module does.

The schema is not written by hand. It is read from the trained model itself, so it
can never drift out of step with what the model was actually fitted on.
"""
from __future__ import annotations

import math

import numpy as np

# Plausible ranges, used to catch unit mistakes rather than to judge the data.
# A temperature of 300 means someone passed Kelvin; a slope of 95 degrees is a bug.
EXPECTED_RANGES = {
    "NDVI": (-1.0, 1.0), "NDWI": (-1.0, 1.0), "MNDWI": (-1.0, 1.0),
    "B2": (0.0, 1.6), "B3": (0.0, 1.6), "B4": (0.0, 1.6), "B8": (0.0, 1.6), "B11": (0.0, 1.6),
    "rain_season_mm": (0.0, 6000.0), "rain_30d_mm": (0.0, 2000.0),
    "temp_c": (-30.0, 60.0), "dew_c": (-40.0, 45.0), "humidity_pct": (0.0, 100.0),
    "elev_m": (-100.0, 9000.0), "slope_deg": (0.0, 90.0), "hollow_m": (-200.0, 200.0),
    "dist_water_m": (0.0, 500000.0), "water_occurrence": (0.0, 100.0),
}
for _k in range(8):
    EXPECTED_RANGES[f"lulc_c{_k}"] = (0.0, 1.0)
EXPECTED_RANGES["lulc_water"] = (0.0, 1.0)


class SchemaError(ValueError):
    """Raised when features do not match what the model was trained on."""


def schema_of(bundle: dict) -> list[str]:
    """The exact feature names, in order, that this model expects."""
    feats = bundle.get("features")
    if not feats:
        raise SchemaError("The saved model has no feature list. Retrain with Stage 5 "
                          "so the schema is stored alongside the weights.")
    return list(feats)


def validate(columns: dict, bundle: dict, *, strict_range: bool = False) -> np.ndarray:
    """Assemble a feature matrix from a dict of named columns, or refuse.

    Checks, in the order a mistake is most likely:
      missing columns, extra columns, length mismatch, non-numeric, NaN, range.

    Returns the matrix in the model's own column order, so position errors are
    impossible by construction.
    """
    want = schema_of(bundle)
    have = set(columns)

    missing = [w for w in want if w not in have]
    if missing:
        raise SchemaError(
            f"{len(missing)} feature(s) the model needs are missing: {missing}\n"
            f"The model was trained on {len(want)} features. If you added land-cover "
            f"features with 'add-landcover', retrain the risk model and rebuild the "
            f"risk map so all three agree.")

    extra = sorted(have - set(want))
    if extra:
        print(f"  note: {len(extra)} column(s) not used by this model, ignored: {extra}")

    lengths = {len(np.asarray(columns[w]).ravel()) for w in want}
    if len(lengths) != 1:
        raise SchemaError(f"Feature columns have different lengths: {sorted(lengths)}")

    cols = []
    for w in want:
        v = np.asarray(columns[w], dtype=object).ravel()
        try:
            v = v.astype(np.float64)
        except (TypeError, ValueError) as e:
            raise SchemaError(f"Feature '{w}' is not numeric: {e}") from None
        cols.append(v)
    X = np.column_stack(cols)

    bad = ~np.isfinite(X)
    if bad.any():
        per = {want[j]: int(bad[:, j].sum()) for j in range(X.shape[1]) if bad[:, j].any()}
        raise SchemaError(
            f"{int(bad.sum())} non-finite value(s) (NaN or infinity) in: {per}\n"
            "Drop those rows before predicting; a tree model cannot interpret them.")

    warnings = []
    for j, w in enumerate(want):
        rng = EXPECTED_RANGES.get(w)
        if not rng:
            continue
        lo, hi = rng
        out = int(((X[:, j] < lo) | (X[:, j] > hi)).sum())
        if out:
            warnings.append(f"{w}: {out} value(s) outside {lo} to {hi} "
                            f"(observed {X[:, j].min():.3g} to {X[:, j].max():.3g})")
    if warnings:
        msg = "Values outside their expected range:\n  " + "\n  ".join(warnings)
        if strict_range:
            raise SchemaError(msg + "\nThis usually means wrong units, such as Kelvin "
                                    "instead of Celsius.")
        print("  WARNING: " + msg)

    return X


def describe(bundle: dict) -> str:
    want = schema_of(bundle)
    groups = {"satellite": [], "climate": [], "terrain": [], "land cover": [], "other": []}
    for w in want:
        if w.startswith("lulc_"):
            groups["land cover"].append(w)
        elif w in ("NDVI", "NDWI", "MNDWI", "B2", "B3", "B4", "B8", "B11", "water_occurrence"):
            groups["satellite"].append(w)
        elif w in ("rain_season_mm", "rain_30d_mm", "temp_c", "dew_c", "humidity_pct"):
            groups["climate"].append(w)
        elif w in ("elev_m", "slope_deg", "hollow_m", "dist_water_m"):
            groups["terrain"].append(w)
        else:
            groups["other"].append(w)
    lines = [f"model expects {len(want)} features:"]
    for g, items in groups.items():
        if items:
            lines.append(f"  {g:<11} {len(items):>2}  {', '.join(items)}")
    return "\n".join(lines)


def risk_score_100(probability: float) -> int:
    """One representation used everywhere: 0-1 internally, 0-100 when shown.

    The dashboard, the reports and the spoken pitch all say "82 out of 100", while
    the model keeps its probability. Converting in one place stops the two drifting.
    """
    return int(round(float(np.clip(probability, 0.0, 1.0)) * 100))


def spread_out(items: list[dict], min_separation_km: float, limit: int) -> list[dict]:
    """Pick the best items that are not all in the same field.

    The highest-scoring cells cluster: eight adjacent 100 m cells of one wetland are
    one place to visit, not eight. Showing them all wastes the list, so each pick
    must be at least `min_separation_km` from every pick before it.
    """
    out: list[dict] = []
    for it in sorted(items, key=lambda c: -c.get("suitability", 0)):
        if len(out) >= limit:
            break
        far = True
        for k in out:
            dlat = (k["lat"] - it["lat"]) * 111.32
            dlon = (k["lon"] - it["lon"]) * 111.32 * math.cos(math.radians(k["lat"]))
            if math.hypot(dlat, dlon) < min_separation_km:
                far = False
                break
        if far:
            out.append(it)
    return out
