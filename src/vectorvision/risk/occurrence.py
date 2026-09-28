"""Stage 4a — the TARGET for the risk model: real Anopheles occurrence records.

No dataset labels a 100 m cell "breeding site / not breeding site". What does exist
are field-survey records of where malaria vectors were actually caught. That is real
ground truth for vector presence, and it is the same evidence the Malaria Atlas
Project used to build its vector maps (presence records plus background points,
fitted with tree models).

Three honest limitations, all reported in the metadata file:
  1. Presence-only data. Nobody publishes "we looked here and found nothing", so the
     zeros are BACKGROUND points, not confirmed absences. The model therefore learns
     "conditions like where vectors were found" vs "conditions across the region".
  2. Survey bias. Records cluster near roads, towns and research institutes. Thinning
     to one record per grid cell reduces this but cannot remove it.
  3. Time mismatch. Records span many years; the satellite features come from one
     season. So the features describe the place, not the day of the catch.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

GBIF = "https://api.gbif.org/v1"


def _get(url: str, params: dict, tries: int = 4) -> dict:
    import urllib.error
    import urllib.parse
    import urllib.request
    q = urllib.parse.urlencode(params, doseq=True)
    for a in range(tries):
        try:
            with urllib.request.urlopen(f"{url}?{q}", timeout=60) as r:
                return json.loads(r.read().decode())
        except urllib.error.URLError as e:
            if a == tries - 1:
                raise SystemExit(f"GBIF request failed: {e}\nOn Kaggle, switch Internet ON in Settings.")
            time.sleep(2 * (a + 1))
    return {}


def fetch_occurrences(cfg: dict, out_dir: Path, force: bool = False) -> tuple[np.ndarray, dict]:
    """Download every georeferenced Anopheles record for the country, with caching."""
    o = cfg["occurrence"]
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_p = out_dir / "anopheles_gbif_raw.json"

    if raw_p.exists() and not force:
        recs = json.loads(raw_p.read_text())
        print(f"using cached GBIF download: {len(recs):,} records")
    else:
        m = _get(f"{GBIF}/species/match", {"name": o["genus"], "rank": "GENUS"})
        key = m.get("usageKey")
        if not key:
            raise SystemExit(f"GBIF could not match the genus {o['genus']}")
        print(f"GBIF taxon key for {o['genus']}: {key} ({m.get('scientificName')})")

        recs, offset = [], 0
        while len(recs) < o["max_records"]:
            page = _get(f"{GBIF}/occurrence/search", {
                "taxonKey": key, "country": o["country"], "hasCoordinate": "true",
                "hasGeospatialIssue": "false", "limit": 300, "offset": offset})
            got = page.get("results", [])
            if not got:
                break
            for r in got:
                recs.append({k: r.get(k) for k in
                             ("decimalLatitude", "decimalLongitude", "year", "species",
                              "coordinateUncertaintyInMeters", "basisOfRecord", "datasetKey")})
            offset += len(got)
            print(f"  downloaded {len(recs):,} of {page.get('count', '?'):,}", end="\r")
            if page.get("endOfRecords"):
                break
        print(f"\n  GBIF returned {len(recs):,} records")
        raw_p.write_text(json.dumps(recs))

    # ---- quality filtering
    kept, drop = [], {"no_coords": 0, "too_old": 0, "coarse_coords": 0}
    for r in recs:
        lat, lon = r.get("decimalLatitude"), r.get("decimalLongitude")
        if lat is None or lon is None:
            drop["no_coords"] += 1; continue
        if r.get("year") and r["year"] < o["min_year"]:
            drop["too_old"] += 1; continue
        unc = r.get("coordinateUncertaintyInMeters")
        if unc is not None and unc > 5000:      # coarser than our 100 m cells can use
            drop["coarse_coords"] += 1; continue
        kept.append((lat, lon))
    pts = np.array(kept, float) if kept else np.zeros((0, 2))
    print(f"  after filtering: {len(pts):,} records  (dropped {drop})")

    species = {}
    for r in recs:
        if r.get("species"):
            species[r["species"]] = species.get(r["species"], 0) + 1
    top = sorted(species.items(), key=lambda kv: -kv[1])[:6]
    if top:
        print("  most recorded species: " + ", ".join(f"{s} ({n})" for s, n in top))

    thinned = thin_points(pts, o["thin_km"])
    print(f"  after thinning to one per {o['thin_km']} km: {len(thinned):,} presence points")
    meta = {"downloaded": len(recs), "after_filtering": len(pts), "presence_points": len(thinned),
            "dropped": drop, "top_species": top, "source": "GBIF occurrence API",
            "genus": o["genus"], "country": o["country"], "min_year": o["min_year"]}
    return thinned, meta


def thin_points(pts: np.ndarray, km: float) -> np.ndarray:
    """Keep one point per km-sized cell. Surveys repeat at the same few sites, and
    without thinning the model would mostly learn where entomologists like to work."""
    if len(pts) == 0:
        return pts
    dlat = km / 111.32
    seen, out = set(), []
    for lat, lon in pts:
        dlon = km / (111.32 * max(math.cos(math.radians(lat)), 0.2))
        cell = (int(lat / dlat), int(lon / dlon))
        if cell not in seen:
            seen.add(cell)
            out.append((lat, lon))
    return np.array(out, float)


def background_points(presence: np.ndarray, bbox: tuple[float, float, float, float],
                      n: int, min_km: float, seed: int, inside=None) -> np.ndarray:
    """Random background points across the same region.

    These are NOT confirmed absences. They describe the range of conditions that
    exist in the study region, which is what a presence-background model compares
    the presence points against.
    """
    rng = np.random.default_rng(seed)
    lat0, lat1, lon0, lon1 = bbox
    dlat = min_km / 111.32
    occupied = set()
    for lat, lon in presence:
        dlon = min_km / (111.32 * max(math.cos(math.radians(lat)), 0.2))
        occupied.add((int(lat / dlat), int(lon / dlon)))
    out, guard = [], 0
    while len(out) < n and guard < n * 200:
        guard += 1
        lat = rng.uniform(lat0, lat1)
        lon = rng.uniform(lon0, lon1)
        dlon = min_km / (111.32 * max(math.cos(math.radians(lat)), 0.2))
        if (int(lat / dlat), int(lon / dlon)) in occupied:
            continue
        if inside is not None and not inside(lat, lon):
            continue
        occupied.add((int(lat / dlat), int(lon / dlon)))
        out.append((lat, lon))
    return np.array(out, float)
