# Datasets

Every dataset is real, open, and downloaded from its official source. Nothing here is generated.

| # | Dataset | Used for | Stage | Folder |
|---|---|---|---|---|
| 1 | Sen-2 LULC | Satellite U-Net (land cover) | 1–3 | `satellite/sen2lulc/` |
| 2 | Sentinel-2 SR, CHIRPS, ERA5-Land, SRTM, JRC Surface Water | Environmental features per 100 m cell | 4 | fetched via Earth Engine |
| 3 | Anopheles occurrence records | **Real target** for the risk model | 4–5 | `vectors/` |
| 4 | FAO GAUL boundaries | Gondiya district outline | 4 | fetched via Earth Engine |
| 5 | FloodNet (supervised v1.0) | Drone water U-Net | 6 | `drone/floodnet/` |

---

## 1. Sen-2 LULC — have it already

- **Source:** Mendeley Data, dataset `f4ky6ks248` — https://data.mendeley.com/datasets/f4ky6ks248
- **What it is:** Sentinel-2 RGB patches over India with pixel masks for seven land-cover classes.
- **Why:** trains the satellite U-Net. Its output becomes land-cover fractions per cell.
- **Setup:** put `SEN-2 LULC.zip` in `data/satellite/sen2lulc/`. Stage 1 extracts it.
- **Limit:** RGB only (no near-infrared), and in our earlier run 79% of water labels were
  single-pixel speckle. Stage 1 re-measures this on your copy.

## 2. Environmental layers — no download, Earth Engine

| Layer | Earth Engine ID | Gives |
|---|---|---|
| Sentinel-2 surface reflectance | `COPERNICUS/S2_SR_HARMONIZED` | NDVI, NDWI, MNDWI; RGB chips for the U-Net |
| CHIRPS daily rainfall | `UCSB-CHG/CHIRPS/DAILY` | rainfall totals |
| ERA5-Land monthly | `ECMWF/ERA5_LAND/MONTHLY_AGGR` | temperature, dew point → humidity |
| SRTM 30 m | `USGS/SRTMGL1_003` | elevation, slope |
| JRC Global Surface Water | `JRC/GSW1_4/GlobalSurfaceWater` | distance to permanent water |

## 3. Anopheles occurrence — the real target for the risk model

No dataset labels 100 m cells as "breeding site / not breeding site". What *does* exist are
**field-survey records of where malaria vectors were actually found**. That is real ground truth
for *where Anopheles occur*, and it is exactly what the Malaria Atlas Project used to build its own
vector distribution maps (Sinka et al., presence + pseudo-absence with tree-based models).

- **Primary source:** Malaria Atlas Project vector occurrence database
  (41 dominant vector species, including India's *An. culicifacies*, *An. fluviatilis*,
  *An. stephensi*). Access: R package `malariaAtlas`, function `getVecOcc(country = "India")`,
  or the MAP data portal https://data.malariaatlas.org/maps.
- **Secondary source:** GBIF occurrence API, genus *Anopheles*, country India —
  https://www.gbif.org
- **Why:** gives the risk model a target that came from the field, not from a formula.
- **What it can prove:** where environmental conditions resemble places vectors were recorded.
- **What it cannot prove:** that a specific cell contains larvae today, or malaria cases.
  That is why the drone stage exists.

Stage 4 will download these and report exactly how many records it found.

## 4. FloodNet — real drone imagery with water labels

- **Source:** https://github.com/BinaLab/FloodNet-Supervised_v1.0 (links to the download)
- **What it is:** 2,343 images from a DJI Mavic Pro after Hurricane Harvey, split ~60/20/20,
  pixel labels for 10 classes including **Water**, **Pool**, **Road-flooded**, **Building-flooded**.
- **Licence:** Community Data License Agreement (permissive). Cite the paper.
- **Why:** the drone U-Net learns standing water from real aerial photos, replacing synthetic frames.
- **Limit:** Texas suburbs, not rural Maharashtra. Fine-tune later on your own footage and report
  both numbers separately.
- **Setup:** download in Stage 6. About 5 GB.
