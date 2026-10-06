"""Stage 7 — frame-by-frame detection on drone video.

A single-image detector is not yet a flight tool. Two things have to be added:

  1. Temporal filtering. Real water sits still and appears in many frames in a row.
     Glare, shadow, motion blur and video noise flicker in and out. Requiring a
     detection to survive several frames removes most false alarms at a small cost
     in recall.
  2. Grouping. The same pool seen in forty frames is ONE breeding site, not forty.
     Detections are tracked by position and merged into pools, each with an area in
     square metres, a confidence and a severity score.

Everything here is measurable before flying: `simulate_flight` builds panning
sequences from held-out FloodNet images, whose masks give real ground truth, so the
value of the temporal filter can be reported with numbers rather than asserted.

The detector is passed in as `predict_fn`, which keeps this file independent of
PyTorch and lets the whole pipeline be tested with a stub.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------- geometry
def ground_sample_distance(altitude_m: float, hfov_deg: float, width_px: int) -> float:
    """Metres covered by one pixel, straight down."""
    width_m = 2.0 * altitude_m * math.tan(math.radians(hfov_deg / 2.0))
    return width_m / max(width_px, 1)


def pixels_to_m2(n_px: int, gsd_m: float) -> float:
    return float(n_px) * gsd_m * gsd_m


# --------------------------------------------------------------------- temporal
def persistence_vote(raw: np.ndarray, window: int, votes: int) -> np.ndarray:
    """Sliding-window majority filter over per-frame yes/no detections.

    A frame counts as water only if at least `votes` of the `window` frames centred
    on it also detected water. A one-frame flash of glare cannot survive this;
    a real pool the drone flies over does.
    """
    raw = np.asarray(raw).astype(int)
    n = len(raw)
    if n == 0 or window <= 1:
        return raw.astype(bool)
    half = window // 2
    out = np.zeros(n, bool)
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        out[i] = raw[lo:hi].sum() >= votes
    return out


class PoolTracker:
    """Groups detections across frames into distinct pools by position."""

    def __init__(self, match_radius: float = 0.15, max_gap: int = 5):
        self.tracks: list[dict] = []
        self.r2 = match_radius ** 2
        self.max_gap = max_gap

    def update(self, frame_idx: int, detections: list[dict], fix: dict | None = None) -> None:
        for d in detections:
            hit = None
            for t in self.tracks:
                if frame_idx - t["last_frame"] > self.max_gap:
                    continue
                if (t["cx"] - d["cx"]) ** 2 + (t["cy"] - d["cy"]) ** 2 <= self.r2:
                    hit = t
                    break
            if hit is None:
                self.tracks.append({"cx": d["cx"], "cy": d["cy"], "first_frame": frame_idx,
                                    "last_frame": frame_idx, "n_frames": 1,
                                    "areas_px": [d["area_px"]], "confs": [d["conf"]],
                                    "gsds": [d.get("gsd_m")], "fix": fix, "best_conf": d["conf"],
                                    "best_cx": d["cx"], "best_cy": d["cy"]})
            else:
                hit["last_frame"] = frame_idx
                hit["n_frames"] += 1
                hit["cx"] = 0.7 * hit["cx"] + 0.3 * d["cx"]
                hit["cy"] = 0.7 * hit["cy"] + 0.3 * d["cy"]
                hit["areas_px"].append(d["area_px"])
                hit["confs"].append(d["conf"])
                hit.setdefault("gsds", []).append(d.get("gsd_m"))
                # Keep the fix from the clearest sighting: that frame saw the pool
                # best, so its position and heading give the most reliable projection.
                if fix and d["conf"] >= hit.get("best_conf", 0):
                    hit["fix"], hit["best_conf"] = fix, d["conf"]
                    hit["best_cx"], hit["best_cy"] = d["cx"], d["cy"]

    def confirmed(self, min_frames: int, gsd_m: float, fps: float = 30.0) -> list[dict]:
        out = []
        for t in self.tracks:
            if t["n_frames"] < min_frames:
                continue
            area_px = float(np.median(t["areas_px"]))
            # Use the ground scale recorded while this pool was in view. Height
            # changes during a flight, so one fixed scale would mis-size every pool
            # seen at a different altitude.
            seen_gsd = [g for g in t.get("gsds", []) if g]
            scale = float(np.median(seen_gsd)) if seen_gsd else gsd_m
            rec = {"frames_seen": t["n_frames"],
                        "first_frame": t["first_frame"], "last_frame": t["last_frame"],
                        "t_start_s": round(t["first_frame"] / max(fps, 1e-6), 2),
                        "t_end_s": round(t["last_frame"] / max(fps, 1e-6), 2),
                        "area_px": area_px,
                   "area_m2": round(pixels_to_m2(area_px, scale), 2),
                   "gsd_m_used": round(scale, 4),
                        "confidence": round(float(np.mean(t["confs"])), 3),
                   "frame_x": round(t["cx"], 3), "frame_y": round(t["cy"], 3)}
            if t.get("fix"):
                rec["gps"] = t["fix"]
                rec["frame_x_at_fix"] = round(t.get("best_cx", t["cx"]), 3)
                rec["frame_y_at_fix"] = round(t.get("best_cy", t["cy"]), 3)
            out.append(rec)
        return sorted(out, key=lambda d: -d["area_m2"])

    @property
    def n_raw(self) -> int:
        return len(self.tracks)


# --------------------------------------------------------------------- severity
def severity(area_m2: float, confidence: float, frames_seen: int,
             cell_risk: float = 0.5, slope_deg: float = 2.0) -> dict:
    """Combine what the drone saw with what the satellite model knows.

    Area matters because a bigger pool supports more larvae. Persistence matters
    because a pool must survive 9-12 days for a breeding cycle to finish, and a pool
    seen in many frames is more likely to be a real standing body. Slope matters
    because water drains off a hillside. Cell risk carries the climate and terrain
    context from Stage 5.
    """
    a = float(np.clip(math.log1p(max(area_m2, 0)) / math.log1p(60), 0, 1))
    c = float(np.clip(confidence, 0, 1))
    p = float(np.clip(frames_seen / 12.0, 0, 1))
    f = float(math.exp(-max(slope_deg, 0) / 6.0))
    s = 0.32 * a + 0.18 * c + 0.15 * p + 0.25 * float(np.clip(cell_risk, 0, 1)) + 0.10 * f
    band = "critical" if s > .72 else "high" if s > .55 else "moderate" if s > .35 else "low"
    return {"severity_score": round(s, 3), "severity": band,
            "drivers": {"pool_size": round(a, 2), "detector_confidence": round(c, 2),
                        "persistence": round(p, 2), "area_risk": round(cell_risk, 2),
                        "ground_flatness": round(f, 2)}}


# --------------------------------------------------------------------- pipeline
def detect_sequence(frames, predict_fn, cfg_v: dict, gsd_m: float, fps: float = 30.0,
                    fixes: dict | None = None) -> dict:
    """Run the detector over a sequence of frames and group what it finds.

    `frames`    iterable of (frame_index, RGB uint8 array)
    `predict_fn` takes a batch of frames, returns water probability maps
    `fixes`     optional {frame_index: telemetry dict}. When present, each pool's
                area is measured using the height at that instant rather than one
                assumed altitude, and it carries the fix needed for real coordinates.
    """
    import cv2

    thr = float(cfg_v["prob_threshold"])
    min_px = int(cfg_v["min_pool_px"])
    tracker = PoolTracker(float(cfg_v["match_radius"]), int(cfg_v.get("max_gap", 5)))
    raw_flags, per_frame = [], []

    fixes = fixes or {}
    for idx, img in frames:
        fix = fixes.get(idx)
        # Height changes during a flight, and ground scale goes with it: the same
        # puddle covers four times the pixels at half the altitude.
        gsd_here = gsd_m
        if fix and fix.get("alt_m") and fix["alt_m"] >= 2.0:
            gsd_here = ground_sample_distance(fix["alt_m"], float(cfg_v["hfov_deg"]),
                                              img.shape[1])
        prob = predict_fn(img[None])[0]
        h, w = prob.shape
        binm = (prob >= thr).astype(np.uint8)
        binm = cv2.morphologyEx(binm, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, lab, stats, cent = cv2.connectedComponentsWithStats(binm, 8)
        dets = []
        for k in range(1, n):
            area = int(stats[k, cv2.CC_STAT_AREA])
            if area < min_px:
                continue
            comp = lab == k
            dets.append({"cx": float(cent[k][0] / w), "cy": float(cent[k][1] / h),
                         "area_px": area, "conf": float(prob[comp].mean())})
        for d in dets:
            d["gsd_m"] = gsd_here
        tracker.update(idx, dets, fix=fix)
        raw_flags.append(len(dets) > 0)
        per_frame.append({"frame": int(idx), "t_s": round(idx / max(fps, 1e-6), 2),
                          "raw_water": bool(dets), "pools": len(dets),
                          "max_conf": round(max((d["conf"] for d in dets), default=0.0), 3),
                          "gps": bool(fix), "alt_m": (fix or {}).get("alt_m")})

    smooth = persistence_vote(np.array(raw_flags, bool),
                              int(cfg_v["persist_window"]), int(cfg_v["persist_votes"]))
    for rec, s in zip(per_frame, smooth):
        rec["water"] = bool(s)

    pools = tracker.confirmed(int(cfg_v["persist_votes"]), gsd_m, fps)
    for p in pools:
        p.update(severity(p["area_m2"], p["confidence"], p["frames_seen"]))

    n_fix = sum(1 for f in per_frame if f["gps"])
    return {"frames_analysed": len(per_frame),
            "frames_with_gps": n_fix,
            "frames_raw_water": int(np.sum(raw_flags)),
            "frames_after_filter": int(smooth.sum()),
            "raw_tracks": tracker.n_raw, "confirmed_pools": len(pools),
            "gsd_m_per_px": round(gsd_m, 4), "pools": pools, "frames": per_frame}


# --------------------------------------------------------------------- simulation
def simulate_flight(img: np.ndarray, mask: np.ndarray, n_steps: int = 12,
                    crop_frac: float = 0.6, out_size: int = 512):
    """Pan a crop window across one image, as if the drone were flying over it.

    Each step is a frame whose ground truth comes from the same crop of the real
    mask, so a detector can be scored frame by frame without any invented labels.
    """
    import cv2
    H, W = mask.shape
    ch, cw = int(H * crop_frac), int(W * crop_frac)
    ys = np.linspace(0, H - ch, n_steps).astype(int)
    xs = np.linspace(0, W - cw, n_steps).astype(int)
    for i, (y, x) in enumerate(zip(ys, xs)):
        sub_i = img[y:y + ch, x:x + cw]
        sub_m = mask[y:y + ch, x:x + cw]
        yield (i,
               cv2.resize(sub_i, (out_size, out_size), interpolation=cv2.INTER_AREA),
               cv2.resize(sub_m, (out_size, out_size), interpolation=cv2.INTER_NEAREST))


def evaluate_filter(X: np.ndarray, Y: np.ndarray, predict_fn, cfg_v: dict,
                    n_images: int = 20, n_steps: int = 12, seed: int = 42) -> dict:
    """Does temporal filtering actually help? Measured on held-out images.

    Frames come from panning over real test images, so the truth is known. Raw
    per-frame decisions are compared with the same decisions after the persistence
    vote. The filter should raise precision and specificity; some recall loss is
    expected and acceptable.
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), min(n_images, len(X)), replace=False)
    min_px = int(cfg_v["min_pool_px"])
    raw_all, smooth_all, truth_all = [], [], []

    for j in idx:
        seq = list(simulate_flight(X[j], Y[j], n_steps, out_size=X.shape[1]))
        frames = [(i, im) for i, im, _ in seq]
        truth = [int((m > 0).sum()) >= min_px for _, _, m in seq]
        res = detect_sequence(frames, predict_fn, cfg_v,
                              ground_sample_distance(cfg_v["altitude_m"], cfg_v["hfov_deg"],
                                                     X.shape[1]))
        raw_all += [f["raw_water"] for f in res["frames"]]
        smooth_all += [f["water"] for f in res["frames"]]
        truth_all += truth

    def score(pred):
        pred, t = np.array(pred, bool), np.array(truth_all, bool)
        tp = int((pred & t).sum()); fp = int((pred & ~t).sum())
        tn = int((~pred & ~t).sum()); fn = int((~pred & t).sum())
        prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
        spec = tn / max(tn + fp, 1)
        return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "precision": prec, "recall": rec,
                "specificity": spec, "f1": 2 * prec * rec / max(prec + rec, 1e-12),
                "accuracy": (tp + tn) / max(len(t), 1),
                "balanced_accuracy": (rec + spec) / 2}

    raw_s, sm_s = score(raw_all), score(smooth_all)
    return {"images": int(len(idx)), "frames_per_image": n_steps,
            "total_frames": len(truth_all),
            "prevalence": float(np.mean(truth_all)),
            "per_frame_raw": raw_s, "after_persistence_filter": sm_s,
            "change": {k: round(sm_s[k] - raw_s[k], 4)
                       for k in ("precision", "recall", "specificity", "f1", "balanced_accuracy")},
            "settings": {k: cfg_v[k] for k in ("prob_threshold", "min_pool_px",
                                               "persist_window", "persist_votes")}}
