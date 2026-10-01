"""Stage 10 — the dashboard.

Reads the files the earlier stages actually wrote. Nothing is mocked: if a stage has
not run, the interface says so rather than showing a placeholder number, because a
dashboard that invents data is worse than no dashboard.

    outputs/fusion/risk_map.json        Stage 8a   district risk cells
    outputs/fusion/targets.json         Stage 8b   ranked GPS targets
    outputs/mission/mission_report.json Stage 9    drop simulation + safety
    outputs/metrics/stage*.json         Stages 3,5,6,7   measured model results
"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

HERE = Path(__file__).parent


def _read(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def build_app(outputs: Path, models: Path) -> FastAPI:
    app = FastAPI(title="Vector Vision")
    METRICS = outputs / "metrics"
    FUSION = outputs / "fusion"
    MISSION = outputs / "mission"

    def status() -> dict:
        """Which stages have run, and the one number that matters from each."""
        s3 = _read(METRICS / "stage3_test_report.json")
        s5 = _read(METRICS / "stage5_risk_report.json")
        s6 = _read(METRICS / "stage6_drone_test.json")
        s7 = _read(METRICS / "stage7_filter_eval.json")
        s7v = _read(METRICS / "stage7_video.json")
        rm = _read(FUSION / "risk_map.json")
        tg = _read(FUSION / "targets.json")
        ms = _read(MISSION / "mission_report.json")

        stages = []

        def add(n, name, done, headline="", detail=""):
            stages.append({"stage": n, "name": name, "done": bool(done),
                           "headline": headline, "detail": detail})

        add(3, "Satellite land cover", s3,
            f"mean IoU {s3['cleaned_labels']['mean_iou']:.3f}" if s3 else "",
            f"water IoU {s3['cleaned_labels']['per_class'][s3['water_index']]['iou']:.3f} "
            f"on {s3['test_tiles']:,} unseen tiles" if s3 else "run: python main.py eval-lulc")
        add(5, "Habitat risk model", s5,
            f"ROC-AUC {s5['spatial_cv']['roc_auc']:.3f}" if s5 else "",
            (f"spatial CV, 95% CI {s5['spatial_cv']['roc_auc_ci95'][0]:.3f}"
             f"-{s5['spatial_cv']['roc_auc_ci95'][1]:.3f}; coordinates alone "
             f"{s5['coords_only_control']['roc_auc']:.3f}") if s5 else "run: python main.py train-risk")
        add(6, "Drone water detector", s6,
            f"water IoU {s6['best_by_frame_f1']['water_iou']:.3f}" if s6 else "",
            (f"frame F1 {s6['best_by_frame_f1']['frame_f1']:.3f} vs always-water baseline "
             f"{s6['best_by_frame_f1'].get('baseline_always_water_f1', float('nan')):.3f}")
            if s6 else "run: python main.py train-drone")
        add(7, "Video + persistence filter", s7 or s7v,
            (f"specificity {s7['per_frame_raw']['specificity']:.2f} -> "
             f"{s7['after_persistence_filter']['specificity']:.2f}") if s7 else
            (f"{s7v['confirmed_pools']} pools confirmed" if s7v else ""),
            "filter measured on held-out images" if s7 else "run: python main.py infer-video")
        add(8, "Priority index", rm,
            f"{len(rm['cells']):,} cells scored" if rm else "",
            (f"{tg['targets_found']} confirmed targets" if tg else
             "no flight fused yet") if rm else "run: python main.py fuse")
        add(9, "Mission + drop", ms,
            f"{ms['drop_simulation']['within_1m_pct']:.0f}% within 1 m" if ms else "",
            (f"median miss {ms['drop_simulation']['cep50_m']} m"
             + ("" if ms["drop_simulation"]["meets_1m_requirement"]
                else "; the 1 m requirement is NOT met")) if ms else "run: python main.py mission")

        limits = [
            "Risk labels are presence and background, not presence and absence: a 0 means "
            "a random place in the country, not a confirmed absence.",
            "Vector presence is not an active breeding site. The drone confirms water; only "
            "a field visit confirms larvae.",
            "The priority index ranks sites for visiting. It is not a probability that "
            "larvae are present.",
            "Severity bands are quantiles within this district, not an absolute scale.",
        ]
        if s5:
            gap = s5["spatial_cv"]["roc_auc"] - s5["coords_only_control"]["roc_auc"]
            limits.insert(1, f"Coordinates alone reach ROC-AUC "
                             f"{s5['coords_only_control']['roc_auc']:.3f} versus "
                             f"{s5['spatial_cv']['roc_auc']:.3f} with all features, so part of "
                             f"the model's skill reflects where surveys happened (+{gap:.3f}).")
        if s6:
            limits.append("The drone detector was trained on FloodNet (suburban Texas after a "
                          "hurricane). Transfer to local footage must be measured separately.")
        return {"stages": stages, "limits": limits,
                "district": (rm or {}).get("meta", {}).get("district"),
                "state": (rm or {}).get("meta", {}).get("state")}

    @app.get("/api/status")
    def api_status():
        return JSONResponse(status())

    @app.get("/api/risk-map")
    def api_risk_map():
        d = _read(FUSION / "risk_map.json")
        if not d:
            raise HTTPException(404, "No risk map yet. Run: python main.py fuse")
        return JSONResponse(d)

    @app.get("/api/targets")
    def api_targets():
        d = _read(FUSION / "targets.json")
        if not d:
            raise HTTPException(404, "No targets yet. Run Stage 7 on a flight, then Stage 8.")
        return JSONResponse(d)

    @app.get("/api/mission")
    def api_mission():
        d = _read(MISSION / "mission_report.json")
        if not d:
            raise HTTPException(404, "No mission yet. Run: python main.py mission")
        return JSONResponse(d)

    @app.get("/api/metrics")
    def api_metrics():
        out = {}
        for key, name in (("stage3", "stage3_test_report.json"),
                          ("stage5", "stage5_risk_report.json"),
                          ("stage6", "stage6_drone_test.json"),
                          ("stage7", "stage7_filter_eval.json"),
                          ("stage7_video", "stage7_video.json")):
            d = _read(METRICS / name)
            if d:
                out[key] = d
        return JSONResponse(out)

    @app.get("/api/mission-plan")
    def api_plan():
        p = MISSION / "mission.plan"
        if not p.exists():
            raise HTTPException(404, "No mission.plan yet.")
        return FileResponse(p, filename="vector_vision.plan", media_type="application/json")

    static = HERE / "static"
    if static.exists():
        app.mount("/", StaticFiles(directory=static, html=True), name="ui")
    return app


def serve(outputs: Path, models: Path, host: str = "127.0.0.1", port: int = 8000) -> None:
    import uvicorn
    uvicorn.run(build_app(outputs, models), host=host, port=port)
