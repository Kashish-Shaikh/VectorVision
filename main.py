"""Vector Vision — one command line for every stage.

    python main.py verify            Stage 1: environment + dataset verification
    python main.py train-lulc        Stage 2: Satellite U-Net
    python main.py eval-lulc         Stage 3: held-out test evaluation
    python main.py build-features    Stage 4: occurrence + environmental features
    python main.py train-risk        Stage 5: Random Forest risk model
    python main.py train-drone       Stage 6: Drone U-Net on FloodNet
    python main.py stages            show every stage and its status
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from vectorvision.utils.config import load_config, p, seed_everything


STAGES = [
    (1, "verify",         "Environment + dataset verification",                "ready"),
    (2, "train-lulc",     "Satellite U-Net (from scratch) on Sen-2 LULC",       "ready"),
    (3, "eval-lulc",      "Test U-Net on held-out tiles, per-class IoU",        "ready"),
    (4, "build-features", "Gondiya 100 m grid + vector-occurrence training set", "ready"),
    (5, "train-risk",     "Random Forest risk model + SHAP explanations",        "ready"),
    (6, "train-drone",    "Drone water U-Net (from scratch) on FloodNet",       "ready"),
    (7, "infer-video", "Frame-by-frame video detection + tracking", "ready"),
    (8, "fuse",           "Breeding Site Priority Index + GPS targets",          "ready"),
    (9, "mission",        "Waypoints, flight + payload-drop simulation",         "ready"),
    (10, "dashboard",     "Web dashboard",                                       "planned"),
]


def cmd_stages(_args, _cfg):
    print("\nVector Vision stages")
    for n, name, desc, status in STAGES:
        print(f"  {n:>2}. {name:<15} {desc:<52} [{status}]")


def cmd_verify(args, cfg):
    from vectorvision.utils.environment import check_environment
    from vectorvision.satellite.inspect_sen2lulc import inspect_sen2lulc

    print("=" * 72)
    print("STAGE 1 - environment and dataset verification")
    print("=" * 72)

    env = check_environment(cfg, check_ee=not args.skip_ee)

    rep = inspect_sen2lulc(
        p(cfg["paths"]["sen2lulc_root"]),
        p(cfg["paths"]["outputs_dir"]) / "metrics",
        n_samples=args.samples,
        do_extract=not args.no_extract,
        seed=cfg["project"]["seed"],
    )

    print("\nLater datasets (download in their own stages)")
    for label, key in (
        ("FloodNet drone imagery", "floodnet_root"),
        ("Anopheles occurrence points", "vectors_dir"),
    ):
        path = p(cfg["paths"][key])
        present = path.exists() and any(path.iterdir())
        print(
            f"  {'[ OK ]' if present else '[----]'} "
            f"{label:<30} "
            f"{'present' if present else 'not yet - fine for now'}"
        )

    fails = [e for e in env if e["status"] == "FAIL"]
    ds_ok = rep.get("status") in ("PASS", "WARN")

    summary = {
        "environment": env,
        "dataset_status": rep.get("status"),
        "stage1_passed": not fails and ds_ok,
    }

    out = (
        p(cfg["paths"]["outputs_dir"])
        / "metrics"
        / "stage1_summary.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, default=str))

    print("\n" + "=" * 72)

    if summary["stage1_passed"]:
        print("STAGE 1 PASSED")
        print("  report : outputs/metrics/stage1_dataset_report.json")
        print("  figure : outputs/graphs/stage1_samples.png")
    else:
        print("STAGE 1 NOT PASSED - fix the [FAIL] lines above, then run again")

    print("=" * 72)

    return 0 if summary["stage1_passed"] else 1


def cmd_train_lulc(args, cfg):
    from vectorvision.satellite.cache import build_cache
    from vectorvision.utils.config import get_device

    print("=" * 72)
    print("STAGE 2 - satellite U-Net, trained from random initialisation")
    print("=" * 72)

    dev = get_device(cfg["project"].get("device", "auto"))

    if dev == "cpu" and not args.allow_cpu:
        print(
            "No GPU found. Turn one on or pass --allow_cpu to force CPU training."
        )
        return 1

    if args.subset:
        cfg["satellite_unet"]["train_subset"] = args.subset

    outputs = p(cfg["paths"]["outputs_dir"])

    cache = build_cache(
        cfg,
        p(cfg["paths"]["sen2lulc_root"]),
        outputs,
        force=args.rebuild_cache,
    )

    if args.cache_only:
        print("cache built; stopping as asked (--cache-only)")
        return 0

    from vectorvision.satellite.train import train

    res = train(
        cfg,
        cache,
        p(cfg["paths"]["models_dir"]) / "unet_lulc",
        dev,
        epochs=args.epochs,
        resume=args.resume,
    )

    print("\n" + "=" * 72)
    print(
        f"STAGE 2 DONE - best validation mean IoU "
        f"{res['best_mean_iou']:.4f}"
    )
    print("  weights : models/unet_lulc/best.pt")
    print("  log     : models/unet_lulc/history.csv")
    print("  curves  : models/unet_lulc/training_curves.png")
    print("=" * 72)

    return 0


def cmd_eval_lulc(args, cfg):
    from vectorvision.satellite.evaluate import evaluate
    from vectorvision.utils.config import get_device

    print("=" * 72)
    print("STAGE 3 - test the satellite U-Net on the untouched test split")
    print("=" * 72)

    dev = get_device(cfg["project"].get("device", "auto"))

    rep = evaluate(
        cfg,
        p(cfg["paths"]["sen2lulc_root"]),
        p(cfg["paths"]["outputs_dir"]),
        p(cfg["paths"]["models_dir"]) / "unet_lulc",
        dev,
        args.tiles,
    )

    w = rep["water_index"]

    print(f"\nTEST RESULTS on {rep['test_tiles']:,} unseen tiles")

    for tag in ("raw_labels", "cleaned_labels"):
        r = rep[tag]

        print(
            f"\n  {tag.replace('_', ' ')}: "
            f"mean IoU {r['mean_iou']:.4f} "
            f"(95% CI {r['ci95']['mean_iou'][0]:.3f}-"
            f"{r['ci95']['mean_iou'][1]:.3f}), "
            f"pixel accuracy {r['pixel_accuracy']:.4f}"
        )

        print(
            f"    {'class':<16}"
            f"{'IoU':>7}"
            f"{'prec':>7}"
            f"{'recall':>8}"
            f"{'F1':>7}"
            f"{'share':>8}"
        )

        for c in r["per_class"]:
            print(
                f"    {c['class']:<16}"
                f"{c['iou']:7.3f}"
                f"{c['precision']:7.3f}"
                f"{c['recall']:8.3f}"
                f"{c['f1']:7.3f}"
                f"{c['share_pct']:7.2f}%"
            )

        print(
            f"    water IoU 95% CI "
            f"{r['ci95']['water_iou'][0]:.3f}-"
            f"{r['ci95']['water_iou'][1]:.3f}"
        )

    t = rep["tile_water_detection"]

    print(
        f"\n  does the tile contain a water body? "
        f"accuracy {t['accuracy']:.3f} "
        f"(95% CI {t['accuracy_ci95'][0]:.3f}-"
        f"{t['accuracy_ci95'][1]:.3f}), "
        f"precision {t['precision']:.3f}, "
        f"recall {t['recall']:.3f}, "
        f"{t['positive_tiles']} of {t['tiles']} tiles contain water"
    )

    print("\n  water threshold sweep (cleaned labels):")

    for s_ in rep["water_threshold_sweep"][::2]:
        print(
            f"    threshold {s_['threshold']:.2f}  "
            f"precision {s_['precision']:.3f}  "
            f"recall {s_['recall']:.3f}  "
            f"IoU {s_['iou']:.3f}"
        )

    print("\n" + "=" * 72)
    print("STAGE 3 DONE")
    print("  report : outputs/metrics/stage3_test_report.json")
    print("  graphs : outputs/graphs/stage3_*.png")
    print("=" * 72)

    return 0


def cmd_build_features(args, cfg):
    from vectorvision.risk.features import build_training_table

    print("=" * 72)
    print("STAGE 4 - real occurrence records + environmental features")
    print("=" * 72)

    meta = build_training_table(
        cfg,
        p(cfg["paths"]["vectors_dir"]),
        force=args.rebuild,
    )

    print("\n" + "=" * 72)
    print("STAGE 4 DONE")
    print(f"  presence rows   : {meta['presence_sampled']:,}")
    print(f"  background rows : {meta['background_sampled']:,}")
    print(
        f"  dropped points  : "
        f"{meta['dropped_masked_pixels']:,} (masked pixels)"
    )
    print("  table  : data/vectors/risk_training_table.csv")
    print("  meta   : data/vectors/risk_training_meta.json")
    print("\n  Remember: 0 means BACKGROUND, not a confirmed absence.")
    print("=" * 72)

    return 0


def cmd_train_risk(args, cfg):
    from vectorvision.risk.train_risk import train

    print("=" * 72)
    print("STAGE 5 - risk model on real occurrence records")
    print("=" * 72)

    table = p(cfg["paths"]["vectors_dir"]) / "risk_training_table.csv"

    if not table.exists():
        print("Training table not found. Run: python main.py build-features")
        return 1

    rep = train(
        cfg,
        table,
        p(cfg["paths"]["models_dir"]) / "risk_rf",
        p(cfg["paths"]["outputs_dir"]),
    )

    print("\n" + "=" * 72)
    print("STAGE 5 DONE")

    print(
        f"  headline (spatial CV) ROC-AUC "
        f"{rep['spatial_cv']['roc_auc']:.3f} "
        f"(95% CI "
        f"{rep['spatial_cv']['roc_auc_ci95'][0]:.3f}-"
        f"{rep['spatial_cv']['roc_auc_ci95'][1]:.3f})"
    )

    print("  model  : models/risk_rf/risk_rf.joblib")
    print("  report : outputs/metrics/stage5_risk_report.json")
    print("  graphs : outputs/graphs/stage5_*.png")
    print("  Report the SPATIAL CV number, not the random-fold one.")
    print("=" * 72)

    return 0

def cmd_train_drone(args, cfg):
    from vectorvision.vision.floodnet import build_cache, summarise
    from vectorvision.utils.config import get_device

    print("=" * 72)
    print("STAGE 6 - drone water detector")
    print("=" * 72)

    root = p(cfg["paths"]["floodnet_root"])

    if not root.exists():
        print(f"FloodNet not found at {root}")
        print(
            "On Kaggle: Add Input > search 'FloodNet' > "
            "aerial-imagery-dataset-floodnet-challenge"
        )
        print(
            "Then point paths.floodnet_root in configs/config.yaml "
            "at it."
        )
        return 1

    if args.inspect:
        summarise(root)
        return 0

    dev = get_device(cfg["project"].get("device", "auto"))

    if dev == "cpu" and not args.allow_cpu:
        print("No GPU found. Turn one on, or pass --allow-cpu (very slow).")
        return 1

    cache = build_cache(
        cfg,
        root,
        p(cfg["paths"]["outputs_dir"]) / "cache_floodnet",
        force=args.rebuild_cache,
    )

    if args.cache_only:
        print("cache built; stopping as asked (--cache-only)")
        return 0

    from vectorvision.vision.train_drone import test, train

    out = p(cfg["paths"]["models_dir"]) / "drone_unet"

    # ------------------------------------------------------------
    # TEST ONLY: load the existing best.pt and do NOT retrain
    # ------------------------------------------------------------
    if args.test_only:
        print("\n" + "=" * 72)
        print("STAGE 6 TEST ONLY - using existing best.pt")
        print("=" * 72)

        best_model = out / "best.pt"

        if not best_model.exists():
            print(f"ERROR: saved model not found at {best_model}")
            print("Train Stage 6 first before using --test-only.")
            return 1

        rep = test(
            cfg,
            cache,
            out,
            p(cfg["paths"]["outputs_dir"]),
            dev,
        )

        if rep:
            b = rep["best_by_frame_f1"]

            print("\n" + "=" * 72)
            print(
                f"HELD-OUT TEST ({rep['test_images']} images, "
                f"threshold {b['threshold']})"
            )

            print(
                f"  frame F1       : {b['frame_f1']:.3f}"
            )
            print(
                f"  precision      : {b['frame_precision']:.3f}"
            )
            print(
                f"  recall         : {b['frame_recall']:.3f}"
            )
            print(
                f"  accuracy       : {b['frame_accuracy']:.3f}"
            )
            print(
                f"  water IoU      : {b['water_iou']:.3f}"
            )

            # New diagnostics from train_drone.py
            if "frame_summary" in rep:
                fs = rep["frame_summary"]

                print("\nFRAME-LEVEL BASELINE COMPARISON")
                print("-" * 72)

                for key in [
                    "frames",
                    "frames_with_water",
                    "prevalence",
                    "frame_precision",
                    "frame_recall",
                    "frame_specificity",
                    "frame_f1",
                    "frame_accuracy",
                    "frame_balanced_accuracy",
                    "always_water_f1",
                    "always_water_accuracy",
                ]:
                    if key in fs:
                        value = fs[key]

                        if isinstance(value, float):
                            print(f"  {key:28s}: {value:.3f}")
                        else:
                            print(f"  {key:28s}: {value}")

            print("\nmodel  :", best_model)
            print(
                "report :",
                p(cfg["paths"]["outputs_dir"])
                / "metrics"
                / "stage6_drone_test.json",
            )
            print("=" * 72)

        return 0

    # ------------------------------------------------------------
    # NORMAL TRAINING
    # ------------------------------------------------------------
    res = train(
        cfg,
        cache,
        out,
        dev,
        epochs=args.epochs,
        resume=args.resume,
    )

    rep = test(
        cfg,
        cache,
        out,
        p(cfg["paths"]["outputs_dir"]),
        dev,
    )

    print("\n" + "=" * 72)
    print(
        f"STAGE 6 DONE - best validation frame F1 "
        f"{res['best_frame_f1']:.3f}"
    )

    if rep:
        b = rep["best_by_frame_f1"]

        print(
            f"  HELD-OUT TEST ({rep['test_images']} images, "
            f"threshold {b['threshold']}):"
        )

        print(
            f"    frame F1 {b['frame_f1']:.3f} "
            f"(precision {b['frame_precision']:.3f}, "
            f"recall {b['frame_recall']:.3f}, "
            f"accuracy {b['frame_accuracy']:.3f})"
        )

        print(f"    water IoU {b['water_iou']:.3f}")

    print("  model  : models/drone_unet/best.pt")
    print("  report : outputs/metrics/stage6_drone_test.json")
    print("  graphs : outputs/graphs/stage6_examples.png")
    print("=" * 72)

    return 0

def cmd_infer_video(args, cfg):
    from vectorvision.utils.config import get_device
    from vectorvision.vision.run_video import run_simulation, run_video

    print("=" * 72)
    print("STAGE 7 - frame-by-frame detection on drone video")
    print("=" * 72)
    dev = get_device(cfg["project"].get("device", "auto"))
    models = p(cfg["paths"]["models_dir"]) / "drone_unet"
    outputs = p(cfg["paths"]["outputs_dir"])

    if args.simulate or not args.video:
        if not args.simulate:
            print("No --video given, so measuring the temporal filter on held-out images instead.\n")
        rep = run_simulation(cfg, outputs / "cache_floodnet", models, outputs, dev,
                             args.images, args.steps)
        r, s = rep["per_frame_raw"], rep["after_persistence_filter"]
        print(f"simulated flights over {rep['images']} held-out images, "
              f"{rep['total_frames']} frames, prevalence {rep['prevalence']:.3f}")
        print(f"\n{'':<22}{'per frame':>12}{'after filter':>14}{'change':>10}")
        for k in ("precision", "recall", "specificity", "f1", "balanced_accuracy"):
            print(f"  {k:<20}{r[k]:>12.3f}{s[k]:>14.3f}{rep['change'][k]:>+10.3f}")
        print(f"\n  false alarms: {r['fp']} -> {s['fp']}   missed frames: {r['fn']} -> {s['fn']}")
        print("\n  report : outputs/metrics/stage7_filter_eval.json")
    else:
        res = run_video(cfg, Path(args.video), models, outputs, dev)
        print(f"\nframes analysed     : {res['frames_analysed']}")
        print(f"frames with water   : {res['frames_raw_water']} raw -> "
              f"{res['frames_after_filter']} after the persistence filter")
        print(f"pools               : {res['raw_tracks']} tracked -> "
              f"{res['confirmed_pools']} confirmed")
        if res["pools"]:
            print(f"\n  {'severity':<10}{'area':>10}{'conf':>7}{'frames':>8}{'time':>14}")
            for x in res["pools"][:10]:
                print(f"  {x['severity']:<10}{x['area_m2']:>8.1f} m2{x['confidence']:>7.2f}"
                      f"{x['frames_seen']:>8}{x['t_start_s']:>8.1f}-{x['t_end_s']:.1f}s")
        print("\n  report : outputs/metrics/stage7_video.json")
        print("  frames : outputs/graphs/stage7_frames.png")
    print("=" * 72)
    return 0


def cmd_fuse(args, cfg):
    from vectorvision.fusion.grid import build_risk_map
    from vectorvision.fusion.priority import fuse, top_zones

    print("=" * 72)
    print("STAGE 8 - Breeding Site Priority Index")
    print("=" * 72)

    outputs = p(cfg["paths"]["outputs_dir"])
    fuse_dir = outputs / "fusion"
    model = p(cfg["paths"]["models_dir"]) / "risk_rf" / "risk_rf.joblib"

    if not model.exists():
        print("Risk model not found. Run:  python main.py train-risk")
        return 1

    rm_path = fuse_dir / "risk_map.json"

    if args.rebuild_map or not rm_path.exists():
        risk_map = build_risk_map(cfg, model, fuse_dir)
    else:
        risk_map = json.loads(rm_path.read_text())
        print(
            f"using existing risk map: {len(risk_map['cells']):,} cells "
            f"(--rebuild-map to redo)"
        )

    print("\nWHERE TO FLY (satellite side only)")
    print(f"  {'rank':>4}{'lat':>11}{'lon':>11}{'score':>8}  reasons")

    for i, z in enumerate(top_zones(risk_map, args.top), 1):
        print(
            f"  {i:>4}{z['lat']:>11.4f}{z['lon']:>11.4f}"
            f"{z['suitability']:>8.3f}  "
            f"{z['reasons'][0] if z['reasons'] else ''}"
        )

    vj = (
        Path(args.video_json)
        if args.video_json
        else outputs / "metrics" / "stage7_video.json"
    )

    if not vj.exists():
        print(
            f"\nNo drone results yet ({vj.name}). Run Stage 7 on a flight video to"
        )
        print("turn these zones into confirmed GPS targets.")
        print(f"\n  risk map : {fuse_dir / 'risk_map.json'}")
        print("=" * 72)
        return 0

    video = json.loads(vj.read_text())

    cell = None
    if args.cell:
        lat, lon = (float(x) for x in args.cell.split(","))
        cell = {"lat": lat, "lon": lon}

    res = fuse(risk_map, video, cfg, survey_cell=cell)
    (fuse_dir / "targets.json").write_text(json.dumps(res, indent=2))

    print(
        f"\nCONFIRMED TARGETS: {res['targets_found']}   "
        f"flight route {res['route_km']} km"
    )

    if res["targets"]:
        print(
            f"  {'visit':>5}{'PBI':>7}{'band':>10}{'area':>9}"
            f"{'S':>6}{'C':>6}{'A':>6}   position"
        )

        for t in res["targets"][:args.top]:
            f_ = t["factors"]

            print(
                f"  {t.get('visit_order', '-'):>5}"
                f"{t['priority_index']:>7.3f}"
                f"{t['band']:>10}"
                f"{t['area_m2']:>7.1f} m2"
                f"{f_['habitat_suitability']:>6.2f}"
                f"{f_['drone_confirmation']:>6.2f}"
                f"{f_['larval_capacity']:>6.2f}   "
                f"{t['lat']:.4f}, {t['lon']:.4f}"
            )

        print("\n  top target reasons:")

        for r in res["targets"][0]["reasons"]:
            print(f"    - {r}")

    print(f"\n  {res['index_definition']}")
    print(f"  {res['caveat']}")
    print(f"\n  risk map : {fuse_dir / 'risk_map.json'}")
    print(f"  targets  : {fuse_dir / 'targets.json'}")
    print("=" * 72)

    return 0

def cmd_mission(args, cfg):
    from vectorvision.drone.mission import plan_mission

    print("=" * 72)
    print("STAGE 9 - mission planning + payload-drop simulation")
    print("=" * 72)

    outputs = p(cfg["paths"]["outputs_dir"])
    targets_path = outputs / "fusion" / "targets.json"
    out_dir = outputs / "mission"

    if not targets_path.exists():
        print(f"Targets file not found: {targets_path}")
        print("Run Stage 8 first:")
        print("  python main.py fuse")
        return 1

    home = None
    if args.home:
        lat, lon = (float(x) for x in args.home.split(","))
        home = (lat, lon)

    result = plan_mission(
        cfg,
        targets_path,
        out_dir,
        home,
    )

    print("\n" + "=" * 72)
    print("STAGE 9 DONE")
    print("  mission plan : outputs/mission/mission.plan")
    print("  report       : outputs/mission/mission_report.json")
    print("=" * 72)

    return 0
def main():
    ap = argparse.ArgumentParser(description="Vector Vision")
    sub = ap.add_subparsers(dest="cmd")

    # ------------------------------------------------------------------
    # Stage 1
    # ------------------------------------------------------------------
    v = sub.add_parser("verify", help="Stage 1")

    v.add_argument(
        "--samples",
        type=int,
        default=300,
        help="tiles to inspect",
    )

    v.add_argument(
        "--skip-ee",
        action="store_true",
        help="skip the Earth Engine check",
    )

    v.add_argument(
        "--no-extract",
        action="store_true",
        help="do not auto-extract the zip",
    )

    # ------------------------------------------------------------------
    # Stage 2
    # ------------------------------------------------------------------
    t = sub.add_parser("train-lulc", help="Stage 2")

    t.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="override config epochs",
    )

    t.add_argument(
        "--subset",
        type=int,
        default=None,
        help="override number of training tiles",
    )

    t.add_argument(
        "--resume",
        action="store_true",
        help="continue from models/unet_lulc/last.pt",
    )

    t.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="re-read the dataset",
    )

    t.add_argument(
        "--cache-only",
        action="store_true",
        help="build the cache, then stop",
    )

    t.add_argument(
        "--allow-cpu",
        action="store_true",
        help="train without a GPU",
    )

    # ------------------------------------------------------------------
    # Stage 3
    # ------------------------------------------------------------------
    e = sub.add_parser("eval-lulc", help="Stage 3")

    e.add_argument(
        "--tiles",
        type=int,
        default=10000,
        help="test tiles to evaluate",
    )

    # ------------------------------------------------------------------
    # Stage 4
    # ------------------------------------------------------------------
    b = sub.add_parser("build-features", help="Stage 4")

    b.add_argument(
        "--rebuild",
        action="store_true",
        help="re-download and re-sample",
    )

    # ------------------------------------------------------------------
    # Stage 5
    # ------------------------------------------------------------------
    sub.add_parser("train-risk", help="Stage 5")

    # ------------------------------------------------------------------
    # Stage 6
    # ------------------------------------------------------------------
    dr = sub.add_parser("train-drone", help="Stage 6")

    dr.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="override config epochs",
    )

    dr.add_argument(
        "--resume",
        action="store_true",
        help="continue from models/drone_unet/last.pt",
    )

    dr.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="re-read and rebuild the FloodNet cache",
    )

    dr.add_argument(
        "--cache-only",
        action="store_true",
        help="build the FloodNet cache, then stop",
    )

    dr.add_argument(
        "--inspect",
        action="store_true",
        help="show the FloodNet dataset layout, then stop",
    )

    dr.add_argument(
        "--allow-cpu",
        action="store_true",
        help="allow extremely slow CPU training",
    )

    dr.add_argument(
        "--test-only",
        action="store_true",
        help="re-test the saved model without retraining",
    )


    # ------------------------------------------------------------------
    # Stage 8
    # ------------------------------------------------------------------
    fu = sub.add_parser("fuse", help="Stage 8")

    fu.add_argument(
        "--rebuild-map",
        action="store_true",
        help="re-score the district grid",
    )

    fu.add_argument(
        "--video-json",
        default=None,
        help="Stage 7 output to fuse",
    )

    fu.add_argument(
        "--cell",
        default=None,
        metavar="LAT,LON",
        help="survey area centre, used when the flight had no telemetry",
    )

    fu.add_argument(
        "--top",
        type=int,
        default=10,
        help="number of targets/zones to print",
    )

# ------------------------------------------------------------------
# Stage 9
# ------------------------------------------------------------------
    mi = sub.add_parser("mission", help="Stage 9")

    mi.add_argument(
        "--home",
        default=None,
        metavar="LAT,LON",
        help="home position for the mission, e.g. 21.45,80.19",
    )

    # ------------------------------------------------------------------
    # General
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Stage 7
    # ------------------------------------------------------------------
    iv = sub.add_parser("infer-video", help="Stage 7")

    iv.add_argument(
        "--video",
        default=None,
        help="path to an MP4 from the drone",
    )

    iv.add_argument(
        "--simulate",
        action="store_true",
        help="measure the temporal filter on held-out FloodNet images",
    )

    iv.add_argument(
        "--images",
        type=int,
        default=20,
        help="images to simulate flights over",
    )

    iv.add_argument(
        "--steps",
        type=int,
        default=12,
        help="frames per simulated flight",
    )

    # ------------------------------------------------------------------
    # General
    # ------------------------------------------------------------------
    sub.add_parser("stages", help="list stages")
    args = ap.parse_args()

    cfg = load_config()

    seed_everything(cfg["project"]["seed"])

    if args.cmd == "verify":
        sys.exit(cmd_verify(args, cfg))

    if args.cmd == "train-lulc":
        sys.exit(cmd_train_lulc(args, cfg))

    if args.cmd == "eval-lulc":
        sys.exit(cmd_eval_lulc(args, cfg))

    if args.cmd == "build-features":
        sys.exit(cmd_build_features(args, cfg))

    if args.cmd == "train-risk":
        sys.exit(cmd_train_risk(args, cfg))

    if args.cmd == "train-drone":
        sys.exit(cmd_train_drone(args, cfg))

    if args.cmd == "infer-video":
        sys.exit(cmd_infer_video(args, cfg))

    if args.cmd == "fuse":
        sys.exit(cmd_fuse(args, cfg))

    if args.cmd == "mission":
        sys.exit(cmd_mission(args, cfg))

    cmd_stages(args, cfg)


if __name__ == "__main__":
    main()