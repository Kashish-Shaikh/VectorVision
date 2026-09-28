"""Vector Vision — one command line for every stage.

    python main.py verify            Stage 1: environment + dataset verification
    python main.py stages            show every stage and its status

Later stages are added one at a time, and only after the previous one works.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from vectorvision.utils.config import load_config, p, seed_everything  # noqa: E402

STAGES = [
    (1, "verify",        "Environment + dataset verification",               "ready"),
    (2, "train-lulc",    "Satellite U-Net (from scratch) on Sen-2 LULC",       "ready"),
    (3, "eval-lulc",     "Test U-Net on held-out tiles, per-class IoU",        "ready"),
    (4, "build-features","Gondiya 100 m grid + vector-occurrence training set", "planned"),
    (5, "train-risk",    "Random Forest risk model + SHAP explanations",       "planned"),
    (6, "train-drone",   "Drone water U-Net (from scratch) on FloodNet",       "planned"),
    (7, "infer-video",   "Frame-by-frame video detection + tracking",          "planned"),
    (8, "fuse",          "Breeding Site Priority Index + GPS targets",         "planned"),
    (9, "mission",       "Waypoints, flight + payload-drop simulation",        "planned"),
    (10, "dashboard",    "Web dashboard",                                      "planned"),
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

    rep = inspect_sen2lulc(p(cfg["paths"]["sen2lulc_root"]),
                           p(cfg["paths"]["outputs_dir"]) / "metrics",
                           n_samples=args.samples, do_extract=not args.no_extract,
                           seed=cfg["project"]["seed"])

    print("\nLater datasets (download in their own stages)")
    for label, key in (("FloodNet drone imagery", "floodnet_root"),
                       ("Anopheles occurrence points", "vectors_dir")):
        path = p(cfg["paths"][key])
        present = path.exists() and any(path.iterdir())
        print(f"  {'[ OK ]' if present else '[----]'} {label:<30} {'present' if present else 'not yet - fine for now'}")

    fails = [e for e in env if e["status"] == "FAIL"]
    ds_ok = rep.get("status") in ("PASS", "WARN")
    summary = {"environment": env, "dataset_status": rep.get("status"),
               "stage1_passed": not fails and ds_ok}
    out = p(cfg["paths"]["outputs_dir"]) / "metrics" / "stage1_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, default=str))

    print("\n" + "=" * 72)
    if summary["stage1_passed"]:
        print("STAGE 1 PASSED")
        print("  report : outputs/metrics/stage1_dataset_report.json")
        print("  figure : outputs/graphs/stage1_samples.png  <- open it and name the classes")
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
        print("No GPU found. Turn one on (Kaggle: Settings > Accelerator > GPU T4 x2;")
        print("Colab: Runtime > Change runtime type > T4 GPU), or pass --allow-cpu to force it.")
        return 1
    if args.subset:
        cfg["satellite_unet"]["train_subset"] = args.subset
    outputs = p(cfg["paths"]["outputs_dir"])
    cache = build_cache(cfg, p(cfg["paths"]["sen2lulc_root"]), outputs, force=args.rebuild_cache)
    if args.cache_only:
        print("cache built; stopping as asked (--cache-only)")
        return 0
    from vectorvision.satellite.train import train   # imports torch; not needed for --cache-only
    res = train(cfg, cache, p(cfg["paths"]["models_dir"]) / "unet_lulc", dev,
                epochs=args.epochs, resume=args.resume)
    print("\n" + "=" * 72)
    print(f"STAGE 2 DONE - best validation mean IoU {res['best_mean_iou']:.4f}")
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
    rep = evaluate(cfg, p(cfg["paths"]["sen2lulc_root"]), p(cfg["paths"]["outputs_dir"]),
                   p(cfg["paths"]["models_dir"]) / "unet_lulc", dev, args.tiles)
    w = rep["water_index"]
    print(f"\nTEST RESULTS on {rep['test_tiles']:,} unseen tiles")
    for tag in ("raw_labels", "cleaned_labels"):
        r = rep[tag]
        print(f"\n  {tag.replace('_', ' ')}: mean IoU {r['mean_iou']:.4f} "
              f"(95% CI {r['ci95']['mean_iou'][0]:.3f}-{r['ci95']['mean_iou'][1]:.3f}), "
              f"pixel accuracy {r['pixel_accuracy']:.4f}")
        print(f"    {'class':<16}{'IoU':>7}{'prec':>7}{'recall':>8}{'F1':>7}{'share':>8}")
        for c in r["per_class"]:
            print(f"    {c['class']:<16}{c['iou']:7.3f}{c['precision']:7.3f}{c['recall']:8.3f}"
                  f"{c['f1']:7.3f}{c['share_pct']:7.2f}%")
        print(f"    water IoU 95% CI {r['ci95']['water_iou'][0]:.3f}-{r['ci95']['water_iou'][1]:.3f}")
    t = rep["tile_water_detection"]
    print(f"\n  does the tile contain a water body? accuracy {t['accuracy']:.3f} "
          f"(95% CI {t['accuracy_ci95'][0]:.3f}-{t['accuracy_ci95'][1]:.3f}), "
          f"precision {t['precision']:.3f}, recall {t['recall']:.3f}, "
          f"{t['positive_tiles']} of {t['tiles']} tiles contain water")
    print("\n  water threshold sweep (cleaned labels):")
    for s_ in rep["water_threshold_sweep"][::2]:
        print(f"    threshold {s_['threshold']:.2f}  precision {s_['precision']:.3f}  "
              f"recall {s_['recall']:.3f}  IoU {s_['iou']:.3f}")
    print("\n" + "=" * 72)
    print("STAGE 3 DONE")
    print("  report : outputs/metrics/stage3_test_report.json")
    print("  graphs : outputs/graphs/stage3_*.png")
    print("=" * 72)
    return 0


def main():
    ap = argparse.ArgumentParser(description="Vector Vision")
    sub = ap.add_subparsers(dest="cmd")
    v = sub.add_parser("verify", help="Stage 1")
    v.add_argument("--samples", type=int, default=300, help="tiles to inspect")
    v.add_argument("--skip-ee", action="store_true", help="skip the Earth Engine check")
    v.add_argument("--no-extract", action="store_true", help="do not auto-extract the zip")
    t = sub.add_parser("train-lulc", help="Stage 2")
    t.add_argument("--epochs", type=int, default=None, help="override config epochs")
    t.add_argument("--subset", type=int, default=None, help="override number of training tiles")
    t.add_argument("--resume", action="store_true", help="continue from models/unet_lulc/last.pt")
    t.add_argument("--rebuild-cache", action="store_true", help="re-read the dataset")
    t.add_argument("--cache-only", action="store_true", help="build the cache, then stop")
    t.add_argument("--allow-cpu", action="store_true", help="train without a GPU (very slow)")
    e = sub.add_parser("eval-lulc", help="Stage 3")
    e.add_argument("--tiles", type=int, default=10000, help="test tiles to evaluate (max ~32,079)")
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
    cmd_stages(args, cfg)


if __name__ == "__main__":
    main()
