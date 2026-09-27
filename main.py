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
    (3, "eval-lulc",     "Test U-Net on held-out tiles, per-class IoU",        "next"),
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
    sub.add_parser("stages", help="list stages")
    args = ap.parse_args()

    cfg = load_config()
    seed_everything(cfg["project"]["seed"])
    if args.cmd == "verify":
        sys.exit(cmd_verify(args, cfg))
    if args.cmd == "train-lulc":
        sys.exit(cmd_train_lulc(args, cfg))
    cmd_stages(args, cfg)


if __name__ == "__main__":
    main()
