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
    (2, "train-lulc",    "Satellite U-Net (from scratch) on Sen-2 LULC",       "next"),
    (3, "eval-lulc",     "Test U-Net on held-out tiles, per-class IoU",        "planned"),
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


def main():
    ap = argparse.ArgumentParser(description="Vector Vision")
    sub = ap.add_subparsers(dest="cmd")
    v = sub.add_parser("verify", help="Stage 1")
    v.add_argument("--samples", type=int, default=300, help="tiles to inspect")
    v.add_argument("--skip-ee", action="store_true", help="skip the Earth Engine check")
    v.add_argument("--no-extract", action="store_true", help="do not auto-extract the zip")
    sub.add_parser("stages", help="list stages")
    args = ap.parse_args()

    cfg = load_config()
    seed_everything(cfg["project"]["seed"])
    if args.cmd == "verify":
        sys.exit(cmd_verify(args, cfg))
    cmd_stages(args, cfg)


if __name__ == "__main__":
    main()
