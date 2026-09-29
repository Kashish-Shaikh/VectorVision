"""Stage 5 — the risk model: Random Forest on real occurrence records.

What the model actually learns
------------------------------
Presence rows are places where Anopheles mosquitoes were really caught. Background
rows are random places in the same country. So the model learns:

    "how much do the conditions here resemble places where vectors were found,
     compared with the country as a whole?"

That is habitat suitability. It is NOT a probability that a specific 100 m cell has
larvae today. Only the drone stage can check that.

Three controls, because a single good-looking score proves nothing
------------------------------------------------------------------
1. Spatial block cross-validation. Nearby points share weather, terrain and the same
   survey. Random folds let the model memorise a place from its neighbours, which
   inflates the score. Folds are whole 1-degree blocks instead.
2. Random-fold comparison. Running both shows how much leakage there would have been.
   A big gap is evidence that the spatial score is the honest one.
3. Label shuffle. Train on shuffled labels; AUC must fall to about 0.5. If it does
   not, something in the pipeline is leaking the answer.

A coordinates-only baseline is also reported. If latitude and longitude alone score
as well as the environmental features, the model has learned geography, not ecology.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _load(csv_p: Path):
    import pandas as pd
    df = pd.read_csv(csv_p)
    y = df["presence"].to_numpy(int)
    coords = df[["lat", "lon"]].to_numpy(float)
    feats = [c for c in df.columns if c not in ("lat", "lon", "presence")]
    X = df[feats].to_numpy(float)
    ok = np.isfinite(X).all(1)
    if (~ok).any():
        print(f"  dropped {int((~ok).sum())} rows with non-finite values")
    return X[ok], y[ok], coords[ok], feats


def _blocks(coords: np.ndarray, deg: float) -> np.ndarray:
    """Whole lat/lon blocks become fold groups, so neighbours stay together."""
    return np.array([f"{int(a // deg)}_{int(b // deg)}" for a, b in coords])


def _forest(cfg: dict, seed: int):
    from sklearn.ensemble import RandomForestClassifier
    r = cfg["risk_model"]
    return RandomForestClassifier(
        n_estimators=r["n_estimators"], max_depth=None, min_samples_leaf=3,
        max_features="sqrt", class_weight="balanced_subsample",
        n_jobs=-1, random_state=seed)


def _cv_scores(X, y, groups, cfg, seed, spatial=True, n_splits=5):
    """Out-of-fold predictions, so every score comes from unseen data."""
    from sklearn.model_selection import GroupKFold, StratifiedKFold
    oof = np.full(len(y), np.nan)
    if spatial:
        uniq = np.unique(groups)
        k = min(n_splits, len(uniq))
        if k < 2:
            raise SystemExit("Not enough spatial blocks for cross-validation.")
        splitter = GroupKFold(n_splits=k).split(X, y, groups)
    else:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(X, y)
    for tr, te in splitter:
        if len(np.unique(y[tr])) < 2:
            continue
        m = _forest(cfg, seed).fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    return oof


def _auc_ci(y, p, seed=42, n_boot=1000):
    from sklearn.metrics import average_precision_score, roc_auc_score
    ok = np.isfinite(p)
    y, p = y[ok], p[ok]
    rng = np.random.default_rng(seed)
    a = np.empty(n_boot)
    for b in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        a[b] = roc_auc_score(y[i], p[i]) if len(np.unique(y[i])) > 1 else np.nan
    return {"roc_auc": float(roc_auc_score(y, p)),
            "pr_auc": float(average_precision_score(y, p)),
            "prevalence": float(y.mean()),
            "roc_auc_ci95": [float(np.nanpercentile(a, 2.5)), float(np.nanpercentile(a, 97.5))]}


def _at_threshold(y, p, thr):
    from sklearn.metrics import confusion_matrix
    ok = np.isfinite(p)
    pred = (p[ok] >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y[ok], pred, labels=[0, 1]).ravel()
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return {"threshold": float(thr), "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
            "precision": float(prec), "recall": float(rec),
            "f1": float(2 * prec * rec / max(prec + rec, 1e-12)),
            "specificity": float(tn / max(tn + fp, 1))}


def train(cfg: dict, table: Path, out_models: Path, out_dir: Path) -> dict:
    import joblib
    from sklearn.inspection import permutation_importance

    seed = cfg["project"]["seed"]
    deg = cfg["risk_model"]["spatial_block_deg"]
    out_models.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics").mkdir(parents=True, exist_ok=True)

    X, y, coords, feats = _load(table)
    groups = _blocks(coords, deg)
    print(f"{len(y):,} rows | {int(y.sum()):,} presence, {int((1-y).sum()):,} background | "
          f"{len(feats)} features | {len(np.unique(groups))} spatial blocks of {deg}deg")

    # ---- main result: spatially blocked cross-validation
    oof = _cv_scores(X, y, groups, cfg, seed, spatial=True)
    main = _auc_ci(y, oof, seed)
    print(f"\nSPATIAL CV   ROC-AUC {main['roc_auc']:.3f} "
          f"(95% CI {main['roc_auc_ci95'][0]:.3f}-{main['roc_auc_ci95'][1]:.3f})   "
          f"PR-AUC {main['pr_auc']:.3f}  (prevalence {main['prevalence']:.3f})")

    # ---- control 1: random folds, to measure how much leakage they allow
    oof_rand = _cv_scores(X, y, groups, cfg, seed, spatial=False)
    rand = _auc_ci(y, oof_rand, seed)
    gap = rand["roc_auc"] - main["roc_auc"]
    print(f"RANDOM CV    ROC-AUC {rand['roc_auc']:.3f}   (gap {gap:+.3f} -> "
          f"{'spatial leakage is real; the spatial score is the honest one' if gap > 0.03 else 'little leakage'})")

    # ---- control 2: shuffled labels must score about 0.5
    rng = np.random.default_rng(seed)
    y_shuf = rng.permutation(y)
    oof_shuf = _cv_scores(X, y_shuf, groups, cfg, seed, spatial=True)
    shuf = _auc_ci(y_shuf, oof_shuf, seed)
    print(f"LABEL SHUFFLE ROC-AUC {shuf['roc_auc']:.3f}   "
          f"({'PASS, no hidden leak' if abs(shuf['roc_auc'] - .5) < .08 else 'FAIL - investigate'})")

    # ---- control 3: coordinates only (is it ecology, or just geography?)
    oof_xy = _cv_scores(coords, y, groups, cfg, seed, spatial=True)
    xy = _auc_ci(y, oof_xy, seed)
    print(f"COORDS ONLY  ROC-AUC {xy['roc_auc']:.3f}   "
          f"({'features add real information' if main['roc_auc'] - xy['roc_auc'] > 0.03 else 'WARNING: mostly geography'})")

    # ---- operating point, chosen on out-of-fold predictions
    best = max((_at_threshold(y, oof, t) for t in np.arange(0.05, 0.96, 0.05)),
               key=lambda d: d["f1"])
    print(f"\nbest F1 at threshold {best['threshold']:.2f}: precision {best['precision']:.3f}, "
          f"recall {best['recall']:.3f}, specificity {best['specificity']:.3f}")

    # ---- final model on all rows, then explain it
    model = _forest(cfg, seed).fit(X, y)
    pi = permutation_importance(model, X, y, n_repeats=10, random_state=seed, n_jobs=-1,
                                scoring="roc_auc")
    order = np.argsort(-pi.importances_mean)
    print("\nfeature importance (drop in AUC when the feature is shuffled):")
    for i in order[:10]:
        print(f"    {feats[i]:<18}{pi.importances_mean[i]:+.4f}  +/- {pi.importances_std[i]:.4f}")

    shap_info = _shap(model, X, feats, out_dir / "graphs", seed)
    joblib.dump({"model": model, "features": feats, "threshold": best["threshold"],
                 "medians": np.median(X, 0).tolist()}, out_models / "risk_rf.joblib")

    report = {
        "rows": int(len(y)), "presence": int(y.sum()), "background": int((1 - y).sum()),
        "features": feats, "spatial_blocks": int(len(np.unique(groups))), "block_degrees": deg,
        "spatial_cv": main, "random_cv": rand, "leakage_gap": float(gap),
        "label_shuffle_control": shuf, "coords_only_control": xy,
        "operating_point": best,
        "permutation_importance": [{"feature": feats[i],
                                    "mean_auc_drop": float(pi.importances_mean[i]),
                                    "std": float(pi.importances_std[i])} for i in order],
        "shap": shap_info,
        "interpretation": {
            "what_it_predicts": "Similarity of local conditions to places where Anopheles "
                                "vectors were recorded, versus the country as a whole.",
            "what_it_does_not_predict": "Whether a given cell contains larvae now, or malaria cases.",
            "labels": "1 = real field record. 0 = background point, not a confirmed absence.",
        },
    }
    (out_dir / "metrics" / "stage5_risk_report.json").write_text(json.dumps(report, indent=2))
    _plots(y, oof, oof_rand, feats, pi, order, out_dir / "graphs")
    return report


def _shap(model, X, feats, gdir: Path, seed):
    try:
        import shap
    except ImportError:
        print("\n(shap not installed - permutation importance above is the explanation; "
              "pip install shap to add per-prediction reasons)")
        return {"available": False}
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    gdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    sub = X[rng.choice(len(X), min(400, len(X)), replace=False)]
    vals = shap.TreeExplainer(model).shap_values(sub)
    if isinstance(vals, list):
        vals = vals[1]
    elif vals.ndim == 3:
        vals = vals[:, :, 1]
    plt.figure()
    shap.summary_plot(vals, sub, feature_names=feats, show=False, max_display=12)
    plt.tight_layout(); plt.savefig(gdir / "stage5_shap_summary.png", dpi=150); plt.close()
    mean_abs = np.abs(vals).mean(0)
    order = np.argsort(-mean_abs)
    print("\nSHAP: features that move predictions most")
    for i in order[:8]:
        print(f"    {feats[i]:<18}{mean_abs[i]:.4f}")
    return {"available": True, "sampled_rows": int(len(sub)),
            "mean_abs": [{"feature": feats[i], "value": float(mean_abs[i])} for i in order]}


def explain_row(bundle: dict, row: np.ndarray, top_k: int = 4) -> dict:
    """Plain-language reasons for one cell: swap each feature to its median and see
    how far the score moves. Used by the dashboard."""
    model, feats, med = bundle["model"], bundle["features"], np.array(bundle["medians"])
    full = float(model.predict_proba(row.reshape(1, -1))[0, 1])
    base = float(model.predict_proba(med.reshape(1, -1))[0, 1])
    effects = []
    for j, f in enumerate(feats):
        swapped = row.copy(); swapped[j] = med[j]
        effects.append((f, full - float(model.predict_proba(swapped.reshape(1, -1))[0, 1])))
    effects.sort(key=lambda kv: -abs(kv[1]))
    return {"score": full, "baseline": base,
            "reasons": [{"feature": f, "effect": round(e, 4),
                         "direction": "raises" if e > 0 else "lowers"} for f, e in effects[:top_k]]}


def _plots(y, oof, oof_rand, feats, pi, order, gdir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, roc_curve
    gdir.mkdir(parents=True, exist_ok=True)

    ok = np.isfinite(oof)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    for p, lab in ((oof, "spatial CV"), (oof_rand, "random CV (optimistic)")):
        m = np.isfinite(p)
        fpr, tpr, _ = roc_curve(y[m], p[m])
        ax[0].plot(fpr, tpr, label=lab)
    ax[0].plot([0, 1], [0, 1], "k--", lw=.8)
    ax[0].set_xlabel("false positive rate"); ax[0].set_ylabel("true positive rate")
    ax[0].set_title("ROC"); ax[0].legend(frameon=False); ax[0].grid(alpha=.3)
    pr, rc, _ = precision_recall_curve(y[ok], oof[ok])
    ax[1].plot(rc, pr); ax[1].axhline(y.mean(), ls="--", c="k", lw=.8, label="random guess")
    ax[1].set_xlabel("recall"); ax[1].set_ylabel("precision")
    ax[1].set_title("Precision-recall (spatial CV)"); ax[1].legend(frameon=False); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(gdir / "stage5_curves.png", dpi=150); plt.close(fig)

    k = order[:12][::-1]
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.barh([feats[i] for i in k], [pi.importances_mean[i] for i in k],
            xerr=[pi.importances_std[i] for i in k])
    ax.set_xlabel("drop in ROC-AUC when shuffled"); ax.grid(axis="x", alpha=.3)
    fig.tight_layout(); fig.savefig(gdir / "stage5_importance.png", dpi=150); plt.close(fig)
