"""
ML pipeline for KTH DNS boundary layer data.

Three ML tasks on real DNS data:
1. Boundary layer region classification (viscous sublayer / buffer / log / wake)
2. Re_tau prediction from profile features (regression)
3. Anomaly detection — identify unusual profile points across Re_theta

Features extracted via SQL; predictions written back to database.
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingRegressor,
    IsolationForest,
    RandomForestClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sqlalchemy import text

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import DatabaseConfig
from src.db import get_engine, init_schema
from src.feature_engineering import compute_derived_features, get_profile_features_per_point

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Task 1: BL Region Classification
# ---------------------------------------------------------------------------

def train_region_classifier(engine) -> dict:
    """
    Classify boundary layer regions (viscous sublayer, buffer, log, wake, freestream)
    from velocity profile features at each (Re_theta, y+) point.
    """
    logger.info("Extracting per-point features via SQL...")
    df = get_profile_features_per_point(engine)

    feature_cols = [
        "u_plus", "u_rms_plus", "v_rms_plus", "w_rms_plus",
        "uv_plus", "p_rms_plus", "skewness_u", "flatness_u",
        "du_dy_plus", "tke_plus",
    ]

    # Exclude freestream (sparse) for cleaner classification
    df = df[df["bl_region"] != "freestream"].copy()
    df = df.dropna(subset=feature_cols)

    X = df[feature_cols].values
    y = df["bl_region"].values

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_sc, y_enc, test_size=0.25, random_state=42, stratify=y_enc
    )

    clf = RandomForestClassifier(
        n_estimators=200, max_depth=15, min_samples_split=5,
        random_state=42, class_weight="balanced"
    )
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)

    acc = accuracy_score(y_te, y_pred)
    f1 = f1_score(y_te, y_pred, average="weighted")

    logger.info(f"Region classifier — Accuracy: {acc:.3f}, F1: {f1:.3f}")
    logger.info(f"\n{classification_report(y_te, y_pred, target_names=le.classes_)}")

    # Feature importance
    importances = pd.Series(clf.feature_importances_, index=feature_cols)
    logger.info(f"Top features:\n{importances.sort_values(ascending=False).head(5)}")

    # Cross-validation
    cv = cross_val_score(clf, X_sc, y_enc, cv=5, scoring="accuracy")
    logger.info(f"5-fold CV: {cv.mean():.3f} ± {cv.std():.3f}")

    return {
        "model": clf, "scaler": scaler, "label_encoder": le,
        "feature_cols": feature_cols,
        "metrics": {"accuracy": acc, "f1_score": f1, "cv_mean": cv.mean(), "cv_std": cv.std()},
        "confusion_matrix": confusion_matrix(y_te, y_pred),
        "X_test": X_te, "y_test": y_te, "y_pred": y_pred,
        "importances": importances,
        "df": df, "X_scaled": X_sc, "y_encoded": y_enc,
    }


# ---------------------------------------------------------------------------
# Task 2: Re_tau Regression from Profile Features
# ---------------------------------------------------------------------------

def train_retau_predictor(engine) -> dict:
    """
    Predict Re_tau from SQL-derived profile features.
    Demonstrates regression on a physically meaningful target.
    """
    logger.info("Extracting derived features via SQL CTE pipeline...")
    features = compute_derived_features(engine)

    feature_cols = [
        "u_rms_peak", "v_rms_peak", "w_rms_peak", "uv_peak",
        "tke_peak", "p_rms_peak", "anisotropy_ratio",
        "production_peak", "dissipation_wall",
        "shape_factor", "u_rms_peak_yplus",
    ]

    df = features.dropna(subset=feature_cols + ["re_tau"]).copy()
    X = df[feature_cols].values
    y = df["re_tau"].values

    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)

    # With only 10 data points, use leave-one-out style evaluation
    from sklearn.model_selection import LeaveOneOut
    loo = LeaveOneOut()

    y_pred_loo = np.zeros_like(y)
    for train_idx, test_idx in loo.split(X_sc):
        reg = GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42)
        reg.fit(X_sc[train_idx], y[train_idx])
        y_pred_loo[test_idx] = reg.predict(X_sc[test_idx])

    r2 = r2_score(y, y_pred_loo)
    mae = mean_absolute_error(y, y_pred_loo)
    rmse = np.sqrt(mean_squared_error(y, y_pred_loo))

    logger.info(f"Re_tau predictor (LOO-CV) — R²: {r2:.3f}, MAE: {mae:.1f}, RMSE: {rmse:.1f}")

    # Train final model on all data
    reg_final = GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42)
    reg_final.fit(X_sc, y)

    importances = pd.Series(reg_final.feature_importances_, index=feature_cols)
    logger.info(f"Top features:\n{importances.sort_values(ascending=False).head(5)}")

    return {
        "model": reg_final, "scaler": scaler, "feature_cols": feature_cols,
        "metrics": {"r2": r2, "mae": mae, "rmse": rmse},
        "y_true": y, "y_pred": y_pred_loo,
        "re_theta": df["re_theta"].values,
        "importances": importances,
    }


# ---------------------------------------------------------------------------
# Task 3: Anomaly Detection on Profile Points
# ---------------------------------------------------------------------------

def detect_anomalies(engine) -> dict:
    """
    Detect anomalous profile points using Isolation Forest + LOF.
    An anomalous point might indicate DNS resolution issues,
    insufficient averaging, or unusual flow physics.
    """
    df = get_profile_features_per_point(engine)

    feature_cols = [
        "u_rms_plus", "v_rms_plus", "w_rms_plus",
        "uv_plus", "tke_plus", "skewness_u", "flatness_u",
    ]

    df_clean = df.dropna(subset=feature_cols).copy()
    X = df_clean[feature_cols].values
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)

    iso = IsolationForest(n_estimators=200, contamination=0.03, random_state=42)
    iso_labels = iso.fit_predict(X_sc)
    iso_scores = iso.decision_function(X_sc)

    lof = LocalOutlierFactor(n_neighbors=30, contamination=0.03)
    lof_labels = lof.fit_predict(X_sc)

    consensus = (iso_labels == -1) & (lof_labels == -1)

    n_iso = (iso_labels == -1).sum()
    n_lof = (lof_labels == -1).sum()
    n_both = consensus.sum()

    logger.info(f"Isolation Forest: {n_iso}/{len(df_clean)} anomalies")
    logger.info(f"LOF: {n_lof}/{len(df_clean)} anomalies")
    logger.info(f"Consensus: {n_both} anomalies")

    df_clean["iso_anomaly"] = iso_labels == -1
    df_clean["lof_anomaly"] = lof_labels == -1
    df_clean["consensus_anomaly"] = consensus
    df_clean["anomaly_score"] = iso_scores

    return {
        "df": df_clean,
        "n_anomalies": {"iso": n_iso, "lof": n_lof, "consensus": n_both},
        "feature_cols": feature_cols,
    }


# ---------------------------------------------------------------------------
# Write predictions to database
# ---------------------------------------------------------------------------

def write_results_to_db(engine, region_results, retau_results, anomaly_results):
    """Write all ML results back to database with ACID transactions."""
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM predictions"))

        # Region classification: write per-point predictions
        # (too many rows — write summary instead)
        metrics = region_results["metrics"]
        conn.execute(text("""
            INSERT INTO model_registry (model_name, model_version, model_type,
                hyperparameters, metric_name, metric_value, n_training_samples, notes)
            VALUES (:name, :ver, :type, :params, :mname, :mval, :n, :notes)
        """), {
            "name": "region_classifier", "ver": "v1.0", "type": "classifier",
            "params": json.dumps({"n_estimators": 200, "max_depth": 15}),
            "mname": "accuracy", "mval": metrics["accuracy"],
            "n": len(region_results["y_encoded"]),
            "notes": f"BL region classification, 5-fold CV: {metrics['cv_mean']:.3f}",
        })

        # Re_tau regression predictions
        for i in range(len(retau_results["y_true"])):
            conn.execute(text("""
                INSERT INTO predictions (condition_id, model_name, model_version,
                    predicted_re_tau)
                VALUES (
                    (SELECT condition_id FROM simulation_conditions
                     ORDER BY ABS(re_theta - :rt) LIMIT 1),
                    'retau_regressor', 'v1.0', :pred
                )
            """), {
                "rt": float(retau_results["re_theta"][i]),
                "pred": float(retau_results["y_pred"][i]),
            })

        # Regression model registry
        conn.execute(text("""
            INSERT INTO model_registry (model_name, model_version, model_type,
                metric_name, metric_value, n_training_samples, notes)
            VALUES (:name, :ver, :type, :mname, :mval, :n, :notes)
        """), {
            "name": "retau_regressor", "ver": "v1.0", "type": "regressor",
            "mname": "r2", "mval": retau_results["metrics"]["r2"],
            "n": len(retau_results["y_true"]),
            "notes": f"LOO-CV, MAE={retau_results['metrics']['mae']:.1f}",
        })

    logger.info("Results written to database")


# ---------------------------------------------------------------------------
# Generate ML figures
# ---------------------------------------------------------------------------

def generate_ml_figures(region_results, retau_results, anomaly_results):
    """Generate ML-specific figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    plt.rcParams.update({
        "font.family": "serif", "font.size": 11, "axes.labelsize": 13,
        "savefig.dpi": 300, "savefig.bbox": "tight",
        "axes.grid": True, "grid.alpha": 0.25,
    })
    OUTPUT = "data/processed"

    # Fig 11: Region classification confusion matrix + importance
    fig = plt.figure(figsize=(14, 5.5))
    gs = plt.GridSpec(1, 2, width_ratios=[1, 1.2], wspace=0.35)

    ax1 = fig.add_subplot(gs[0])
    le = region_results["label_encoder"]
    cm = region_results["confusion_matrix"]
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    disp.plot(ax=ax1, cmap="Blues", values_format="d", colorbar=False)
    ax1.set_title(f"(a) Region Classification — Acc: {region_results['metrics']['accuracy']:.1%}", fontsize=12)
    ax1.tick_params(axis="x", rotation=25)

    ax2 = fig.add_subplot(gs[1])
    imp = region_results["importances"].sort_values()
    ax2.barh(range(len(imp)), imp.values, color="#3498db", edgecolor="0.3", height=0.7)
    ax2.set_yticks(range(len(imp)))
    ax2.set_yticklabels(imp.index, fontsize=9)
    ax2.set_xlabel("Feature Importance (Gini)")
    ax2.set_title("(b) Feature Importance", fontsize=12)
    for i in range(-1, -4, -1):
        ax2.text(imp.values[i]+0.005, len(imp)+i, f"{imp.values[i]:.3f}", va="center", fontsize=8, color="0.3")

    plt.tight_layout()
    fig.savefig(f"{OUTPUT}/fig11_region_classification.png")
    plt.close(fig)
    print("  [11] Region classification")

    # Fig 12: Re_tau prediction — true vs predicted
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    ax = axes[0]
    ax.plot(retau_results["y_true"], retau_results["y_pred"], "ko", ms=8, zorder=3)
    lims = [200, 1400]
    ax.plot(lims, lims, "r--", lw=1.2, alpha=0.6, label="Perfect prediction")
    for i in range(len(retau_results["y_true"])):
        ax.annotate(f"  {retau_results['re_theta'][i]:.0f}",
                     (retau_results["y_true"][i], retau_results["y_pred"][i]),
                     fontsize=7, color="0.4")
    ax.set_xlabel("True $Re_\\tau$"); ax.set_ylabel("Predicted $Re_\\tau$")
    r2 = retau_results["metrics"]["r2"]
    ax.set_title(f"(a) $Re_\\tau$ Prediction (LOO-CV, $R^2={r2:.3f}$)")
    ax.legend(fontsize=9); ax.set_xlim(lims); ax.set_ylim(lims)

    ax = axes[1]
    imp_r = retau_results["importances"].sort_values()
    ax.barh(range(len(imp_r)), imp_r.values, color="#e67e22", edgecolor="0.3", height=0.7)
    ax.set_yticks(range(len(imp_r))); ax.set_yticklabels(imp_r.index, fontsize=9)
    ax.set_xlabel("Feature Importance"); ax.set_title("(b) GBR Feature Importance")

    plt.suptitle("$Re_\\tau$ Regression from SQL-Engineered Features", fontsize=14, y=1.01)
    plt.tight_layout()
    fig.savefig(f"{OUTPUT}/fig12_retau_prediction.png")
    plt.close(fig)
    print("  [12] Re_tau prediction")

    # Fig 13: Anomaly detection map
    df_a = anomaly_results["df"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    ax = axes[0]
    normal = df_a[~df_a["iso_anomaly"]]
    anom = df_a[df_a["iso_anomaly"]]
    sc = ax.scatter(normal["re_theta"], normal["y_plus"], c=normal["anomaly_score"],
                    cmap="RdYlGn", s=5, alpha=0.4)
    ax.scatter(anom["re_theta"], anom["y_plus"], c="red", s=30, marker="x",
               linewidths=1, zorder=5, label=f"IF anomaly ({len(anom)})")
    plt.colorbar(sc, ax=ax, label="Anomaly Score", shrink=0.8)
    ax.set_xlabel("$Re_\\theta$"); ax.set_ylabel("$y^+$"); ax.set_yscale("log")
    ax.set_title("(a) Isolation Forest Anomaly Scores"); ax.legend(fontsize=8)

    ax = axes[1]
    consensus = df_a[df_a["consensus_anomaly"]]
    iso_only = df_a[df_a["iso_anomaly"] & ~df_a["lof_anomaly"]]
    lof_only = df_a[df_a["lof_anomaly"] & ~df_a["iso_anomaly"]]
    norm = df_a[~df_a["iso_anomaly"] & ~df_a["lof_anomaly"]]

    ax.scatter(norm["re_theta"], norm["y_plus"], c="#2c3e50", s=3, alpha=0.2, label=f"Normal ({len(norm)})")
    ax.scatter(iso_only["re_theta"], iso_only["y_plus"], c="#9b59b6", s=25, marker="D",
               edgecolors="k", linewidths=0.3, label=f"IF only ({len(iso_only)})", zorder=4)
    ax.scatter(lof_only["re_theta"], lof_only["y_plus"], c="#f39c12", s=25, marker="s",
               edgecolors="k", linewidths=0.3, label=f"LOF only ({len(lof_only)})", zorder=4)
    ax.scatter(consensus["re_theta"], consensus["y_plus"], c="#e74c3c", s=60, marker="X",
               edgecolors="k", linewidths=0.5, label=f"Both ({len(consensus)})", zorder=5)
    ax.set_xlabel("$Re_\\theta$"); ax.set_ylabel("$y^+$"); ax.set_yscale("log")
    ax.set_title("(b) Method Comparison"); ax.legend(fontsize=7, loc="upper left")

    plt.suptitle("Anomaly Detection on DNS Profile Data", fontsize=14, y=1.01)
    plt.tight_layout()
    fig.savefig(f"{OUTPUT}/fig13_anomaly_detection.png")
    plt.close(fig)
    print("  [13] Anomaly detection")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ML pipeline for KTH DNS data")
    parser.add_argument("--backend", choices=["postgresql", "sqlite"], default="sqlite")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    config = DatabaseConfig.from_env(backend=args.backend)
    engine = get_engine(config)

    logger.info("=" * 60)
    logger.info("Task 1: Boundary Layer Region Classification")
    logger.info("=" * 60)
    region_results = train_region_classifier(engine)

    logger.info("=" * 60)
    logger.info("Task 2: Re_tau Regression")
    logger.info("=" * 60)
    retau_results = train_retau_predictor(engine)

    logger.info("=" * 60)
    logger.info("Task 3: Anomaly Detection")
    logger.info("=" * 60)
    anomaly_results = detect_anomalies(engine)

    logger.info("=" * 60)
    logger.info("Writing results to database")
    logger.info("=" * 60)
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM model_registry"))
    write_results_to_db(engine, region_results, retau_results, anomaly_results)

    logger.info("=" * 60)
    logger.info("Generating ML figures")
    logger.info("=" * 60)
    generate_ml_figures(region_results, retau_results, anomaly_results)

    logger.info("=" * 60)
    logger.info("Pipeline complete!")
    logger.info(f"  Region classifier accuracy: {region_results['metrics']['accuracy']:.3f}")
    logger.info(f"  Re_tau R²: {retau_results['metrics']['r2']:.3f}")
    logger.info(f"  Consensus anomalies: {anomaly_results['n_anomalies']['consensus']}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
