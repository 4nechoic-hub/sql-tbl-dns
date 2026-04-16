"""
ML pipeline for KTH DNS boundary layer data.

Three ML tasks on real DNS data:
1. Boundary layer region classification (viscous sublayer / buffer / log / wake)
2. Re_tau prediction from profile features (regression)
3. Anomaly detection — identify unusual profile points across Re_theta

Validation is leakage-aware:
- supervised models are wrapped in scikit-learn Pipelines
- evaluation holds out whole Reynolds-number conditions via LeaveOneGroupOut
- scaling is fitted only on each training fold, never on the full dataset
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

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
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sqlalchemy import text

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import DatabaseConfig
from src.db import get_engine
from src.feature_engineering import compute_derived_features, get_profile_features_per_point

logger = logging.getLogger(__name__)
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Model builders and evaluation helpers
# ---------------------------------------------------------------------------


def build_region_classifier(random_state: int = RANDOM_STATE) -> Pipeline:
    """Create a leakage-safe classifier pipeline for BL-region prediction."""
    return Pipeline([
        ("scaler", StandardScaler()),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=300,
                max_depth=15,
                min_samples_split=5,
                random_state=random_state,
                class_weight="balanced",
            ),
        ),
    ])



def build_retau_regressor(random_state: int = RANDOM_STATE) -> Pipeline:
    """Create a leakage-safe regression pipeline for Re_tau prediction."""
    return Pipeline([
        ("scaler", StandardScaler()),
        (
            "regressor",
            GradientBoostingRegressor(
                n_estimators=50,
                max_depth=3,
                random_state=random_state,
            ),
        ),
    ])



def build_isolation_forest_pipeline(random_state: int = RANDOM_STATE) -> Pipeline:
    """Create an anomaly-detection pipeline with explicit scaling."""
    return Pipeline([
        ("scaler", StandardScaler()),
        (
            "detector",
            IsolationForest(
                n_estimators=200,
                contamination=0.03,
                random_state=random_state,
            ),
        ),
    ])



def build_lof_pipeline() -> Pipeline:
    """Create a LOF anomaly-detection pipeline with explicit scaling."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("detector", LocalOutlierFactor(n_neighbors=30, contamination=0.03)),
    ])



def _group_summary_from_predictions(
    evaluation_df: pd.DataFrame,
    group_cols: list[str],
    task: str,
) -> pd.DataFrame:
    """Summarize out-of-fold predictions at the held-out condition level."""
    rows: list[dict] = []
    for group_values, group_df in evaluation_df.groupby(group_cols, sort=True):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        row = dict(zip(group_cols, group_values))
        row["n_samples"] = int(len(group_df))
        if task == "classification":
            row["accuracy"] = float(accuracy_score(group_df["y_true"], group_df["y_pred"]))
            row["f1_weighted"] = float(
                f1_score(group_df["y_true"], group_df["y_pred"], average="weighted")
            )
        elif task == "regression":
            errors = group_df["y_pred"] - group_df["y_true"]
            row["predicted_re_tau"] = float(group_df["y_pred"].iloc[0])
            row["true_re_tau"] = float(group_df["y_true"].iloc[0])
            row["signed_error"] = float(errors.iloc[0])
            row["abs_error"] = float(np.abs(errors.iloc[0]))
        else:
            raise ValueError(f"Unknown task type: {task}")
        rows.append(row)

    sort_cols = [col for col in ["re_theta", "condition_id"] if col in group_cols]
    if not sort_cols:
        sort_cols = group_cols
    return pd.DataFrame(rows).sort_values(sort_cols).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Task 1: BL Region Classification
# ---------------------------------------------------------------------------


def train_region_classifier(engine) -> dict:
    """
    Classify boundary-layer regions from DNS profile features.

    Evaluation is leave-one-condition-out: the model never sees points from the
    same Reynolds-number case in both train and validation folds.
    """
    logger.info("Extracting per-point features via SQL...")
    df = get_profile_features_per_point(engine)

    feature_cols = [
        "u_plus",
        "u_rms_plus",
        "v_rms_plus",
        "w_rms_plus",
        "uv_plus",
        "p_rms_plus",
        "skewness_u",
        "flatness_u",
        "du_dy_plus",
        "tke_plus",
    ]

    # Exclude freestream (sparse) for cleaner classification
    df = df[df["bl_region"] != "freestream"].copy()
    df = df.dropna(subset=feature_cols)

    if df.empty:
        raise ValueError("No classification samples available after filtering and NA removal.")

    group_col = "condition_id" if "condition_id" in df.columns else "re_theta"
    n_groups = df[group_col].nunique()
    if n_groups < 2:
        raise ValueError("Need at least two distinct conditions for group-aware validation.")

    X = df[feature_cols]
    y = df["bl_region"].to_numpy()
    groups = df[group_col].to_numpy()

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    pipeline = build_region_classifier()
    logo = LeaveOneGroupOut()
    y_pred_oof = cross_val_predict(
        pipeline,
        X,
        y_enc,
        groups=groups,
        cv=logo,
        method="predict",
    )

    acc = accuracy_score(y_enc, y_pred_oof)
    f1 = f1_score(y_enc, y_pred_oof, average="weighted")

    report = classification_report(
        y_enc,
        y_pred_oof,
        target_names=le.classes_,
        zero_division=0,
    )

    evaluation_df = pd.DataFrame(
        {
            "condition_id": df["condition_id"].to_numpy() if "condition_id" in df.columns else groups,
            "re_theta": df["re_theta"].to_numpy(),
            "y_true": y_enc,
            "y_pred": y_pred_oof,
        }
    )
    per_condition_metrics = _group_summary_from_predictions(
        evaluation_df,
        ["condition_id", "re_theta"],
        task="classification",
    )

    logger.info(
        "Region classifier (leave-one-condition-out) — Accuracy: %.3f, F1: %.3f",
        acc,
        f1,
    )
    logger.info("\n%s", report)
    logger.info(
        "Per-condition holdout metrics:\n%s",
        per_condition_metrics.to_string(
            index=False,
            float_format=lambda value: f"{value:.3f}",
        ),
    )

    pipeline.fit(X, y_enc)
    final_model = pipeline.named_steps["classifier"]
    importances = pd.Series(final_model.feature_importances_, index=feature_cols)
    logger.info("Top features:\n%s", importances.sort_values(ascending=False).head(5).to_string())

    return {
        "pipeline": pipeline,
        "model": final_model,
        "scaler": pipeline.named_steps["scaler"],
        "label_encoder": le,
        "feature_cols": feature_cols,
        "metrics": {
            "accuracy": float(acc),
            "f1_score": float(f1),
            "cv_mean": float(per_condition_metrics["accuracy"].mean()),
            "cv_std": float(per_condition_metrics["accuracy"].std(ddof=0)),
            "cv_strategy": "LeaveOneGroupOut by Reynolds-number condition",
        },
        "confusion_matrix": confusion_matrix(y_enc, y_pred_oof),
        "y_true": y_enc,
        "y_pred": y_pred_oof,
        "importances": importances,
        "df": df,
        "y_encoded": y_enc,
        "per_condition_metrics": per_condition_metrics,
    }


# ---------------------------------------------------------------------------
# Task 2: Re_tau Regression from Profile Features
# ---------------------------------------------------------------------------


def train_retau_predictor(engine) -> dict:
    """
    Predict Re_tau from SQL-derived profile features.

    Each feature vector corresponds to one Reynolds-number condition, so
    leave-one-group-out becomes leave-one-condition-out evaluation.
    """
    logger.info("Extracting derived features via SQL CTE pipeline...")
    features = compute_derived_features(engine)

    feature_cols = [
        "u_rms_peak",
        "v_rms_peak",
        "w_rms_peak",
        "uv_peak",
        "tke_peak",
        "p_rms_peak",
        "anisotropy_ratio",
        "production_peak",
        "dissipation_wall",
        "shape_factor",
        "u_rms_peak_yplus",
    ]

    df = features.dropna(subset=feature_cols + ["re_tau", "condition_id"]).copy()
    if df.empty:
        raise ValueError("No regression samples available after NA removal.")

    X = df[feature_cols]
    y = df["re_tau"].to_numpy(dtype=float)
    groups = df["condition_id"].to_numpy()

    pipeline = build_retau_regressor()
    logo = LeaveOneGroupOut()
    y_pred_logo = cross_val_predict(
        pipeline,
        X,
        y,
        groups=groups,
        cv=logo,
        method="predict",
    )
    y_pred_logo = np.asarray(y_pred_logo, dtype=float)

    r2 = r2_score(y, y_pred_logo)
    mae = mean_absolute_error(y, y_pred_logo)
    rmse = float(np.sqrt(mean_squared_error(y, y_pred_logo)))

    per_condition_metrics = pd.DataFrame(
        {
            "condition_id": df["condition_id"].to_numpy(dtype=int),
            "re_theta": df["re_theta"].to_numpy(dtype=float),
            "y_true": y,
            "y_pred": y_pred_logo,
        }
    )
    per_condition_metrics = _group_summary_from_predictions(
        per_condition_metrics,
        ["condition_id", "re_theta"],
        task="regression",
    )

    logger.info(
        "Re_tau predictor (leave-one-condition-out) — R²: %.3f, MAE: %.1f, RMSE: %.1f",
        r2,
        mae,
        rmse,
    )
    logger.info(
        "Per-condition holdout errors:\n%s",
        per_condition_metrics.to_string(
            index=False,
            float_format=lambda value: f"{value:.3f}",
        ),
    )

    pipeline.fit(X, y)
    final_model = pipeline.named_steps["regressor"]
    importances = pd.Series(final_model.feature_importances_, index=feature_cols)
    logger.info("Top features:\n%s", importances.sort_values(ascending=False).head(5).to_string())

    return {
        "pipeline": pipeline,
        "model": final_model,
        "scaler": pipeline.named_steps["scaler"],
        "feature_cols": feature_cols,
        "metrics": {
            "r2": float(r2),
            "mae": float(mae),
            "rmse": float(rmse),
            "cv_strategy": "LeaveOneGroupOut by Reynolds-number condition",
        },
        "y_true": y,
        "y_pred": y_pred_logo,
        "re_theta": df["re_theta"].to_numpy(dtype=float),
        "condition_id": df["condition_id"].to_numpy(dtype=int),
        "importances": importances,
        "per_condition_metrics": per_condition_metrics,
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
        "u_rms_plus",
        "v_rms_plus",
        "w_rms_plus",
        "uv_plus",
        "tke_plus",
        "skewness_u",
        "flatness_u",
    ]

    df_clean = df.dropna(subset=feature_cols).copy()
    if df_clean.empty:
        raise ValueError("No anomaly-detection samples available after NA removal.")

    X = df_clean[feature_cols]

    iso_pipeline = build_isolation_forest_pipeline()
    lof_pipeline = build_lof_pipeline()

    iso_labels = iso_pipeline.fit_predict(X)
    iso_scores = iso_pipeline.decision_function(X)
    lof_labels = lof_pipeline.fit_predict(X)

    consensus = (iso_labels == -1) & (lof_labels == -1)

    n_iso = int((iso_labels == -1).sum())
    n_lof = int((lof_labels == -1).sum())
    n_both = int(consensus.sum())

    logger.info("Isolation Forest: %s/%s anomalies", n_iso, len(df_clean))
    logger.info("LOF: %s/%s anomalies", n_lof, len(df_clean))
    logger.info("Consensus: %s anomalies", n_both)

    df_clean["iso_anomaly"] = iso_labels == -1
    df_clean["lof_anomaly"] = lof_labels == -1
    df_clean["consensus_anomaly"] = consensus
    df_clean["anomaly_score"] = iso_scores

    return {
        "df": df_clean,
        "n_anomalies": {"iso": n_iso, "lof": n_lof, "consensus": n_both},
        "feature_cols": feature_cols,
        "pipelines": {
            "isolation_forest": iso_pipeline,
            "local_outlier_factor": lof_pipeline,
        },
    }


# ---------------------------------------------------------------------------
# Write predictions to database
# ---------------------------------------------------------------------------


def write_results_to_db(engine, region_results, retau_results, anomaly_results):
    """Write all ML results back to database with ACID transactions."""
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM predictions"))

        # Region classification: write summary only (per-point output is large)
        metrics = region_results["metrics"]
        conn.execute(text("""
            INSERT INTO model_registry (model_name, model_version, model_type,
                hyperparameters, metric_name, metric_value, n_training_samples, notes)
            VALUES (:name, :ver, :type, :params, :mname, :mval, :n, :notes)
        """), {
            "name": "region_classifier",
            "ver": "v1.0",
            "type": "classifier",
            "params": json.dumps({"n_estimators": 300, "max_depth": 15, "group_cv": "LeaveOneGroupOut"}),
            "mname": "accuracy",
            "mval": metrics["accuracy"],
            "n": len(region_results["y_encoded"]),
            "notes": (
                "Surrogate BL-region classification with holdout by Reynolds-number condition; "
                f"mean condition accuracy={metrics['cv_mean']:.3f}"
            ),
        })

        # Re_tau regression predictions: one row per condition
        for condition_id, pred in zip(retau_results["condition_id"], retau_results["y_pred"]):
            conn.execute(text("""
                INSERT INTO predictions (condition_id, model_name, model_version,
                    predicted_re_tau)
                VALUES (:cid, 'retau_regressor', 'v1.0', :pred)
            """), {
                "cid": int(condition_id),
                "pred": float(pred),
            })

        # Regression model registry
        conn.execute(text("""
            INSERT INTO model_registry (model_name, model_version, model_type,
                metric_name, metric_value, n_training_samples, notes)
            VALUES (:name, :ver, :type, :mname, :mval, :n, :notes)
        """), {
            "name": "retau_regressor",
            "ver": "v1.0",
            "type": "regressor",
            "mname": "r2",
            "mval": retau_results["metrics"]["r2"],
            "n": len(retau_results["y_true"]),
            "notes": (
                "Leave-one-condition-out evaluation on condition-level SQL features; "
                f"MAE={retau_results['metrics']['mae']:.1f}"
            ),
        })

    logger.info("Results written to database")


# ---------------------------------------------------------------------------
# Generate ML figures
# ---------------------------------------------------------------------------


def default_ml_output_dir(backend: str) -> Path:
    """Return the default ML figure output directory for a backend."""
    return Path("data/processed_postgres" if backend == "postgresql" else "data/processed")



def generate_ml_figures(region_results, retau_results, anomaly_results, output_dir: str):
    """Generate ML-specific figures."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 13,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.25,
    })
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Fig 11: Region classification confusion matrix + importance
    fig = plt.figure(figsize=(14, 5.5))
    gs = plt.GridSpec(1, 2, width_ratios=[1, 1.2], wspace=0.35)

    ax1 = fig.add_subplot(gs[0])
    le = region_results["label_encoder"]
    cm = region_results["confusion_matrix"]
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    disp.plot(ax=ax1, cmap="Blues", values_format="d", colorbar=False)
    ax1.set_title(
        f"(a) Region Classification — Group-CV Acc: {region_results['metrics']['accuracy']:.1%}",
        fontsize=12,
    )
    ax1.tick_params(axis="x", rotation=25)

    ax2 = fig.add_subplot(gs[1])
    imp = region_results["importances"].sort_values()
    ax2.barh(range(len(imp)), imp.values, color="#3498db", edgecolor="0.3", height=0.7)
    ax2.set_yticks(range(len(imp)))
    ax2.set_yticklabels(imp.index, fontsize=9)
    ax2.set_xlabel("Feature Importance (Gini)")
    ax2.set_title("(b) Feature Importance", fontsize=12)
    for i in range(-1, -4, -1):
        ax2.text(
            imp.values[i] + 0.005,
            len(imp) + i,
            f"{imp.values[i]:.3f}",
            va="center",
            fontsize=8,
            color="0.3",
        )

    plt.tight_layout()
    fig.savefig(output_path / "fig11_region_classification.png")
    plt.close(fig)
    print("  [11] Region classification")

    # Fig 12: Re_tau prediction — true vs predicted
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    ax = axes[0]
    ax.plot(retau_results["y_true"], retau_results["y_pred"], "ko", ms=8, zorder=3)
    lims = [200, 1400]
    ax.plot(lims, lims, "r--", lw=1.2, alpha=0.6, label="Perfect prediction")
    for i in range(len(retau_results["y_true"])):
        ax.annotate(
            f"  {retau_results['re_theta'][i]:.0f}",
            (retau_results["y_true"][i], retau_results["y_pred"][i]),
            fontsize=7,
            color="0.4",
        )
    ax.set_xlabel("True $Re_\\tau$")
    ax.set_ylabel("Predicted $Re_\\tau$")
    r2 = retau_results["metrics"]["r2"]
    ax.set_title(f"(a) $Re_\\tau$ Prediction (Leave-one-condition-out, $R^2={r2:.3f}$)")
    ax.legend(fontsize=9)
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    ax = axes[1]
    imp_r = retau_results["importances"].sort_values()
    ax.barh(range(len(imp_r)), imp_r.values, color="#e67e22", edgecolor="0.3", height=0.7)
    ax.set_yticks(range(len(imp_r)))
    ax.set_yticklabels(imp_r.index, fontsize=9)
    ax.set_xlabel("Feature Importance")
    ax.set_title("(b) GBR Feature Importance")

    plt.suptitle("$Re_\\tau$ Regression from SQL-Engineered Features", fontsize=14, y=1.01)
    plt.tight_layout()
    fig.savefig(output_path / "fig12_retau_prediction.png")
    plt.close(fig)
    print("  [12] Re_tau prediction")

    # Fig 13: Anomaly detection map
    df_a = anomaly_results["df"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    ax = axes[0]
    normal = df_a[~df_a["iso_anomaly"]]
    anom = df_a[df_a["iso_anomaly"]]
    sc = ax.scatter(
        normal["re_theta"],
        normal["y_plus"],
        c=normal["anomaly_score"],
        cmap="RdYlGn",
        s=5,
        alpha=0.4,
    )
    ax.scatter(
        anom["re_theta"],
        anom["y_plus"],
        c="red",
        s=30,
        marker="x",
        linewidths=1,
        zorder=5,
        label=f"IF anomaly ({len(anom)})",
    )
    plt.colorbar(sc, ax=ax, label="Anomaly Score", shrink=0.8)
    ax.set_xlabel("$Re_\\theta$")
    ax.set_ylabel("$y^+$")
    ax.set_yscale("log")
    ax.set_title("(a) Isolation Forest Anomaly Scores")
    ax.legend(fontsize=8)

    ax = axes[1]
    consensus = df_a[df_a["consensus_anomaly"]]
    iso_only = df_a[df_a["iso_anomaly"] & ~df_a["lof_anomaly"]]
    lof_only = df_a[df_a["lof_anomaly"] & ~df_a["iso_anomaly"]]
    norm = df_a[~df_a["iso_anomaly"] & ~df_a["lof_anomaly"]]

    ax.scatter(
        norm["re_theta"],
        norm["y_plus"],
        c="#2c3e50",
        s=3,
        alpha=0.2,
        label=f"Normal ({len(norm)})",
    )
    ax.scatter(
        iso_only["re_theta"],
        iso_only["y_plus"],
        c="#9b59b6",
        s=25,
        marker="D",
        edgecolors="k",
        linewidths=0.3,
        label=f"IF only ({len(iso_only)})",
        zorder=4,
    )
    ax.scatter(
        lof_only["re_theta"],
        lof_only["y_plus"],
        c="#f39c12",
        s=25,
        marker="s",
        edgecolors="k",
        linewidths=0.3,
        label=f"LOF only ({len(lof_only)})",
        zorder=4,
    )
    ax.scatter(
        consensus["re_theta"],
        consensus["y_plus"],
        c="#e74c3c",
        s=60,
        marker="X",
        edgecolors="k",
        linewidths=0.5,
        label=f"Both ({len(consensus)})",
        zorder=5,
    )
    ax.set_xlabel("$Re_\\theta$")
    ax.set_ylabel("$y^+$")
    ax.set_yscale("log")
    ax.set_title("(b) Method Comparison")
    ax.legend(fontsize=7, loc="upper left")

    plt.suptitle("Anomaly Detection on DNS Profile Data", fontsize=14, y=1.01)
    plt.tight_layout()
    fig.savefig(output_path / "fig13_anomaly_detection.png")
    plt.close(fig)
    print("  [13] Anomaly detection")
    logger.info("ML figures saved to %s", output_path.resolve())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="ML pipeline for KTH DNS data")
    parser.add_argument("--backend", choices=["postgresql", "sqlite"], default="sqlite")
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Directory for ML figures. Defaults to data/processed for sqlite "
            "and data/processed_postgres for postgresql."
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    config = DatabaseConfig.from_env(backend=args.backend)
    engine = get_engine(config)
    output_dir = Path(args.output_dir) if args.output_dir else default_ml_output_dir(args.backend)

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
    generate_ml_figures(region_results, retau_results, anomaly_results, output_dir=str(output_dir))

    logger.info("=" * 60)
    logger.info("Pipeline complete!")
    logger.info("  Region classifier accuracy: %.3f", region_results["metrics"]["accuracy"])
    logger.info("  Re_tau R²: %.3f", retau_results["metrics"]["r2"])
    logger.info("  Consensus anomalies: %s", anomaly_results["n_anomalies"]["consensus"])
    logger.info("  ML figures output: %s", output_dir.resolve())
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
