"""
SQL-based feature engineering for KTH DNS boundary layer data.

Extracts ML features entirely via SQL queries — Python only orchestrates
query execution and receives clean DataFrames.
"""

import logging

import pandas as pd
from sqlalchemy import text

logger = logging.getLogger(__name__)


def compute_derived_features(engine) -> pd.DataFrame:
    """
    Extract a comprehensive ML feature vector per Re_theta using SQL.
    All heavy computation (peak finding, aggregation, derived quantities)
    happens in the database.
    """
    sql = """
    WITH base_stats AS (
        SELECT
            sc.condition_id,
            sc.re_theta,
            sc.re_tau,
            sc.cf,
            sc.shape_factor,
            MAX(vp.u_rms_plus) AS u_rms_peak,
            MAX(vp.v_rms_plus) AS v_rms_peak,
            MAX(vp.w_rms_plus) AS w_rms_peak,
            MAX(ABS(vp.uv_plus)) AS uv_peak,
            MAX(0.5 * (vp.u_rms_plus*vp.u_rms_plus + vp.v_rms_plus*vp.v_rms_plus
                      + vp.w_rms_plus*vp.w_rms_plus)) AS tke_peak,
            MAX(vp.p_rms_plus) AS p_rms_peak,
            AVG(CASE WHEN vp.y_plus BETWEEN 5 AND 30 THEN vp.skewness_u ELSE NULL END) AS buffer_skewness,
            AVG(CASE WHEN vp.y_plus BETWEEN 5 AND 30 THEN vp.flatness_u ELSE NULL END) AS buffer_flatness,
            AVG(CASE WHEN vp.y_plus BETWEEN 30 AND 200 THEN vp.skewness_u ELSE NULL END) AS log_skewness,
            AVG(CASE WHEN vp.y_plus BETWEEN 30 AND 200 THEN vp.flatness_u ELSE NULL END) AS log_flatness,
            MAX(vp.omz_rms_plus) AS omz_peak
        FROM velocity_profiles vp
        JOIN simulation_conditions sc ON vp.condition_id = sc.condition_id
        GROUP BY sc.condition_id, sc.re_theta, sc.re_tau, sc.cf, sc.shape_factor
    ),
    budget_stats AS (
        SELECT
            tb.condition_id,
            MAX(tb.production) AS production_peak,
            MIN(tb.dissipation) AS dissipation_wall,
            MAX(tb.visc_diffusion) AS visc_diff_wall
        FROM tke_budgets tb
        WHERE tb.y_plus < 50
        GROUP BY tb.condition_id
    ),
    peak_locations AS (
        SELECT
            sc.condition_id,
            (SELECT vp2.y_plus FROM velocity_profiles vp2
             WHERE vp2.condition_id = sc.condition_id
             ORDER BY vp2.u_rms_plus DESC LIMIT 1) AS u_rms_peak_yplus,
            (SELECT vp3.y_plus FROM velocity_profiles vp3
             WHERE vp3.condition_id = sc.condition_id
             ORDER BY (vp3.u_rms_plus*vp3.u_rms_plus + vp3.v_rms_plus*vp3.v_rms_plus
                      + vp3.w_rms_plus*vp3.w_rms_plus) DESC LIMIT 1) AS tke_peak_yplus
        FROM simulation_conditions sc
    )
    SELECT
        bs.condition_id,
        bs.re_theta,
        bs.re_tau,
        bs.cf,
        bs.shape_factor,
        bs.u_rms_peak,
        bs.v_rms_peak,
        bs.w_rms_peak,
        bs.uv_peak,
        bs.tke_peak,
        bs.p_rms_peak,
        bs.buffer_skewness,
        bs.buffer_flatness,
        bs.log_skewness,
        bs.log_flatness,
        bs.omz_peak,
        CASE WHEN bs.u_rms_peak > 0 THEN bs.v_rms_peak / bs.u_rms_peak ELSE NULL END AS anisotropy_ratio,
        bud.production_peak,
        bud.dissipation_wall,
        bud.visc_diff_wall,
        pl.u_rms_peak_yplus,
        pl.tke_peak_yplus
    FROM base_stats bs
    LEFT JOIN budget_stats bud ON bs.condition_id = bud.condition_id
    LEFT JOIN peak_locations pl ON bs.condition_id = pl.condition_id
    ORDER BY bs.re_theta
    """
    df = pd.read_sql(sql, engine)
    logger.info("Extracted %s feature vectors with %s columns", len(df), len(df.columns))

    # Also store in derived_features table
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM derived_features"))
    df_store = df[[
        "condition_id",
        "re_theta",
        "tke_peak",
        "tke_peak_yplus",
        "u_rms_peak",
        "u_rms_peak_yplus",
        "production_peak",
        "dissipation_wall",
        "shape_factor",
        "cf",
        "re_tau",
    ]].copy()
    df_store["anisotropy_ratio_peak"] = df["anisotropy_ratio"]
    df_store.to_sql("derived_features", engine, if_exists="append", index=False)

    return df



def get_profile_features_per_point(engine) -> pd.DataFrame:
    """
    Extract per-point features for boundary layer region classification.

    Each row is one (condition, y+) point with features and a surrogate
    boundary-layer region label. The stable ``condition_id`` column is
    included so ML evaluation can hold out entire Reynolds-number cases.
    """
    sql = """
    SELECT
        sc.condition_id,
        sc.re_theta,
        sc.re_tau,
        vp.y_plus,
        vp.y_delta,
        vp.u_plus,
        vp.u_rms_plus,
        vp.v_rms_plus,
        vp.w_rms_plus,
        vp.uv_plus,
        vp.p_rms_plus,
        vp.skewness_u,
        vp.flatness_u,
        vp.du_dy_plus,
        0.5 * (vp.u_rms_plus*vp.u_rms_plus + vp.v_rms_plus*vp.v_rms_plus
              + vp.w_rms_plus*vp.w_rms_plus) AS tke_plus,
        CASE
            WHEN vp.y_plus < 5 THEN 'viscous_sublayer'
            WHEN vp.y_plus < 30 THEN 'buffer_layer'
            WHEN vp.y_plus < 0.3 * sc.re_tau THEN 'log_layer'
            WHEN vp.y_delta < 1.0 THEN 'wake_region'
            ELSE 'freestream'
        END AS bl_region
    FROM velocity_profiles vp
    JOIN simulation_conditions sc ON vp.condition_id = sc.condition_id
    WHERE vp.y_plus > 0.1
    ORDER BY sc.re_theta, vp.y_plus
    """
    return pd.read_sql(sql, engine)
