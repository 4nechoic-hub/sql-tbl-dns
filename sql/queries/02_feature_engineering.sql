-- =============================================================================
-- Advanced Feature Engineering — Window Functions, CTEs, Subqueries
-- =============================================================================
-- Demonstrates: OVER (PARTITION BY ... ORDER BY), LAG, LEAD, ROW_NUMBER,
-- NTILE, WITH (CTE chains), correlated subqueries.
-- =============================================================================

-- Q1: TKE profile computed from Reynolds stresses via SQL
-- TKE+ = 0.5 * (u'_rms^2 + v'_rms^2 + w'_rms^2) in inner units
-- Plus: streamwise gradient using LAG window function
WITH tke_profiles AS (
    SELECT
        sc.re_theta,
        sc.re_tau,
        vp.y_plus,
        vp.y_delta,
        0.5 * (vp.u_rms_plus * vp.u_rms_plus
             + vp.v_rms_plus * vp.v_rms_plus
             + vp.w_rms_plus * vp.w_rms_plus) AS tke_plus,
        vp.u_rms_plus,
        vp.uv_plus
    FROM velocity_profiles vp
    JOIN simulation_conditions sc ON vp.condition_id = sc.condition_id
)
SELECT
    re_theta,
    y_plus,
    tke_plus,
    -- Rank within each Re_theta profile
    ROW_NUMBER() OVER (PARTITION BY re_theta ORDER BY y_plus) AS point_rank,
    -- Running maximum of TKE (tracks the near-wall peak)
    MAX(tke_plus) OVER (
        PARTITION BY re_theta ORDER BY y_plus
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS tke_running_max,
    -- Wall-normal gradient: d(TKE)/d(y+) via finite differences
    (tke_plus - LAG(tke_plus) OVER (PARTITION BY re_theta ORDER BY y_plus))
    / NULLIF(y_plus - LAG(y_plus) OVER (PARTITION BY re_theta ORDER BY y_plus), 0)
    AS dtke_dyplus
FROM tke_profiles
ORDER BY re_theta, y_plus;


-- Q2: Reynolds number scaling of peak turbulence quantities
-- Uses CTE to find peak values, then analyses Re dependence
WITH peak_stats AS (
    SELECT
        sc.condition_id,
        sc.re_theta,
        sc.re_tau,
        sc.cf,
        MAX(vp.u_rms_plus) AS u_rms_peak,
        MAX(0.5 * (vp.u_rms_plus * vp.u_rms_plus
                  + vp.v_rms_plus * vp.v_rms_plus
                  + vp.w_rms_plus * vp.w_rms_plus)) AS tke_peak,
        MAX(ABS(vp.uv_plus)) AS uv_peak
    FROM velocity_profiles vp
    JOIN simulation_conditions sc ON vp.condition_id = sc.condition_id
    GROUP BY sc.condition_id, sc.re_theta, sc.re_tau, sc.cf
),
with_deltas AS (
    SELECT
        *,
        -- Change from previous Re_theta (streamwise evolution)
        u_rms_peak - LAG(u_rms_peak) OVER (ORDER BY re_theta) AS delta_u_rms_peak,
        tke_peak - LAG(tke_peak) OVER (ORDER BY re_theta) AS delta_tke_peak,
        -- Percentage change in cf
        100.0 * (cf - LAG(cf) OVER (ORDER BY re_theta))
            / NULLIF(LAG(cf) OVER (ORDER BY re_theta), 0) AS cf_pct_change
    FROM peak_stats
)
SELECT * FROM with_deltas ORDER BY re_theta;


-- Q3: Profile shape classification using NTILE
-- Divide the wall-normal extent into named regions
SELECT
    sc.re_theta,
    vp.y_plus,
    vp.u_plus,
    vp.u_rms_plus,
    CASE
        WHEN vp.y_plus < 5 THEN 'viscous_sublayer'
        WHEN vp.y_plus < 30 THEN 'buffer_layer'
        WHEN vp.y_plus < 0.3 * sc.re_tau THEN 'log_layer'
        WHEN vp.y_delta < 1.0 THEN 'wake_region'
        ELSE 'freestream'
    END AS bl_region,
    NTILE(10) OVER (PARTITION BY sc.re_theta ORDER BY vp.y_plus) AS decile
FROM velocity_profiles vp
JOIN simulation_conditions sc ON vp.condition_id = sc.condition_id
ORDER BY sc.re_theta, vp.y_plus;


-- Q4: TKE budget balance check — do terms sum to zero?
-- Production + Dissipation + Transport + Diffusion ≈ 0
WITH budget_check AS (
    SELECT
        sc.re_theta,
        tb.y_plus,
        tb.production,
        tb.dissipation,
        tb.turb_diffusion,
        tb.vel_pressure,
        tb.visc_diffusion,
        tb.convection,
        tb.residual,
        (tb.production + tb.dissipation + tb.turb_diffusion
         + tb.vel_pressure + tb.visc_diffusion + tb.convection) AS budget_sum
    FROM tke_budgets tb
    JOIN simulation_conditions sc ON tb.condition_id = sc.condition_id
)
SELECT
    re_theta,
    MAX(ABS(budget_sum)) AS max_imbalance,
    AVG(ABS(budget_sum)) AS avg_imbalance,
    AVG(ABS(residual)) AS avg_residual,
    COUNT(*) AS n_points,
    CASE
        WHEN MAX(ABS(budget_sum)) < 0.01 THEN 'BALANCED'
        ELSE 'CHECK_REQUIRED'
    END AS status
FROM budget_check
GROUP BY re_theta
ORDER BY re_theta;


-- Q5: Cross-Re comparison at matched y+ locations
-- Compare velocity profiles at y+ = 15 (buffer layer peak) across all Re
-- Uses window function to find closest point to target y+
WITH ranked AS (
    SELECT
        sc.re_theta,
        sc.re_tau,
        vp.y_plus,
        vp.u_plus,
        vp.u_rms_plus,
        vp.v_rms_plus,
        vp.uv_plus,
        ROW_NUMBER() OVER (
            PARTITION BY sc.re_theta
            ORDER BY ABS(vp.y_plus - 15.0)
        ) AS rank_closest
    FROM velocity_profiles vp
    JOIN simulation_conditions sc ON vp.condition_id = sc.condition_id
)
SELECT
    re_theta,
    re_tau,
    y_plus,
    u_plus,
    u_rms_plus,
    v_rms_plus,
    uv_plus
FROM ranked
WHERE rank_closest = 1
ORDER BY re_theta;


-- Q6: Comprehensive ML feature vector per Re_theta via CTE chain
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
        AVG(vp.skewness_u) AS avg_skewness,
        AVG(vp.flatness_u) AS avg_flatness,
        MAX(vp.p_rms_plus) AS p_rms_peak
    FROM velocity_profiles vp
    JOIN simulation_conditions sc ON vp.condition_id = sc.condition_id
    GROUP BY sc.condition_id, sc.re_theta, sc.re_tau, sc.cf, sc.shape_factor
),
budget_stats AS (
    SELECT
        tb.condition_id,
        MAX(tb.production) AS production_peak,
        MIN(tb.dissipation) AS dissipation_min,
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
    bs.avg_skewness,
    bs.avg_flatness,
    CASE WHEN bs.u_rms_peak > 0 THEN bs.v_rms_peak / bs.u_rms_peak ELSE NULL END AS anisotropy_ratio,
    bud.production_peak,
    bud.dissipation_min,
    bud.visc_diff_wall,
    pl.u_rms_peak_yplus,
    pl.tke_peak_yplus
FROM base_stats bs
LEFT JOIN budget_stats bud ON bs.condition_id = bud.condition_id
LEFT JOIN peak_locations pl ON bs.condition_id = pl.condition_id
ORDER BY bs.re_theta;
