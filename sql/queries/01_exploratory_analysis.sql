-- =============================================================================
-- Exploratory Data Analysis — KTH DNS Boundary Layer Data
-- =============================================================================
-- SQL concepts: SELECT, WHERE, JOIN, GROUP BY, ORDER BY, COUNT, DISTINCT,
-- aggregate functions, CASE expressions.
-- =============================================================================

-- Q1: Dataset overview
SELECT
    COUNT(DISTINCT sc.re_theta) AS n_reynolds_numbers,
    COUNT(*) AS total_profile_points,
    MIN(sc.re_theta) AS re_theta_min,
    MAX(sc.re_theta) AS re_theta_max,
    MIN(sc.re_tau) AS re_tau_min,
    MAX(sc.re_tau) AS re_tau_max
FROM velocity_profiles vp
JOIN simulation_conditions sc ON vp.condition_id = sc.condition_id;

-- Q2: Integral boundary layer parameters across Re_theta
-- Shows how shape factor and skin friction evolve with Reynolds number
SELECT
    re_theta,
    re_tau,
    shape_factor AS H12,
    cf,
    re_delta_star,
    CASE
        WHEN re_theta < 1000 THEN 'low-Re'
        WHEN re_theta < 2500 THEN 'moderate-Re'
        ELSE 'high-Re'
    END AS re_category
FROM simulation_conditions
ORDER BY re_theta;

-- Q3: Data density per simulation — count wall-normal points
SELECT
    sc.re_theta,
    sc.re_tau,
    COUNT(vp.profile_id) AS n_vel_points,
    (SELECT COUNT(*) FROM tke_budgets tb WHERE tb.condition_id = sc.condition_id) AS n_tke_points,
    (SELECT COUNT(*) FROM reynolds_stress_budgets rs WHERE rs.condition_id = sc.condition_id) AS n_rs_points,
    MIN(vp.y_plus) AS y_plus_min,
    MAX(vp.y_plus) AS y_plus_max
FROM simulation_conditions sc
LEFT JOIN velocity_profiles vp ON sc.condition_id = vp.condition_id
GROUP BY sc.condition_id, sc.re_theta, sc.re_tau
ORDER BY sc.re_theta;

-- Q4: Near-wall peak of streamwise Reynolds stress
-- Demonstrates GROUP BY with aggregate + JOIN
SELECT
    sc.re_theta,
    sc.re_tau,
    MAX(vp.u_rms_plus) AS u_rms_peak,
    -- This subquery finds the y+ where the peak occurs
    (SELECT vp2.y_plus FROM velocity_profiles vp2
     WHERE vp2.condition_id = sc.condition_id
     ORDER BY vp2.u_rms_plus DESC LIMIT 1) AS peak_y_plus
FROM velocity_profiles vp
JOIN simulation_conditions sc ON vp.condition_id = sc.condition_id
GROUP BY sc.condition_id, sc.re_theta, sc.re_tau
ORDER BY sc.re_theta;

-- Q5: Identify log-layer region (30 < y+ < 0.3*Re_tau)
-- Shows velocity deficit from log-law: U+ = (1/kappa)*ln(y+) + B
SELECT
    sc.re_theta,
    vp.y_plus,
    vp.u_plus,
    (1.0/0.41) * LOG(vp.y_plus) + 5.2 AS u_plus_log_law,
    vp.u_plus - ((1.0/0.41) * LOG(vp.y_plus) + 5.2) AS deficit_from_log_law
FROM velocity_profiles vp
JOIN simulation_conditions sc ON vp.condition_id = sc.condition_id
WHERE vp.y_plus > 30 AND vp.y_plus < 0.3 * sc.re_tau
ORDER BY sc.re_theta, vp.y_plus;

-- Q6: Reynolds stress anisotropy — ratio of v'_rms to u'_rms
SELECT
    sc.re_theta,
    vp.y_plus,
    vp.u_rms_plus,
    vp.v_rms_plus,
    vp.w_rms_plus,
    CASE
        WHEN vp.u_rms_plus > 0.01 THEN vp.v_rms_plus / vp.u_rms_plus
        ELSE NULL
    END AS anisotropy_v_u,
    CASE
        WHEN vp.u_rms_plus > 0.01 THEN vp.w_rms_plus / vp.u_rms_plus
        ELSE NULL
    END AS anisotropy_w_u
FROM velocity_profiles vp
JOIN simulation_conditions sc ON vp.condition_id = sc.condition_id
WHERE vp.y_plus BETWEEN 10 AND 200
ORDER BY sc.re_theta, vp.y_plus;
