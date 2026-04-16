-- =============================================================================
-- Turbulent Boundary Layer SQL Analytics Pipeline
-- SQLite schema definition for KTH DNS data (Schlatter & Örlü, 2010)
-- =============================================================================

CREATE TABLE IF NOT EXISTS simulation_conditions (
    condition_id INTEGER PRIMARY KEY AUTOINCREMENT,
    re_theta REAL NOT NULL UNIQUE,
    re_delta_star REAL,
    re_tau REAL,
    shape_factor REAL,
    cf REAL,
    u_tau REAL,
    delta_99 REAL,
    source_file TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS velocity_profiles (
    profile_id INTEGER PRIMARY KEY AUTOINCREMENT,
    condition_id INTEGER NOT NULL REFERENCES simulation_conditions(condition_id),
    y_delta REAL NOT NULL,
    y_plus REAL NOT NULL,
    u_plus REAL NOT NULL,
    u_rms_plus REAL NOT NULL,
    v_rms_plus REAL NOT NULL,
    w_rms_plus REAL NOT NULL,
    uv_plus REAL NOT NULL,
    p_rms_plus REAL,
    pu_plus REAL,
    pv_plus REAL,
    skewness_u REAL,
    flatness_u REAL,
    du_dy_plus REAL,
    v_plus REAL,
    omx_rms_plus REAL,
    omy_rms_plus REAL,
    omz_rms_plus REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (condition_id, y_plus)
);

CREATE INDEX IF NOT EXISTS idx_vp_condition ON velocity_profiles (condition_id);
CREATE INDEX IF NOT EXISTS idx_vp_yplus ON velocity_profiles (y_plus);
CREATE INDEX IF NOT EXISTS idx_vp_ydelta ON velocity_profiles (y_delta);

CREATE TABLE IF NOT EXISTS tke_budgets (
    budget_id INTEGER PRIMARY KEY AUTOINCREMENT,
    condition_id INTEGER NOT NULL REFERENCES simulation_conditions(condition_id),
    y_delta REAL NOT NULL,
    y_plus REAL NOT NULL,
    convection REAL,
    production REAL,
    dissipation REAL,
    turb_diffusion REAL,
    vel_pressure REAL,
    visc_diffusion REAL,
    residual REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (condition_id, y_plus)
);

CREATE INDEX IF NOT EXISTS idx_tke_condition ON tke_budgets (condition_id);
CREATE INDEX IF NOT EXISTS idx_tke_yplus ON tke_budgets (y_plus);

CREATE TABLE IF NOT EXISTS reynolds_stress_budgets (
    rs_budget_id INTEGER PRIMARY KEY AUTOINCREMENT,
    condition_id INTEGER NOT NULL REFERENCES simulation_conditions(condition_id),
    component TEXT NOT NULL CHECK (component IN ('uu', 'vv', 'ww', 'uv')),
    y_delta REAL NOT NULL,
    y_plus REAL NOT NULL,
    convection REAL,
    production REAL,
    dissipation REAL,
    turb_diffusion REAL,
    vel_pressure REAL,
    visc_diffusion REAL,
    residual REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (condition_id, component, y_plus)
);

CREATE INDEX IF NOT EXISTS idx_rs_condition ON reynolds_stress_budgets (condition_id);
CREATE INDEX IF NOT EXISTS idx_rs_component ON reynolds_stress_budgets (component);

CREATE TABLE IF NOT EXISTS derived_features (
    feature_id INTEGER PRIMARY KEY AUTOINCREMENT,
    condition_id INTEGER NOT NULL REFERENCES simulation_conditions(condition_id),
    re_theta REAL,
    tke_peak REAL,
    tke_peak_yplus REAL,
    u_rms_peak REAL,
    u_rms_peak_yplus REAL,
    production_peak REAL,
    dissipation_wall REAL,
    log_layer_slope REAL,
    wake_strength REAL,
    anisotropy_ratio_peak REAL,
    shape_factor REAL,
    cf REAL,
    re_tau REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (condition_id)
);

CREATE TABLE IF NOT EXISTS predictions (
    prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    condition_id INTEGER NOT NULL REFERENCES simulation_conditions(condition_id),
    model_name TEXT NOT NULL,
    model_version TEXT,
    predicted_re_tau REAL,
    predicted_cf REAL,
    cluster_label INTEGER,
    anomaly_score REAL,
    is_anomaly BOOLEAN DEFAULT 0,
    predicted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_pred_model ON predictions (model_name);

CREATE TABLE IF NOT EXISTS model_registry (
    model_id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name TEXT NOT NULL,
    model_version TEXT NOT NULL,
    model_type TEXT NOT NULL,
    hyperparameters TEXT,
    metric_name TEXT,
    metric_value REAL,
    n_training_samples INTEGER,
    training_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    notes TEXT,
    UNIQUE (model_name, model_version)
);
