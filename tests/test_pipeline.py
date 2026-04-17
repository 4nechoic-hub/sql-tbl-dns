"""
Tests for the TBL SQL Analytics Pipeline with KTH DNS data.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import text

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import DatabaseConfig
from src.db import get_engine, init_schema
from src.feature_engineering import (
    compute_derived_features,
    get_profile_features_per_point,
)
from src.ingest_kth import (
    ingest_kth_data,
    parse_budget_profile,
    parse_header,
    parse_velocity_profile,
)


@pytest.fixture(scope="module")
def engine_with_data():
    """Create test database with real KTH data."""
    config = DatabaseConfig(backend="sqlite", sqlite_path=":memory:")
    engine = get_engine(config)
    init_schema(engine)

    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw", "kth_dns")
    if os.path.exists(data_dir):
        ingest_kth_data(engine, data_dir)
    else:
        pytest.skip("KTH data not available")

    return engine


class TestParsing:
    def test_parse_header(self):
        data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw", "kth_dns")
        if not os.path.exists(data_dir):
            pytest.skip("KTH data not available")
        header = parse_header(os.path.join(data_dir, "vel_0670_dns.prof"))
        assert "re_theta" in header
        assert abs(header["re_theta"] - 677.452) < 1.0

    def test_parse_velocity_profile(self):
        data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw", "kth_dns")
        if not os.path.exists(data_dir):
            pytest.skip("KTH data not available")
        df = parse_velocity_profile(os.path.join(data_dir, "vel_0670_dns.prof"))
        assert len(df) == 513
        assert "u_plus" in df.columns
        assert "u_rms_plus" in df.columns
        assert df["y_plus"].iloc[0] == 0.0

    def test_parse_budget_profile(self):
        data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw", "kth_dns")
        if not os.path.exists(data_dir):
            pytest.skip("KTH data not available")
        df = parse_budget_profile(os.path.join(data_dir, "bud_0670_dns_k.prof"))
        assert len(df) > 500
        assert "production" in df.columns
        assert "dissipation" in df.columns


class TestIngestion:
    def test_condition_count(self, engine_with_data):
        with engine_with_data.connect() as conn:
            n = conn.execute(text("SELECT COUNT(*) FROM simulation_conditions")).scalar()
        assert n == 10

    def test_velocity_profile_count(self, engine_with_data):
        with engine_with_data.connect() as conn:
            n = conn.execute(text("SELECT COUNT(*) FROM velocity_profiles")).scalar()
        assert n == 5130  # 10 Re × 513 points

    def test_budget_count(self, engine_with_data):
        with engine_with_data.connect() as conn:
            n_tke = conn.execute(text("SELECT COUNT(*) FROM tke_budgets")).scalar()
            n_rs = conn.execute(text("SELECT COUNT(*) FROM reynolds_stress_budgets")).scalar()
        assert n_tke == 5130
        assert n_rs == 20520  # 10 Re × 513 points × 4 components


class TestFeatureEngineering:
    def test_derived_features(self, engine_with_data):
        df = compute_derived_features(engine_with_data)
        assert len(df) == 10
        assert "tke_peak" in df.columns
        assert df["tke_peak"].min() > 0

    def test_profile_features(self, engine_with_data):
        df = get_profile_features_per_point(engine_with_data)
        assert len(df) > 5000
        assert "bl_region" in df.columns
        regions = set(df["bl_region"].unique())
        assert "log_layer" in regions
        assert "buffer_layer" in regions


class TestPhysics:
    """Verify physical consistency of the DNS data."""

    def test_log_law(self, engine_with_data):
        """U+ should approximately follow log law in the log layer."""
        df = pd.read_sql("""
            SELECT vp.y_plus, vp.u_plus FROM velocity_profiles vp
            JOIN simulation_conditions sc ON vp.condition_id = sc.condition_id
            WHERE sc.re_theta > 2000 AND vp.y_plus BETWEEN 50 AND 200
        """, engine_with_data)
        log_law = (1/0.41) * np.log(df["y_plus"]) + 5.2
        residual = (df["u_plus"] - log_law).abs().mean()
        assert residual < 1.0  # Within 1 wall unit

    def test_wall_boundary(self, engine_with_data):
        """U+ = 0 and u'_rms = 0 at the wall."""
        df = pd.read_sql("""
            SELECT u_plus, u_rms_plus FROM velocity_profiles
            WHERE y_plus = 0.0
        """, engine_with_data)
        assert (df["u_plus"] == 0).all()
        assert (df["u_rms_plus"] == 0).all()

    def test_cf_decreases(self, engine_with_data):
        """Skin friction should decrease with increasing Re_theta."""
        df = pd.read_sql(
            "SELECT re_theta, cf FROM simulation_conditions ORDER BY re_theta",
            engine_with_data
        )
        assert (df["cf"].diff().dropna() < 0).all()
