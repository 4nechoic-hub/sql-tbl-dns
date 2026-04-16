"""
Data ingestion for KTH DNS boundary layer data (Schlatter & Örlü, 2010).

Parses .prof files from the KTH dataset, extracts integral quantities
from file headers, and ingests velocity profiles and budget data into
the relational database.

Data source: https://www.mech.kth.se/~pschlatt/DATA/
Reference: Schlatter & Örlü, J. Fluid Mech. 659:116-126, 2010
"""

import argparse
import glob
import logging
import os
import re
import sys

import numpy as np
import pandas as pd
from sqlalchemy import text

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import DatabaseConfig
from src.db import get_engine, init_schema

logger = logging.getLogger(__name__)


def _to_native_float(value):
    """Return a plain Python float for SQL parameter binding."""
    if value is None:
        return None
    return float(value)


# ---------------------------------------------------------------------------
# File parsers
# ---------------------------------------------------------------------------

def parse_header(filepath: str) -> dict:
    """
    Extract integral quantities from .prof file header.
    Returns dict with re_theta, re_delta_star, re_tau, shape_factor, cf.
    """
    header = {}
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("%") or line.startswith("Wall-normal"):
                # Try to extract quantities from comment lines
                pass
            # Match patterns like: Re_{\theta}   =        677.452
            m = re.search(r"Re_\{\\theta\}\s*=\s*([\d.]+)", line)
            if m:
                header["re_theta"] = float(m.group(1))
            m = re.search(r"Re_\{\\delta\^\*\}\s*=\s*([\d.]+)", line)
            if m:
                header["re_delta_star"] = float(m.group(1))
            m = re.search(r"Re_\{\\tau\}\s*=\s*([\d.]+)", line)
            if m:
                header["re_tau"] = float(m.group(1))
            m = re.search(r"H_\{12\}\s*=\s*([\d.]+)", line)
            if m:
                header["shape_factor"] = float(m.group(1))
            m = re.search(r"c_f\s*=\s*([\d.]+)", line)
            if m:
                header["cf"] = float(m.group(1))
            # Stop reading header once we hit data
            if line and line[0].isdigit():
                break
    return header


def parse_velocity_profile(filepath: str) -> pd.DataFrame:
    """
    Parse vel_XXXX_dns.prof file.

    Columns (2012 update, 17 columns):
    y/delta_99, y+, U+, urms+, vrms+, wrms+, uv+, prms+, pu+, pv+,
    S(u), F(u), dU+/dy+, V+, omxrms+, omyrms+, omzrms+
    """
    col_names = [
        "y_delta", "y_plus", "u_plus", "u_rms_plus", "v_rms_plus",
        "w_rms_plus", "uv_plus", "p_rms_plus", "pu_plus", "pv_plus",
        "skewness_u", "flatness_u", "du_dy_plus", "v_plus",
        "omx_rms_plus", "omy_rms_plus", "omz_rms_plus",
    ]

    rows = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            # Skip header/comment lines
            if not line or line[0] in ("%", "D", "R", "S", "H", "c", "W", "y"):
                continue
            # Data lines start with a digit or whitespace followed by digit
            parts = line.split()
            if len(parts) >= 17:
                try:
                    vals = [float(x) for x in parts[:17]]
                    rows.append(vals)
                except ValueError:
                    continue

    df = pd.DataFrame(rows, columns=col_names)
    return df


def parse_budget_profile(filepath: str) -> pd.DataFrame:
    """
    Parse bud_XXXX_dns_*.prof file.

    Columns (9 columns):
    y/delta_99, y+, conv+, prod+, diss+, t-diff+, velp+, vis-diff+, residual+
    """
    col_names = [
        "y_delta", "y_plus", "convection", "production", "dissipation",
        "turb_diffusion", "vel_pressure", "visc_diffusion", "residual",
    ]

    rows = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or not line[0].isdigit():
                continue
            parts = line.split()
            if len(parts) >= 9:
                try:
                    vals = [float(x) for x in parts[:9]]
                    rows.append(vals)
                except ValueError:
                    continue

    df = pd.DataFrame(rows, columns=col_names)
    return df


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

def ingest_kth_data(engine, data_dir: str):
    """
    Ingest all KTH DNS .prof files into the database.
    """
    # Clear existing data
    with engine.begin() as conn:
        for table in ["predictions", "derived_features", "reynolds_stress_budgets",
                       "tke_budgets", "velocity_profiles", "simulation_conditions"]:
            try:
                conn.execute(text(f"DELETE FROM {table}"))
            except Exception:
                pass
    logger.info("Cleared existing data")

    # Find all velocity profile files
    vel_files = sorted(glob.glob(os.path.join(data_dir, "vel_*_dns.prof")))
    if not vel_files:
        raise FileNotFoundError(f"No velocity profile files found in {data_dir}")

    logger.info(f"Found {len(vel_files)} velocity profile files")

    for vel_file in vel_files:
        # Extract Re_theta from filename
        basename = os.path.basename(vel_file)
        re_match = re.search(r"vel_(\d+)_dns", basename)
        if not re_match:
            continue
        re_str = re_match.group(1)

        # Parse header for integral quantities
        header = parse_header(vel_file)
        re_theta = header.get("re_theta", float(re_str))
        re_tau = header.get("re_tau")
        cf = header.get("cf")

        # Compute u_tau from cf: u_tau = U_inf * sqrt(cf/2)
        # In inner scaling, U_inf = Re_tau / (Re_theta * sqrt(cf/2))
        u_tau = _to_native_float(np.sqrt(cf / 2.0)) if cf is not None else None

        # Insert simulation condition
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO simulation_conditions
                (re_theta, re_delta_star, re_tau, shape_factor, cf, u_tau, source_file)
                VALUES (:rt, :rds, :rtau, :h, :cf, :utau, :src)
            """), {
                "rt": _to_native_float(re_theta),
                "rds": _to_native_float(header.get("re_delta_star")),
                "rtau": _to_native_float(re_tau),
                "h": _to_native_float(header.get("shape_factor")),
                "cf": _to_native_float(cf),
                "utau": u_tau,
                "src": basename,
            })

            # Get the condition_id
            result = conn.execute(text(
                "SELECT condition_id FROM simulation_conditions WHERE re_theta = :rt"
            ), {"rt": re_theta})
            condition_id = result.scalar()

        # Parse and ingest velocity profile
        vel_df = parse_velocity_profile(vel_file)
        vel_df["condition_id"] = condition_id
        vel_df.to_sql("velocity_profiles", engine, if_exists="append", index=False)
        logger.info(f"  Re_theta={re_theta:.0f}: {len(vel_df)} velocity profile points")

        # Parse and ingest TKE budget
        tke_file = os.path.join(data_dir, f"bud_{re_str}_dns_k.prof")
        if os.path.exists(tke_file):
            tke_df = parse_budget_profile(tke_file)
            tke_df["condition_id"] = condition_id
            tke_df.to_sql("tke_budgets", engine, if_exists="append", index=False)

        # Parse and ingest Reynolds stress budgets
        for component in ["uu", "vv", "ww", "uv"]:
            bud_file = os.path.join(data_dir, f"bud_{re_str}_dns_{component}.prof")
            if os.path.exists(bud_file):
                bud_df = parse_budget_profile(bud_file)
                bud_df["condition_id"] = condition_id
                bud_df["component"] = component
                bud_df.to_sql("reynolds_stress_budgets", engine,
                              if_exists="append", index=False)

    # Verification
    with engine.connect() as conn:
        n_cond = conn.execute(text("SELECT COUNT(*) FROM simulation_conditions")).scalar()
        n_vel = conn.execute(text("SELECT COUNT(*) FROM velocity_profiles")).scalar()
        n_tke = conn.execute(text("SELECT COUNT(*) FROM tke_budgets")).scalar()
        n_rs = conn.execute(text("SELECT COUNT(*) FROM reynolds_stress_budgets")).scalar()

    logger.info(f"\nIngestion complete:")
    logger.info(f"  Simulation conditions: {n_cond}")
    logger.info(f"  Velocity profile rows: {n_vel:,}")
    logger.info(f"  TKE budget rows:       {n_tke:,}")
    logger.info(f"  RS budget rows:        {n_rs:,}")
    logger.info(f"  Total rows:            {n_vel + n_tke + n_rs:,}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Ingest KTH DNS boundary layer data")
    parser.add_argument(
        "--data-dir", default="data/raw/kth_dns",
        help="Directory containing .prof files (default: data/raw/kth_dns)",
    )
    parser.add_argument(
        "--backend", choices=["postgresql", "sqlite"], default="sqlite",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    config = DatabaseConfig.from_env(backend=args.backend)
    engine = get_engine(config)
    init_schema(engine)

    ingest_kth_data(engine, args.data_dir)


if __name__ == "__main__":
    main()
