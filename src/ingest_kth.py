"""
Data ingestion for KTH DNS boundary layer data (Schlatter & Örlü, 2010).

Parses .prof files from the KTH dataset, extracts integral quantities
from file headers, and ingests velocity profiles and budget data into
the relational database.

Data source: https://www.mech.kth.se/~pschlatt/DATA/
Reference: Schlatter & Örlü, J. Fluid Mech. 659:116-126, 2010
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import SupportsFloat

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DatabaseConfig
from src.db import get_engine, init_schema

logger = logging.getLogger(__name__)


def _to_native_float(value: SupportsFloat | None) -> float | None:
    """Return a plain Python float for SQL parameter binding."""
    if value is None:
        return None
    return float(value)


# ---------------------------------------------------------------------------
# File parsers
# ---------------------------------------------------------------------------


def parse_header(filepath: str | Path) -> dict[str, float]:
    """
    Extract integral quantities from a ``.prof`` file header.

    Returns a dictionary with Reynolds-number and skin-friction metadata.
    """
    header: dict[str, float] = {}
    path = Path(filepath)

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("%") or line.startswith("Wall-normal"):
                pass

            match = re.search(r"Re_\{\\theta\}\s*=\s*([\d.]+)", line)
            if match:
                header["re_theta"] = float(match.group(1))

            match = re.search(r"Re_\{\\delta\^\*\}\s*=\s*([\d.]+)", line)
            if match:
                header["re_delta_star"] = float(match.group(1))

            match = re.search(r"Re_\{\\tau\}\s*=\s*([\d.]+)", line)
            if match:
                header["re_tau"] = float(match.group(1))

            match = re.search(r"H_\{12\}\s*=\s*([\d.]+)", line)
            if match:
                header["shape_factor"] = float(match.group(1))

            match = re.search(r"c_f\s*=\s*([\d.]+)", line)
            if match:
                header["cf"] = float(match.group(1))

            if line and line[0].isdigit():
                break

    return header


def parse_velocity_profile(filepath: str | Path) -> pd.DataFrame:
    """
    Parse a ``vel_XXXX_dns.prof`` file.

    Columns (2012 update, 17 columns):
    y/delta_99, y+, U+, urms+, vrms+, wrms+, uv+, prms+, pu+, pv+,
    S(u), F(u), dU+/dy+, V+, omxrms+, omyrms+, omzrms+
    """
    col_names = [
        "y_delta",
        "y_plus",
        "u_plus",
        "u_rms_plus",
        "v_rms_plus",
        "w_rms_plus",
        "uv_plus",
        "p_rms_plus",
        "pu_plus",
        "pv_plus",
        "skewness_u",
        "flatness_u",
        "du_dy_plus",
        "v_plus",
        "omx_rms_plus",
        "omy_rms_plus",
        "omz_rms_plus",
    ]

    rows: list[list[float]] = []
    path = Path(filepath)

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line[0] in ("%", "D", "R", "S", "H", "c", "W", "y"):
                continue

            parts = line.split()
            if len(parts) >= 17:
                try:
                    rows.append([float(value) for value in parts[:17]])
                except ValueError:
                    continue

    return pd.DataFrame(rows, columns=col_names)


def parse_budget_profile(filepath: str | Path) -> pd.DataFrame:
    """
    Parse a ``bud_XXXX_dns_*.prof`` file.

    Columns (9 columns):
    y/delta_99, y+, conv+, prod+, diss+, t-diff+, velp+, vis-diff+, residual+
    """
    col_names = [
        "y_delta",
        "y_plus",
        "convection",
        "production",
        "dissipation",
        "turb_diffusion",
        "vel_pressure",
        "visc_diffusion",
        "residual",
    ]

    rows: list[list[float]] = []
    path = Path(filepath)

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or not line[0].isdigit():
                continue

            parts = line.split()
            if len(parts) >= 9:
                try:
                    rows.append([float(value) for value in parts[:9]])
                except ValueError:
                    continue

    return pd.DataFrame(rows, columns=col_names)


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------


def ingest_kth_data(engine: Engine, data_dir: str | Path) -> None:
    """Ingest all KTH DNS ``.prof`` files into the database."""
    data_dir_path = Path(data_dir)
    tables = [
        "predictions",
        "derived_features",
        "reynolds_stress_budgets",
        "tke_budgets",
        "velocity_profiles",
        "simulation_conditions",
    ]

    with engine.begin() as conn:
        for table in tables:
            try:
                conn.execute(text(f"DELETE FROM {table}"))
            except Exception:
                pass
    logger.info("Cleared existing data")

    vel_files = sorted(data_dir_path.glob("vel_*_dns.prof"))
    if not vel_files:
        raise FileNotFoundError(f"No velocity profile files found in {data_dir_path}")

    logger.info("Found %s velocity profile files", len(vel_files))

    for vel_file in vel_files:
        basename = vel_file.name
        re_match = re.search(r"vel_(\d+)_dns", basename)
        if re_match is None:
            continue
        re_str = re_match.group(1)

        header = parse_header(vel_file)
        re_theta = header.get("re_theta", float(re_str))
        re_tau = header.get("re_tau")
        cf = header.get("cf")
        u_tau = _to_native_float(np.sqrt(cf / 2.0)) if cf is not None else None

        with engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO simulation_conditions
                    (re_theta, re_delta_star, re_tau, shape_factor, cf, u_tau, source_file)
                    VALUES (:rt, :rds, :rtau, :h, :cf, :utau, :src)
                    """
                ),
                {
                    "rt": _to_native_float(re_theta),
                    "rds": _to_native_float(header.get("re_delta_star")),
                    "rtau": _to_native_float(re_tau),
                    "h": _to_native_float(header.get("shape_factor")),
                    "cf": _to_native_float(cf),
                    "utau": u_tau,
                    "src": basename,
                },
            )
            condition_id = int(
                conn.execute(
                    text("SELECT condition_id FROM simulation_conditions WHERE re_theta = :rt"),
                    {"rt": _to_native_float(re_theta)},
                ).scalar_one()
            )

        vel_df = parse_velocity_profile(vel_file)
        vel_df["condition_id"] = condition_id
        vel_df.to_sql("velocity_profiles", engine, if_exists="append", index=False)
        logger.info(
            "  Re_theta=%.0f: %s velocity profile points",
            re_theta,
            len(vel_df),
        )

        tke_file = data_dir_path / f"bud_{re_str}_dns_k.prof"
        if tke_file.exists():
            tke_df = parse_budget_profile(tke_file)
            tke_df["condition_id"] = condition_id
            tke_df.to_sql("tke_budgets", engine, if_exists="append", index=False)

        for component in ["uu", "vv", "ww", "uv"]:
            bud_file = data_dir_path / f"bud_{re_str}_dns_{component}.prof"
            if bud_file.exists():
                bud_df = parse_budget_profile(bud_file)
                bud_df["condition_id"] = condition_id
                bud_df["component"] = component
                bud_df.to_sql("reynolds_stress_budgets", engine, if_exists="append", index=False)

    with engine.connect() as conn:
        n_cond = int(conn.execute(text("SELECT COUNT(*) FROM simulation_conditions")).scalar_one())
        n_vel = int(conn.execute(text("SELECT COUNT(*) FROM velocity_profiles")).scalar_one())
        n_tke = int(conn.execute(text("SELECT COUNT(*) FROM tke_budgets")).scalar_one())
        n_rs = int(conn.execute(text("SELECT COUNT(*) FROM reynolds_stress_budgets")).scalar_one())

    logger.info("\nIngestion complete:")
    logger.info("  Simulation conditions: %s", n_cond)
    logger.info("  Velocity profile rows: %s", f"{n_vel:,}")
    logger.info("  TKE budget rows:       %s", f"{n_tke:,}")
    logger.info("  RS budget rows:        %s", f"{n_rs:,}")
    logger.info("  Total rows:            %s", f"{n_vel + n_tke + n_rs:,}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for KTH DNS ingestion."""
    parser = argparse.ArgumentParser(description="Ingest KTH DNS boundary layer data")
    parser.add_argument(
        "--data-dir",
        default="data/raw/kth_dns",
        help="Directory containing .prof files (default: data/raw/kth_dns)",
    )
    parser.add_argument(
        "--backend",
        choices=["postgresql", "sqlite"],
        default="sqlite",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    config = DatabaseConfig.from_env(backend=args.backend)
    engine = get_engine(config)
    init_schema(engine)
    ingest_kth_data(engine, args.data_dir)


if __name__ == "__main__":
    main()
