"""
Database configuration and connection settings.

Supports both PostgreSQL (production) and SQLite (demonstration/testing).
"""

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class DatabaseConfig:
    """Database connection configuration."""
    backend: str = "postgresql"
    host: str = "localhost"
    port: int = 5432
    database: str = "tbl_analytics"
    user: str = "tbl_user"
    password: str = "tbl_pass"
    sqlite_path: str = "data/tbl_analytics.db"

    @classmethod
    def from_env(cls, backend: str = None) -> "DatabaseConfig":
        """Load configuration from environment variables."""
        return cls(
            backend=backend or os.getenv("DB_BACKEND", "postgresql"),
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=int(os.getenv("POSTGRES_PORT", "5432")),
            database=os.getenv("POSTGRES_DB", "tbl_analytics"),
            user=os.getenv("POSTGRES_USER", "tbl_user"),
            password=os.getenv("POSTGRES_PASSWORD", "tbl_pass"),
            sqlite_path=os.getenv("SQLITE_PATH", "data/tbl_analytics.db"),
        )

    @property
    def connection_string(self) -> str:
        """SQLAlchemy connection string."""
        if self.backend == "sqlite":
            return f"sqlite:///{self.sqlite_path}"
        return (
            f"postgresql://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.database}"
        )


# Flow simulation parameters
SIMULATION_CONFIG = {
    "u_inf": 1.0,            # Freestream velocity (normalised)
    "nu": 1.0 / 800.0,       # Kinematic viscosity (Re_theta ~ 800)
    "n_streamwise": 12,       # Number of streamwise stations
    "n_wall_normal": 15,      # Number of wall-normal heights per station
    "n_spanwise": 1,          # Number of spanwise positions (single plane)
    "n_timesteps": 500,       # Number of temporal snapshots per probe
    "x_range": (0.05, 0.60),  # Streamwise extent [m]
    "y_range": (0.001, 0.10), # Wall-normal extent [m]
    "z_position": 0.0,        # Spanwise plane location [m]
    "dt": 0.002,              # Time step between snapshots [s]
}
