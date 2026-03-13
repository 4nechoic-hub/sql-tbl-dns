"""
Database connection utilities.

Provides engine creation, session management, and schema initialisation
for both PostgreSQL and SQLite backends.
"""

import os
import logging
from contextlib import contextmanager

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from src.config import DatabaseConfig

logger = logging.getLogger(__name__)


def get_engine(config: DatabaseConfig = None):
    """Create SQLAlchemy engine from configuration."""
    if config is None:
        config = DatabaseConfig.from_env()

    engine = create_engine(
        config.connection_string,
        echo=False,
        pool_pre_ping=True if config.backend == "postgresql" else False,
    )
    logger.info(f"Database engine created: {config.backend}")
    return engine


def init_schema(engine, schema_path: str = "sql/schema/01_create_tables.sql"):
    """
    Initialise database schema from SQL file.

    For SQLite, adapts PostgreSQL-specific syntax (SERIAL, GENERATED ALWAYS AS)
    to SQLite-compatible equivalents.
    """
    if not os.path.exists(schema_path):
        logger.warning(f"Schema file not found: {schema_path}")
        return

    with open(schema_path, "r") as f:
        schema_sql = f.read()

    backend = engine.url.get_backend_name()

    if backend == "sqlite":
        schema_sql = _adapt_sql_for_sqlite(schema_sql)

    # Execute each statement separately
    statements = [s.strip() for s in schema_sql.split(";") if s.strip()]

    with engine.begin() as conn:
        for stmt in statements:
            # Skip comments-only blocks
            lines = [l for l in stmt.split("\n") if not l.strip().startswith("--")]
            clean = "\n".join(lines).strip()
            if clean:
                try:
                    conn.execute(text(clean))
                except Exception as e:
                    # Skip errors for IF NOT EXISTS / IF EXISTS
                    if "already exists" in str(e).lower():
                        continue
                    logger.warning(f"Schema statement skipped: {e}")

    logger.info("Database schema initialised")


def _adapt_sql_for_sqlite(sql: str) -> str:
    """Adapt PostgreSQL SQL to SQLite-compatible syntax."""
    import re

    # Replace SERIAL with INTEGER (SQLite auto-increments INTEGER PRIMARY KEY)
    sql = sql.replace("SERIAL PRIMARY KEY", "INTEGER PRIMARY KEY AUTOINCREMENT")
    sql = re.sub(r"\bSERIAL\b", "INTEGER", sql)

    # Replace DOUBLE PRECISION with REAL
    sql = sql.replace("DOUBLE PRECISION", "REAL")

    # Remove GENERATED ALWAYS AS ... STORED columns
    sql = re.sub(
        r",\s*velocity_mag\s+\w+\s+GENERATED ALWAYS AS\s*\([^)]+\)\s*STORED",
        "",
        sql,
    )

    # CURRENT_TIMESTAMP works natively in SQLite — leave as is

    # Remove inline SQL comments that break SQLite parsing
    sql = re.sub(r"--[^\n]*", "", sql)

    return sql


@contextmanager
def get_session(engine=None):
    """Context manager for database sessions with automatic commit/rollback."""
    if engine is None:
        engine = get_engine()

    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def execute_sql_file(engine, filepath: str) -> list:
    """Execute a SQL file and return results from the last SELECT statement."""
    with open(filepath, "r") as f:
        sql_content = f.read()

    results = []
    statements = [s.strip() for s in sql_content.split(";") if s.strip()]

    with engine.begin() as conn:
        for stmt in statements:
            lines = [l for l in stmt.split("\n") if not l.strip().startswith("--")]
            clean = "\n".join(lines).strip()
            if clean:
                result = conn.execute(text(clean))
                if result.returns_rows:
                    results = result.fetchall()

    return results
