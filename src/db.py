"""
Database connection utilities.

Provides engine creation, session management, and schema initialisation
for both PostgreSQL and SQLite backends.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from src.config import DatabaseConfig

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
SCHEMA_DIR = REPO_ROOT / "sql" / "schema"
SCHEMA_PATHS: dict[str, Path] = {
    "sqlite": SCHEMA_DIR / "schema_sqlite.sql",
    "postgresql": SCHEMA_DIR / "schema_postgres.sql",
    "postgres": SCHEMA_DIR / "schema_postgres.sql",
}


def get_engine(config: DatabaseConfig | None = None) -> Engine:
    """Create a SQLAlchemy engine from configuration."""
    if config is None:
        config = DatabaseConfig.from_env()

    engine = create_engine(
        config.connection_string,
        echo=False,
        pool_pre_ping=(config.backend == "postgresql"),
    )
    logger.info("Database engine created: %s", config.backend)
    return engine


def get_default_schema_path(backend_name: str) -> Path:
    """Return the default schema file for a backend."""
    if backend_name not in SCHEMA_PATHS:
        supported = ", ".join(sorted(SCHEMA_PATHS))
        raise ValueError(f"Unsupported backend '{backend_name}'. Supported backends: {supported}.")
    return SCHEMA_PATHS[backend_name]


def init_schema(engine: Engine, schema_path: str | Path | None = None) -> None:
    """
    Initialise database schema from the backend-specific SQL file.

    When ``schema_path`` is omitted, the function picks the matching file from
    ``sql/schema/schema_sqlite.sql`` or ``sql/schema/schema_postgres.sql``.
    """
    backend_name = engine.url.get_backend_name()
    resolved_schema = (
        Path(schema_path) if schema_path is not None else get_default_schema_path(backend_name)
    )

    if not resolved_schema.exists():
        raise FileNotFoundError(f"Schema file not found: {resolved_schema}")

    schema_sql = resolved_schema.read_text(encoding="utf-8")
    statements = [statement.strip() for statement in schema_sql.split(";") if statement.strip()]

    with engine.begin() as conn:
        for statement in statements:
            clean_statement = _strip_comment_lines(statement)
            if not clean_statement:
                continue
            conn.execute(text(clean_statement))

    try:
        schema_label = resolved_schema.relative_to(REPO_ROOT)
    except ValueError:
        schema_label = resolved_schema

    logger.info(
        "Database schema initialised for %s using %s",
        backend_name,
        schema_label,
    )


@contextmanager
def get_session(engine: Engine | None = None) -> Iterator[Session]:
    """Yield a database session with automatic commit and rollback handling."""
    if engine is None:
        engine = get_engine()

    session_factory = sessionmaker(bind=engine)
    session: Session = session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def execute_sql_file(engine: Engine, filepath: str | Path) -> list[Any]:
    """Execute a SQL file and return rows from the last ``SELECT`` statement."""
    sql_content = Path(filepath).read_text(encoding="utf-8")
    results: list[Any] = []

    with engine.begin() as conn:
        for statement in [part.strip() for part in sql_content.split(";") if part.strip()]:
            clean_statement = _strip_comment_lines(statement)
            if not clean_statement:
                continue
            result = conn.execute(text(clean_statement))
            if result.returns_rows:
                results = list(result.fetchall())

    return results


def _strip_comment_lines(statement: str) -> str:
    """Remove full-line SQL comments and surrounding whitespace."""
    lines = [line for line in statement.splitlines() if not line.strip().startswith("--")]
    return "\n".join(lines).strip()
