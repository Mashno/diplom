from __future__ import annotations

from psycopg import Connection, Error, connect

from config import load_database_settings


def get_connection() -> Connection:
    """Create a PostgreSQL connection using settings from .env."""

    settings = load_database_settings()
    return connect(
        host=settings.host,
        port=settings.port,
        dbname=settings.dbname,
        user=settings.user,
        password=settings.password,
        connect_timeout=settings.connect_timeout,
        sslmode=settings.sslmode,
        prepare_threshold=None,
    )


def check_connection() -> dict[str, str]:
    """Verify that the application can connect to PostgreSQL."""

    try:
        with get_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    select
                        current_database(),
                        current_user,
                        version()
                    """
                )
                database_name, user_name, version = cursor.fetchone()
    except (ValueError, Error) as exc:
        raise ConnectionError(f"Failed to connect to PostgreSQL: {exc}") from exc

    return {
        "database": database_name,
        "user": user_name,
        "version": version,
    }
