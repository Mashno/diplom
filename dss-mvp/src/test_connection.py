from __future__ import annotations

from database import check_connection


def main() -> int:
    """Run a standalone database connection check."""

    try:
        result = check_connection()
        print("Connection successful.")
        print(f"Database: {result['database']}")
        print(f"User: {result['user']}")
        print(f"Server: {result['version']}")
        return 0
    except Exception as exc:
        print(f"Connection failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
