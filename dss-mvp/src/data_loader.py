from __future__ import annotations

from typing import Any

import pandas as pd
from psycopg import Connection

from repositories import insert_telemetry_measurements


def load_telemetry_data(connection: Connection, telemetry_df: pd.DataFrame) -> int:
    """Load validated telemetry data into PostgreSQL."""

    if telemetry_df.empty:
        return 0

    columns = [
        "measured_at",
        "equipment_id",
        "parameter_id",
        "quality_code_id",
        "value",
        "source_system",
    ]
    rows: list[dict[str, Any]] = telemetry_df.loc[:, columns].to_dict(orient="records")
    return insert_telemetry_measurements(connection, rows)
