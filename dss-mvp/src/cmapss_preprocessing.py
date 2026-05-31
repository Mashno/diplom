from __future__ import annotations

from collections.abc import Iterator
from datetime import timedelta

import pandas as pd

from config import (
    CMAPSS_BASE_TIME,
    CMAPSS_FEATURE_COLUMNS,
    CMAPSS_OPERATIONAL_COLUMNS,
    CMAPSS_QUALITY_CODE_ID,
    CMAPSS_RUL_CAP,
    CMAPSS_SENSOR_COLUMNS,
    DEFAULT_CMAPSS_DATASET_ID,
    get_cmapss_source_system,
)


def convert_train_to_telemetry_rows(
    train_df: pd.DataFrame,
    equipment_id_map: dict[int, int],
    parameter_id_map: dict[str, int],
    dataset_id: str = DEFAULT_CMAPSS_DATASET_ID,
) -> Iterator[tuple]:
    """Convert C-MAPSS train cycles into telemetry_measurements rows."""

    required_columns = [*CMAPSS_OPERATIONAL_COLUMNS, *CMAPSS_SENSOR_COLUMNS]
    source_system = get_cmapss_source_system(dataset_id)

    for record in train_df.itertuples(index=False):
        unit_number = int(record.unit_number)
        equipment_id = equipment_id_map[unit_number]
        measured_at = CMAPSS_BASE_TIME + timedelta(hours=int(record.time_in_cycles))

        for column_name in required_columns:
            parameter_id = parameter_id_map[column_name]
            value = float(getattr(record, column_name))
            yield (
                measured_at,
                equipment_id,
                parameter_id,
                CMAPSS_QUALITY_CODE_ID,
                value,
                source_system,
            )


def build_training_dataset(
    train_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Build model features and capped target values for RUL training."""

    training_df = train_df.copy()
    training_df["capped_RUL"] = training_df["RUL"].clip(upper=CMAPSS_RUL_CAP)
    features = training_df.loc[:, CMAPSS_FEATURE_COLUMNS].copy()
    target = training_df["capped_RUL"].astype(float)
    return features, target, list(CMAPSS_FEATURE_COLUMNS)
