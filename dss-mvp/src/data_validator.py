from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def validate_telemetry_data(
    telemetry_df: pd.DataFrame,
    equipment_records: list[dict[str, Any]],
    parameter_records: list[dict[str, Any]],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Validate synthetic telemetry and print quality statistics."""

    if telemetry_df.empty:
        stats = {
            "total_rows": 0,
            "valid_rows": 0,
            "invalid_rows": 0,
            "null_rows": 0,
            "invalid_equipment_rows": 0,
            "invalid_parameter_rows": 0,
            "invalid_value_rows": 0,
            "invalid_timestamp_rows": 0,
            "potential_outliers": 0,
        }
        print("Валидация данных: входной набор телеметрии пуст.")
        return telemetry_df.copy(), stats

    equipment_ids = {int(record["equipment_id"]) for record in equipment_records}
    parameter_limits = pd.DataFrame(parameter_records)[
        ["parameter_id", "critical_min", "critical_max"]
    ].copy()

    validated = telemetry_df.copy()
    validated["measured_at"] = pd.to_datetime(
        validated["measured_at"],
        utc=True,
        errors="coerce",
    )
    validated["value"] = pd.to_numeric(validated["value"], errors="coerce")
    validated["equipment_id"] = pd.to_numeric(validated["equipment_id"], errors="coerce").astype("Int64")
    validated["parameter_id"] = pd.to_numeric(validated["parameter_id"], errors="coerce").astype("Int64")

    validated = validated.merge(parameter_limits, on="parameter_id", how="left")

    null_mask = validated[["measured_at", "equipment_id", "parameter_id", "value"]].isna().any(axis=1)
    invalid_equipment_mask = ~validated["equipment_id"].isin(equipment_ids)
    invalid_parameter_mask = validated["critical_max"].isna() & validated["critical_min"].isna()
    invalid_value_mask = ~np.isfinite(validated["value"])
    invalid_timestamp_mask = validated["measured_at"].isna()

    below_critical_mask = (
        validated["critical_min"].notna()
        & validated["value"].notna()
        & (validated["value"] < pd.to_numeric(validated["critical_min"], errors="coerce"))
    )
    above_critical_mask = (
        validated["critical_max"].notna()
        & validated["value"].notna()
        & (validated["value"] > pd.to_numeric(validated["critical_max"], errors="coerce"))
    )
    potential_outlier_mask = below_critical_mask | above_critical_mask

    invalid_mask = (
        null_mask
        | invalid_equipment_mask
        | invalid_parameter_mask
        | invalid_value_mask
        | invalid_timestamp_mask
    )
    clean_df = validated.loc[~invalid_mask].copy()
    clean_df["equipment_id"] = clean_df["equipment_id"].astype(int)
    clean_df["parameter_id"] = clean_df["parameter_id"].astype(int)
    clean_df["quality_code_id"] = pd.to_numeric(clean_df["quality_code_id"], errors="coerce").fillna(1).astype(int)
    clean_df["value"] = clean_df["value"].astype(float)

    stats = {
        "total_rows": int(len(validated)),
        "valid_rows": int(len(clean_df)),
        "invalid_rows": int(invalid_mask.sum()),
        "null_rows": int(null_mask.sum()),
        "invalid_equipment_rows": int(invalid_equipment_mask.sum()),
        "invalid_parameter_rows": int(invalid_parameter_mask.sum()),
        "invalid_value_rows": int(invalid_value_mask.sum()),
        "invalid_timestamp_rows": int(invalid_timestamp_mask.sum()),
        "potential_outliers": int(potential_outlier_mask.sum()),
    }

    print("Статистика качества данных:")
    print(f"- Всего записей: {stats['total_rows']}")
    print(f"- Валидных записей: {stats['valid_rows']}")
    print(f"- Невалидных записей: {stats['invalid_rows']}")
    print(f"- Строк с пустыми значениями: {stats['null_rows']}")
    print(f"- Строк с некорректным equipment_id: {stats['invalid_equipment_rows']}")
    print(f"- Строк с некорректным parameter_id: {stats['invalid_parameter_rows']}")
    print(f"- Строк с некорректным value: {stats['invalid_value_rows']}")
    print(f"- Строк с некорректным measured_at: {stats['invalid_timestamp_rows']}")
    print(f"- Потенциальных выбросов: {stats['potential_outliers']}")

    return (
        clean_df[
            [
                "measured_at",
                "equipment_id",
                "parameter_id",
                "quality_code_id",
                "value",
                "source_system",
            ]
        ].copy(),
        stats,
    )
