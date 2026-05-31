from __future__ import annotations

from datetime import datetime, timezone

from config import SCENARIO_DAYS
from data_generator import generate_synthetic_telemetry
from data_loader import load_telemetry_data
from data_validator import validate_telemetry_data
from database import get_connection
from repositories import get_equipment, get_telemetry_parameters


def main() -> int:
    """Generate, validate and load only synthetic telemetry."""

    try:
        with get_connection() as connection:
            equipment_records = get_equipment(connection)
            parameter_records = get_telemetry_parameters(connection)

            telemetry_df = generate_synthetic_telemetry(
                equipment_records=equipment_records,
                parameter_records=parameter_records,
                days=SCENARIO_DAYS,
                end_time=datetime.now(timezone.utc),
            )
            print(f"Сгенерировано телеметрических записей: {len(telemetry_df)}.")

            validated_df, _ = validate_telemetry_data(
                telemetry_df=telemetry_df,
                equipment_records=equipment_records,
                parameter_records=parameter_records,
            )
            loaded_count = load_telemetry_data(connection, validated_df)
            print(f"Загружено телеметрических записей: {loaded_count}.")
        return 0
    except Exception as exc:
        print(f"Ошибка загрузки тестовой телеметрии: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
