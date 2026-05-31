from __future__ import annotations

from datetime import datetime, timezone

from config import PROJECT_ROOT, SCENARIO_DAYS
from data_generator import generate_synthetic_telemetry
from data_loader import load_telemetry_data
from data_validator import validate_telemetry_data
from database import check_connection, get_connection
from diagnostics import evaluate_equipment_condition
from recommendations import build_recommendation
from report_builder import build_text_report
from repositories import (
    get_equipment,
    get_latest_results,
    get_maintenance_works,
    get_telemetry_for_equipment,
    get_telemetry_parameters,
    insert_diagnostic_report,
    insert_maintenance_recommendation,
    insert_rul_forecast,
)
from rul_forecasting import forecast_rul


def main() -> int:
    """Execute the full MVP pipeline from telemetry generation to reporting."""

    run_started_at = datetime.now(timezone.utc)

    try:
        check_connection()
        print("Подключение к базе данных успешно.")

        with get_connection() as connection:
            equipment_records = get_equipment(connection)
            parameter_records = get_telemetry_parameters(connection)
            maintenance_works = get_maintenance_works(connection)

            if not equipment_records:
                raise RuntimeError("В таблице app.equipment не найдено активного оборудования.")
            if not parameter_records:
                raise RuntimeError("В таблице telemetry.telemetry_parameters не найдены параметры.")

            print(f"Найдено оборудования: {len(equipment_records)}.")
            print(f"Найдено телеметрических параметров: {len(parameter_records)}.")

            telemetry_df = generate_synthetic_telemetry(
                equipment_records=equipment_records,
                parameter_records=parameter_records,
                days=SCENARIO_DAYS,
                end_time=run_started_at,
            )
            print(f"Сгенерировано телеметрических записей: {len(telemetry_df)}.")

            validated_df, validation_stats = validate_telemetry_data(
                telemetry_df=telemetry_df,
                equipment_records=equipment_records,
                parameter_records=parameter_records,
            )

            loaded_count = load_telemetry_data(connection, validated_df)
            print(f"Загружено телеметрических записей: {loaded_count}.")

            for equipment in equipment_records:
                telemetry_for_equipment = get_telemetry_for_equipment(
                    connection,
                    int(equipment["equipment_id"]),
                    days=SCENARIO_DAYS,
                )

                diagnostic_result = evaluate_equipment_condition(
                    equipment=equipment,
                    telemetry_df=telemetry_for_equipment,
                )
                diagnostic_record = insert_diagnostic_report(
                    connection=connection,
                    equipment_id=int(equipment["equipment_id"]),
                    technical_state=diagnostic_result["technical_state"],
                    degradation_index=diagnostic_result["degradation_index"],
                    anomaly_score=diagnostic_result["anomaly_score"],
                    summary=diagnostic_result["summary"],
                )
                diagnostic_result["diagnostic_report_id"] = diagnostic_record["diagnostic_report_id"]

                rul_result = forecast_rul(
                    equipment=equipment,
                    telemetry_df=telemetry_for_equipment,
                    diagnostic_result=diagnostic_result,
                )
                rul_record = insert_rul_forecast(
                    connection=connection,
                    equipment_id=int(equipment["equipment_id"]),
                    diagnostic_report_id=diagnostic_record["diagnostic_report_id"],
                    predicted_rul_hours=rul_result["predicted_rul_hours"],
                    lower_bound_hours=rul_result["lower_bound_hours"],
                    upper_bound_hours=rul_result["upper_bound_hours"],
                    confidence_score=rul_result["confidence_score"],
                    model_name=rul_result["model_name"],
                    model_version=rul_result["model_version"],
                )
                rul_result["rul_forecast_id"] = rul_record["rul_forecast_id"]

                recommendation_result = build_recommendation(
                    equipment=equipment,
                    diagnostic_result=diagnostic_result,
                    rul_result=rul_result,
                    maintenance_works=maintenance_works,
                    base_date=run_started_at.date(),
                )
                insert_maintenance_recommendation(
                    connection=connection,
                    equipment_id=int(equipment["equipment_id"]),
                    rul_forecast_id=rul_record["rul_forecast_id"],
                    diagnostic_report_id=diagnostic_record["diagnostic_report_id"],
                    recommended_work_id=recommendation_result["recommended_work_id"],
                    priority=recommendation_result["priority"],
                    recommendation_text=recommendation_result["recommendation_text"],
                    recommended_start_date=recommendation_result["recommended_start_date"],
                    recommended_deadline=recommendation_result["recommended_deadline"],
                )

                print()
                print(f"Оборудование: {equipment['equipment_code']}")
                print(f"Состояние: {diagnostic_result['technical_state']}")
                print(f"Индекс деградации: {diagnostic_result['degradation_index']:.2f}")
                print(f"Прогноз RUL: {rul_result['predicted_rul_hours']:.1f} часов")
                print(f"Приоритет рекомендации: {recommendation_result['priority']}")
                print(f"Рекомендация: {recommendation_result['recommendation_text']}")

            latest_results = get_latest_results(connection)

        report_path = build_text_report(
            run_started_at=run_started_at,
            equipment_records=equipment_records,
            generated_count=len(telemetry_df),
            loaded_count=loaded_count,
            validation_stats=validation_stats,
            latest_results=latest_results,
        )
        relative_path = report_path.relative_to(PROJECT_ROOT)
        print()
        print(f"Отчёт сохранён: {relative_path}")
        return 0
    except Exception as exc:
        print(f"Сценарий завершился с ошибкой: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
