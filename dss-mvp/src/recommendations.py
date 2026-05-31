from __future__ import annotations

from datetime import date, timedelta

from psycopg import Connection

from repositories import (
    get_maintenance_work_id_by_code,
    insert_maintenance_recommendation,
)


def build_recommendation(technical_state: str, predicted_rul: float) -> dict[str, object]:
    """Build a maintenance recommendation from the technical state."""

    if technical_state == "normal":
        return {
            "priority": "low",
            "recommendation_text": "Обслуживание не требуется, рекомендуется продолжить штатный мониторинг.",
            "recommended_work_code": None,
            "deadline_offset_days": None,
        }
    if technical_state == "observation_required":
        return {
            "priority": "medium",
            "recommendation_text": "Рекомендуется усилить наблюдение и выполнить дополнительную диагностику.",
            "recommended_work_code": "VIBRO_CHECK",
            "deadline_offset_days": 14,
        }
    if technical_state == "maintenance_required":
        return {
            "priority": "high",
            "recommendation_text": "Рекомендуется включить оборудование в ближайший план регламентных работ.",
            "recommended_work_code": "PLANNED_SERVICE",
            "deadline_offset_days": 7,
        }
    return {
        "priority": "critical",
        "recommendation_text": "Рекомендуется срочное обслуживание оборудования.",
        "recommended_work_code": "BEARING_REPLACE",
        "deadline_offset_days": 1,
    }


def save_recommendation(
    connection: Connection,
    equipment_id: int,
    diagnostic_report_id: int,
    rul_forecast_id: int,
    technical_state: str,
    predicted_rul: float,
    base_date: date | None = None,
) -> dict[str, object]:
    """Build and save a maintenance recommendation in the database."""

    recommendation = build_recommendation(technical_state, predicted_rul)
    recommendation_date = base_date or date.today()
    deadline_offset_days = recommendation["deadline_offset_days"]

    recommended_work_code = recommendation["recommended_work_code"]
    recommended_work_id = None

    if recommendation["priority"] == "medium":
        recommended_work_id = get_maintenance_work_id_by_code(connection, "VIBRO_CHECK")
        if recommended_work_id is None:
            recommended_work_id = get_maintenance_work_id_by_code(connection, "TEMP_CHECK")
    elif recommendation["priority"] == "high":
        recommended_work_id = get_maintenance_work_id_by_code(connection, "PLANNED_SERVICE")
    elif recommendation["priority"] == "critical":
        recommended_work_id = get_maintenance_work_id_by_code(connection, "BEARING_REPLACE")
        if recommended_work_id is None:
            recommended_work_id = get_maintenance_work_id_by_code(connection, "PLANNED_SERVICE")
    elif recommended_work_code:
        recommended_work_id = get_maintenance_work_id_by_code(connection, str(recommended_work_code))

    recommended_start_date = recommendation_date if recommendation["priority"] != "low" else None
    recommended_deadline = (
        recommendation_date + timedelta(days=int(deadline_offset_days))
        if deadline_offset_days is not None
        else None
    )

    saved_record = insert_maintenance_recommendation(
        connection=connection,
        equipment_id=equipment_id,
        rul_forecast_id=rul_forecast_id,
        diagnostic_report_id=diagnostic_report_id,
        recommended_work_id=recommended_work_id,
        priority=str(recommendation["priority"]),
        recommendation_text=str(recommendation["recommendation_text"]),
        recommended_start_date=recommended_start_date,
        recommended_deadline=recommended_deadline,
    )

    saved_record["recommended_work_code"] = recommended_work_code
    return saved_record
