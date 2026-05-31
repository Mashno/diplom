from __future__ import annotations

from typing import Any

from config import CMAPSS_RUL_CAP


def classify_technical_state(predicted_rul: float) -> str:
    """Classify technical state from the predicted remaining useful life."""

    if predicted_rul > 125:
        return "normal"
    if predicted_rul > 80:
        return "observation_required"
    if predicted_rul > 30:
        return "maintenance_required"
    return "critical"


def calculate_degradation_index(predicted_rul: float) -> float:
    """Convert predicted RUL into a 0..1 degradation index."""

    degradation_index = 1.0 - float(predicted_rul) / float(CMAPSS_RUL_CAP)
    return max(0.0, min(1.0, degradation_index))


def build_diagnostic_result(
    equipment: dict[str, Any],
    predicted_rul: float,
) -> dict[str, Any]:
    """Build a diagnostic result for one engine unit."""

    technical_state = classify_technical_state(predicted_rul)
    degradation_index = calculate_degradation_index(predicted_rul)
    anomaly_score = degradation_index
    summary = (
        "По результатам анализа траектории деградации оборудования рассчитан индекс "
        f"деградации {degradation_index:.2f}. Техническое состояние "
        f"классифицировано как {technical_state}."
    )

    return {
        "equipment_id": int(equipment["equipment_id"]),
        "equipment_code": equipment["equipment_code"],
        "technical_state": technical_state,
        "degradation_index": degradation_index,
        "anomaly_score": anomaly_score,
        "summary": summary,
    }
