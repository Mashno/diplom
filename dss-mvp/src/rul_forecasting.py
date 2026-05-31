from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from config import LARGE_RUL_HOURS, RUL_MODEL_NAME, RUL_MODEL_VERSION
from diagnostics import build_degradation_time_series


def forecast_rul(
    equipment: dict[str, Any],
    telemetry_df: pd.DataFrame,
    diagnostic_result: dict[str, Any],
) -> dict[str, Any]:
    """Forecast remaining useful life using a linear degradation trend."""

    degradation_series = diagnostic_result.get("degradation_series")
    if degradation_series is None or degradation_series.empty:
        degradation_series = build_degradation_time_series(telemetry_df)

    if degradation_series.empty or len(degradation_series) < 2:
        predicted_rul_hours = LARGE_RUL_HOURS
        confidence_score = 0.25
        slope = 0.0
        intercept = float(diagnostic_result.get("degradation_index", 0.0))
    else:
        hours = (
            degradation_series["measured_at"] - degradation_series["measured_at"].iloc[0]
        ).dt.total_seconds() / 3600.0
        y_values = degradation_series["degradation_index"].to_numpy(dtype=float)
        slope, intercept = np.polyfit(hours.to_numpy(dtype=float), y_values, 1)

        last_hour = float(hours.iloc[-1])
        if slope <= 0:
            predicted_rul_hours = LARGE_RUL_HOURS
        else:
            critical_time = (1.0 - intercept) / slope
            predicted_rul_hours = max(0.0, critical_time - last_hour)

        predicted = slope * hours.to_numpy(dtype=float) + intercept
        residual_sum = float(np.square(y_values - predicted).sum())
        total_sum = float(np.square(y_values - y_values.mean()).sum())
        r_squared = 1.0 - residual_sum / total_sum if total_sum > 0 else 0.5
        data_coverage = float(np.clip(len(degradation_series) / (30 * 24), 0.0, 1.0))
        confidence_score = float(
            np.clip(0.20 + 0.50 * max(r_squared, 0.0) + 0.30 * data_coverage, 0.05, 0.99)
        )

    lower_bound_hours = max(0.0, predicted_rul_hours * 0.8)
    upper_bound_hours = predicted_rul_hours * 1.2

    return {
        "equipment_id": int(equipment["equipment_id"]),
        "diagnostic_report_id": diagnostic_result.get("diagnostic_report_id"),
        "predicted_rul_hours": float(predicted_rul_hours),
        "lower_bound_hours": float(lower_bound_hours),
        "upper_bound_hours": float(upper_bound_hours),
        "confidence_score": float(confidence_score),
        "model_name": RUL_MODEL_NAME,
        "model_version": RUL_MODEL_VERSION,
        "trend_slope": float(slope),
        "trend_intercept": float(intercept),
    }
