from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from config import (
    DEFAULT_QUALITY_CODE_ID,
    MEASUREMENT_FREQUENCY_HOURS,
    SCENARIO_DAYS,
    SYNTHETIC_SOURCE_SYSTEM,
)


CRITICALITY_FACTORS = {
    "low": 0.35,
    "medium": 0.55,
    "high": 0.82,
    "critical": 1.00,
}


def _as_float(value: Any, fallback: float) -> float:
    """Convert database numeric values into floats."""

    if value is None:
        return fallback
    return float(value)


def _build_parameter_series(
    equipment_id: int,
    criticality_level: str,
    parameter: dict[str, Any],
    timeline: pd.DatetimeIndex,
) -> np.ndarray:
    """Generate one telemetry series with a controlled degradation trend."""

    severity = CRITICALITY_FACTORS.get(criticality_level, 0.50)
    parameter_code = parameter["parameter_code"]

    normal_min = _as_float(parameter["normal_min"], 0.0)
    normal_max = _as_float(parameter["normal_max"], max(normal_min + 1.0, 1.0))
    critical_min = _as_float(parameter["critical_min"], normal_min)
    critical_max = _as_float(parameter["critical_max"], normal_max * 1.25)

    periods = len(timeline)
    hours = np.arange(periods, dtype=float)
    progress = np.linspace(0.0, 1.0, periods)
    daily_wave = np.sin(2.0 * np.pi * hours / 24.0)
    weekly_wave = np.sin(2.0 * np.pi * hours / (24.0 * 7.0))
    rng = np.random.default_rng(seed=equipment_id * 100 + int(parameter["parameter_id"]))

    if parameter_code == "temperature":
        start = normal_max * 0.70 + 4.0 * severity
        end = normal_max + (critical_max - normal_max) * (0.35 + 0.50 * severity)
        noise = rng.normal(0.0, 1.2, periods)
        values = start + (end - start) * progress + 1.7 * daily_wave + 0.8 * weekly_wave + noise
    elif parameter_code == "vibration":
        start = max(normal_max * 0.24, 0.8) + 0.20 * severity
        end = normal_max + (critical_max - normal_max) * (0.30 + 0.55 * severity)
        noise = rng.normal(0.0, 0.18, periods)
        values = start + (end - start) * progress + 0.25 * daily_wave + 0.10 * weekly_wave + noise
    elif parameter_code == "current":
        start = normal_max * 0.60 + 8.0 * severity
        end = normal_max + (critical_max - normal_max) * (0.25 + 0.60 * severity)
        noise = rng.normal(0.0, 2.2, periods)
        values = start + (end - start) * progress + 3.2 * daily_wave + 1.5 * weekly_wave + noise
    elif parameter_code == "load":
        centre = normal_max * 0.78 + 3.0 * severity
        trend = 2.5 * severity * progress
        noise = rng.normal(0.0, 1.8, periods)
        values = centre + trend + 4.5 * daily_wave + 2.2 * weekly_wave + noise
    else:
        span = max(critical_max - normal_min, 1.0)
        start = normal_min + 0.55 * span
        end = start + 0.10 * span * severity
        noise = rng.normal(0.0, 0.05 * span, periods)
        values = start + (end - start) * progress + noise

    return np.clip(values, critical_min, critical_max * 0.995)


def generate_synthetic_telemetry(
    equipment_records: list[dict[str, Any]],
    parameter_records: list[dict[str, Any]],
    days: int = SCENARIO_DAYS,
    end_time: datetime | None = None,
) -> pd.DataFrame:
    """Generate hourly synthetic telemetry for the requested equipment list."""

    if not equipment_records or not parameter_records:
        return pd.DataFrame(
            columns=[
                "measured_at",
                "equipment_id",
                "parameter_id",
                "quality_code_id",
                "value",
                "source_system",
            ]
        )

    scenario_end = end_time or datetime.now(timezone.utc)
    scenario_end = pd.Timestamp(scenario_end)
    if scenario_end.tzinfo is None:
        scenario_end = scenario_end.tz_localize("UTC")
    else:
        scenario_end = scenario_end.tz_convert("UTC")
    scenario_end = scenario_end.floor("h")
    periods = days * 24 // MEASUREMENT_FREQUENCY_HOURS
    timeline = pd.date_range(
        end=scenario_end,
        periods=periods,
        freq=f"{MEASUREMENT_FREQUENCY_HOURS}h",
        tz="UTC",
    )

    frames: list[pd.DataFrame] = []
    for equipment in equipment_records:
        equipment_id = int(equipment["equipment_id"])
        criticality_level = str(equipment["criticality_level"])
        for parameter in parameter_records:
            values = _build_parameter_series(
                equipment_id=equipment_id,
                criticality_level=criticality_level,
                parameter=parameter,
                timeline=timeline,
            )
            frames.append(
                pd.DataFrame(
                    {
                        "measured_at": timeline,
                        "equipment_id": equipment_id,
                        "parameter_id": int(parameter["parameter_id"]),
                        "quality_code_id": DEFAULT_QUALITY_CODE_ID,
                        "value": np.round(values, 4),
                        "source_system": SYNTHETIC_SOURCE_SYSTEM,
                    }
                )
            )

    telemetry = pd.concat(frames, ignore_index=True)
    telemetry.sort_values(
        by=["equipment_id", "parameter_id", "measured_at"],
        inplace=True,
        ignore_index=True,
    )
    return telemetry
