from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from config import get_cmapss_report_path


FINAL_CONCLUSION = (
    "Результаты проверки показывают, что реализованный MVP обеспечивает загрузку "
    "открытых телеметрических данных C-MAPSS, обучение модели прогнозирования "
    "остаточного ресурса, сохранение результатов прогноза в прикладном хранилище и "
    "формирование рекомендаций по обслуживанию оборудования. Следовательно, "
    "реализованный прототип подтверждает работоспособность ключевого аналитического "
    "контура СППР."
)


def _format_metric(value: float) -> str:
    """Format a metric value or return n/a when it is unavailable."""

    if pd.isna(value):
        return "n/a"
    return f"{float(value):.4f}"


def build_cmapss_report(
    run_started_at: datetime,
    train_dataset_id: str,
    test_dataset_id: str,
    train_unit_count: int,
    test_unit_count: int,
    loaded_measurements_count: int,
    model_name: str,
    metrics: dict[str, float],
    results_df: pd.DataFrame,
    experiment_id: int | None = None,
    experiment_title: str | None = None,
    experiment_description: str | None = None,
) -> Path:
    """Build and save the final C-MAPSS MVP text report."""

    report_path = get_cmapss_report_path(experiment_id)
    latest_report_path = get_cmapss_report_path()
    report_rows = results_df.copy().sort_values("equipment_code").reset_index(drop=True)

    lines = [
        "CMAPSS MVP REPORT",
        f"Дата запуска: {run_started_at.astimezone().isoformat()}",
    ]
    if experiment_title:
        lines.append(f"Режим запуска: {experiment_title}")
    if experiment_description:
        lines.append(f"Описание режима: {experiment_description}")

    lines.extend(
        [
            f"Обучающий датасет: {train_dataset_id}",
            f"Тестовый датасет: {test_dataset_id}",
            f"Количество обучающих двигателей: {train_unit_count}",
            f"Количество тестовых двигателей: {test_unit_count}",
            f"Количество загруженных train-телеметрических записей: {loaded_measurements_count}",
            f"Используемая модель: {model_name}",
            f"MAE: {_format_metric(metrics['mae'])}",
            f"RMSE: {_format_metric(metrics['rmse'])}",
            f"R2: {_format_metric(metrics['r2'])}",
            "",
            "Примеры результатов по тестовым двигателям:",
        ]
    )

    display_columns = [
        "equipment_code",
        "true_rul",
        "predicted_rul",
        "recommendation_text",
    ]
    for row in report_rows.loc[:, display_columns].head(15).itertuples(index=False):
        true_rul_text = f"{row.true_rul:.2f}" if pd.notna(row.true_rul) else "n/a"
        lines.extend(
            [
                f"- {row.equipment_code}",
                f"  true_rul: {true_rul_text}",
                f"  predicted_rul: {row.predicted_rul:.2f}",
                f"  recommendation_text: {row.recommendation_text}",
            ]
        )

    lines.extend(
        [
            "",
            "Примечание: в рамках MVP один цикл C-MAPSS интерпретируется как один час работы оборудования.",
            "",
            "Общий вывод:",
            FINAL_CONCLUSION,
        ]
    )

    report_text = "\n".join(lines)
    report_path.write_text(report_text, encoding="utf-8-sig")
    if report_path != latest_report_path:
        latest_report_path.write_text(report_text, encoding="utf-8-sig")
    return report_path
