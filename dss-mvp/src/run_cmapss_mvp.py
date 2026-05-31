from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from cmapss_loader import load_cmapss_rul, load_cmapss_test, load_cmapss_train
from cmapss_preprocessing import convert_train_to_telemetry_rows
from config import (
    CMAPSS_EXPERIMENTS,
    DEFAULT_CMAPSS_DATASET_ID,
    DEFAULT_CMAPSS_EXPERIMENT_ID,
    PROJECT_ROOT,
    get_cmapss_experiment,
    get_cmapss_model_version,
)
from database import check_connection, get_connection
from diagnostics import build_diagnostic_result
from recommendations import save_recommendation
from report_builder import build_cmapss_report
from repositories import (
    delete_previous_cmapss_measurements,
    get_cmapss_measurement_count,
    get_latest_results,
    insert_diagnostic_report,
    insert_rul_forecast,
    insert_telemetry_measurements,
    upsert_cmapss_equipment,
    upsert_cmapss_telemetry_parameters,
)
from rul_model import evaluate_on_test, load_model, predict_rul_for_latest_cycles, save_model, train_rul_model


@dataclass(frozen=True)
class RunSettings:
    """Resolved runtime settings for one MVP scenario run."""

    experiment_id: int | None
    experiment_title: str
    experiment_description: str
    train_dataset_id: str
    test_dataset_id: str
    train_file: str | None
    test_file: str | None
    rul_file: str | None


def _confidence_from_r2(r2_value: float | None) -> float:
    """Convert an R2 score into a coarse confidence estimate."""

    if r2_value is None:
        return 0.75
    if pd.isna(r2_value):
        return 0.75
    return max(0.5, min(0.95, float(r2_value)))


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the C-MAPSS MVP scenario."""

    parser = argparse.ArgumentParser(
        description="Run one of the predefined C-MAPSS MVP experiments.",
    )
    parser.add_argument(
        "--experiment",
        type=int,
        choices=sorted(CMAPSS_EXPERIMENTS.keys()),
        default=DEFAULT_CMAPSS_EXPERIMENT_ID,
        help="Predefined experiment mode: 1=FD001/FD001, 2=FD002/FD002, 3=FD002/FD001.",
    )
    parser.add_argument(
        "--dataset-id",
        help="Legacy shorthand: use the same dataset for training and testing.",
    )
    parser.add_argument("--train-dataset-id")
    parser.add_argument("--test-dataset-id")
    parser.add_argument("--train-file")
    parser.add_argument("--test-file")
    parser.add_argument("--rul-file")
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Force model retraining even if a saved artifact already exists.",
    )
    parser.add_argument(
        "--reload-train-telemetry",
        action="store_true",
        help="Reload train telemetry into PostgreSQL for the selected training dataset.",
    )
    return parser.parse_args()


def _resolve_run_settings(args: argparse.Namespace) -> RunSettings:
    """Resolve the requested run mode into explicit train/test settings."""

    manual_train_dataset_id = args.train_dataset_id or args.dataset_id
    manual_test_dataset_id = args.test_dataset_id or args.dataset_id

    if manual_train_dataset_id or manual_test_dataset_id:
        train_dataset_id = str(manual_train_dataset_id or DEFAULT_CMAPSS_DATASET_ID).upper()
        test_dataset_id = str(manual_test_dataset_id or train_dataset_id).upper()
        return RunSettings(
            experiment_id=None,
            experiment_title="Пользовательский режим",
            experiment_description="Пользователь вручную задал наборы данных для обучения и тестирования.",
            train_dataset_id=train_dataset_id,
            test_dataset_id=test_dataset_id,
            train_file=args.train_file,
            test_file=args.test_file,
            rul_file=args.rul_file,
        )

    experiment = get_cmapss_experiment(args.experiment)
    return RunSettings(
        experiment_id=experiment.experiment_id,
        experiment_title=experiment.title,
        experiment_description=experiment.description,
        train_dataset_id=experiment.train_dataset_id,
        test_dataset_id=experiment.test_dataset_id,
        train_file=args.train_file,
        test_file=args.test_file,
        rul_file=args.rul_file,
    )


def _load_or_train_model(
    train_dataset_id: str,
    retrain: bool,
    train_file: str | None,
    reload_train_telemetry: bool,
) -> tuple[dict, pd.DataFrame | None, bool]:
    """Load a saved model or train a new one when needed."""

    model_artifact = None
    train_df: pd.DataFrame | None = None
    trained_now = False

    try:
        if retrain:
            raise FileNotFoundError
        model_artifact = load_model(dataset_id=train_dataset_id)
    except Exception:
        train_df = load_cmapss_train(dataset_id=train_dataset_id, file_path=train_file)
        model_artifact = train_rul_model(
            train_df,
            training_dataset_id=train_dataset_id,
        )
        model_path = save_model(model_artifact, dataset_id=train_dataset_id)
        model_artifact["model_path"] = str(model_path)
        trained_now = True

    metadata_missing = (
        model_artifact.get("training_unit_count") is None
        or model_artifact.get("training_row_count") is None
        or model_artifact.get("training_dataset_id") is None
    )
    if train_df is None and (reload_train_telemetry or trained_now or metadata_missing):
        train_df = load_cmapss_train(dataset_id=train_dataset_id, file_path=train_file)

    if metadata_missing and train_df is not None:
        model_artifact["training_dataset_id"] = train_dataset_id.upper()
        model_artifact["training_unit_count"] = int(train_df["unit_number"].nunique())
        model_artifact["training_row_count"] = int(len(train_df))
        save_model(model_artifact, dataset_id=train_dataset_id)

    return model_artifact, train_df, trained_now


def _should_load_default_rul(test_file: str | None, rul_file: str | None) -> bool:
    """Return True when the official RUL file should be loaded automatically."""

    return rul_file is not None or test_file is None


def _upsert_equipment_maps(
    connection,
    train_dataset_id: str,
    test_dataset_id: str,
    train_units: list[int],
    test_units: list[int],
    need_train_equipment: bool,
) -> tuple[dict[int, dict], dict[int, dict]]:
    """Create or update equipment records for train and test datasets."""

    train_equipment_map: dict[int, dict] = {}
    test_equipment_map: dict[int, dict] = {}

    if train_dataset_id == test_dataset_id:
        required_units = sorted(set(test_units) | (set(train_units) if need_train_equipment else set()))
        shared_map = upsert_cmapss_equipment(connection, required_units, test_dataset_id)
        if need_train_equipment:
            train_equipment_map = {unit_number: shared_map[unit_number] for unit_number in train_units}
        test_equipment_map = {unit_number: shared_map[unit_number] for unit_number in test_units}
        return train_equipment_map, test_equipment_map

    if need_train_equipment:
        train_equipment_map = upsert_cmapss_equipment(connection, train_units, train_dataset_id)
    test_equipment_map = upsert_cmapss_equipment(connection, test_units, test_dataset_id)
    return train_equipment_map, test_equipment_map


def main() -> int:
    """Execute the C-MAPSS MVP scenario using a saved or freshly trained model."""

    args = _parse_args()
    settings = _resolve_run_settings(args)
    run_started_at = datetime.now(timezone.utc)

    try:
        check_connection()
        print("Подключение к базе данных успешно.")
        print(f"{settings.experiment_title}.")
        print(f"Обучение: {settings.train_dataset_id} -> Тестирование: {settings.test_dataset_id}.")

        model_artifact, train_df, trained_now = _load_or_train_model(
            train_dataset_id=settings.train_dataset_id,
            retrain=bool(args.retrain),
            train_file=settings.train_file,
            reload_train_telemetry=bool(args.reload_train_telemetry),
        )
        if trained_now:
            print(
                f"Модель RUL обучена и сохранена для {settings.train_dataset_id}: "
                f"{model_artifact['model_name']}."
            )
        else:
            print(
                f"Используется сохранённая модель для {settings.train_dataset_id}: "
                f"{model_artifact['model_name']}."
            )

        test_df = load_cmapss_test(
            dataset_id=settings.test_dataset_id,
            file_path=settings.test_file,
        )
        rul_df = (
            load_cmapss_rul(dataset_id=settings.test_dataset_id, file_path=settings.rul_file)
            if _should_load_default_rul(settings.test_file, settings.rul_file)
            else None
        )

        train_unit_count = int(model_artifact.get("training_unit_count") or 0)
        if train_unit_count == 0 and train_df is not None:
            train_unit_count = int(train_df["unit_number"].nunique())
        test_units = sorted(test_df["unit_number"].astype(int).unique().tolist())
        train_units = (
            sorted(train_df["unit_number"].astype(int).unique().tolist())
            if train_df is not None
            else []
        )

        print(f"Обучающих двигателей: {train_unit_count}.")
        print(f"Тестовых двигателей: {len(test_units)}.")

        need_train_equipment = train_df is not None and (args.reload_train_telemetry or trained_now)
        loaded_measurements_count = 0
        with get_connection() as connection:
            train_equipment_map, test_equipment_map = _upsert_equipment_maps(
                connection=connection,
                train_dataset_id=settings.train_dataset_id,
                test_dataset_id=settings.test_dataset_id,
                train_units=train_units,
                test_units=test_units,
                need_train_equipment=need_train_equipment,
            )
            parameter_id_map = upsert_cmapss_telemetry_parameters(connection)

            if need_train_equipment:
                delete_previous_cmapss_measurements(connection, settings.train_dataset_id)
                telemetry_rows = convert_train_to_telemetry_rows(
                    train_df=train_df,
                    equipment_id_map={
                        unit_number: int(record["equipment_id"])
                        for unit_number, record in train_equipment_map.items()
                    },
                    parameter_id_map=parameter_id_map,
                    dataset_id=settings.train_dataset_id,
                )
                loaded_measurements_count = insert_telemetry_measurements(connection, telemetry_rows)
                print(
                    f"Загружено train-телеметрических записей "
                    f"({settings.train_dataset_id}): {loaded_measurements_count}."
                )
            else:
                loaded_measurements_count = get_cmapss_measurement_count(connection, settings.train_dataset_id)
                print(
                    f"Переиспользуется ранее загруженная train-телеметрия "
                    f"({settings.train_dataset_id})."
                )
                print(
                    f"Доступно train-телеметрических записей в БД: "
                    f"{loaded_measurements_count}."
                )

            if rul_df is not None:
                evaluation_df, metrics = evaluate_on_test(model_artifact, test_df, rul_df)
            else:
                predictions_df = predict_rul_for_latest_cycles(model_artifact, test_df)
                evaluation_df = predictions_df.copy()
                evaluation_df["true_rul"] = pd.NA
                metrics = {"mae": float("nan"), "rmse": float("nan"), "r2": float("nan")}

            if rul_df is not None:
                print(f"MAE: {metrics['mae']:.4f}")
                print(f"RMSE: {metrics['rmse']:.4f}")
                print(f"R2: {metrics['r2']:.4f}")
            else:
                print("Истинные значения RUL не переданы: выполняется только inference без метрик качества.")

            confidence_score = _confidence_from_r2(metrics.get("r2"))
            model_version = get_cmapss_model_version(settings.train_dataset_id)

            for row in evaluation_df.itertuples(index=False):
                unit_number = int(row.unit_number)
                equipment = test_equipment_map[unit_number]
                predicted_rul = float(row.predicted_rul)

                diagnostic_result = build_diagnostic_result(equipment, predicted_rul)
                diagnostic_record = insert_diagnostic_report(
                    connection=connection,
                    equipment_id=int(equipment["equipment_id"]),
                    technical_state=diagnostic_result["technical_state"],
                    degradation_index=diagnostic_result["degradation_index"],
                    anomaly_score=diagnostic_result["anomaly_score"],
                    summary=diagnostic_result["summary"],
                )

                rul_record = insert_rul_forecast(
                    connection=connection,
                    equipment_id=int(equipment["equipment_id"]),
                    diagnostic_report_id=int(diagnostic_record["diagnostic_report_id"]),
                    predicted_rul_hours=predicted_rul,
                    lower_bound_hours=predicted_rul * 0.8,
                    upper_bound_hours=predicted_rul * 1.2,
                    confidence_score=confidence_score,
                    model_name=str(model_artifact["model_name"]),
                    model_version=model_version,
                )

                save_recommendation(
                    connection=connection,
                    equipment_id=int(equipment["equipment_id"]),
                    diagnostic_report_id=int(diagnostic_record["diagnostic_report_id"]),
                    rul_forecast_id=int(rul_record["rul_forecast_id"]),
                    technical_state=str(diagnostic_result["technical_state"]),
                    predicted_rul=predicted_rul,
                    base_date=date.today(),
                )

            latest_results = get_latest_results(
                connection,
                dataset_id=settings.test_dataset_id,
                equipment_codes=[record["equipment_code"] for record in test_equipment_map.values()],
            )

        latest_results_df = pd.DataFrame(latest_results)
        unit_to_code_df = pd.DataFrame(
            {
                "unit_number": list(test_equipment_map.keys()),
                "equipment_code": [record["equipment_code"] for record in test_equipment_map.values()],
            }
        )
        evaluation_report_df = evaluation_df.merge(unit_to_code_df, on="unit_number", how="left")
        report_df = latest_results_df.merge(
            evaluation_report_df[["equipment_code", "unit_number", "true_rul", "predicted_rul"]],
            on="equipment_code",
            how="left",
        )

        report_path = build_cmapss_report(
            run_started_at=run_started_at,
            experiment_id=settings.experiment_id,
            experiment_title=settings.experiment_title,
            experiment_description=settings.experiment_description,
            train_dataset_id=settings.train_dataset_id,
            test_dataset_id=settings.test_dataset_id,
            train_unit_count=train_unit_count,
            test_unit_count=len(test_units),
            loaded_measurements_count=loaded_measurements_count,
            model_name=str(model_artifact["model_name"]),
            metrics=metrics,
            results_df=report_df,
        )

        preview_df = report_df.sort_values("equipment_code").head(5)
        for row in preview_df.itertuples(index=False):
            print()
            print(row.equipment_code)
            if pd.notna(row.true_rul):
                print(f"Истинный RUL: {row.true_rul:.1f}")
            print(f"Прогноз RUL: {row.predicted_rul:.1f}")
            print(f"Состояние: {row.technical_state}")
            print(f"Приоритет: {row.priority}")
            print(f"Рекомендация: {row.recommendation_text}")

        relative_path = report_path.relative_to(PROJECT_ROOT)
        print()
        print(f"Отчёт сохранён: {relative_path}")
        return 0
    except Exception as exc:
        print(f"Сценарий завершился с ошибкой: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
