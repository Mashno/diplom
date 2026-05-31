from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = PROJECT_ROOT / "datasets"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

DEFAULT_CMAPSS_DATASET_ID = "FD001"
DEFAULT_CMAPSS_EXPERIMENT_ID = 1
CMAPSS_BASE_TIME = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
CMAPSS_SOURCE_SYSTEM_PREFIX = "cmapss"
CMAPSS_QUALITY_CODE_ID = 1
CMAPSS_RUL_CAP = 125

CMAPSS_OPERATIONAL_COLUMNS = [
    "operational_setting_1",
    "operational_setting_2",
    "operational_setting_3",
]
CMAPSS_SENSOR_COLUMNS = [f"sensor_{index}" for index in range(1, 22)]
CMAPSS_FEATURE_COLUMNS = [
    *CMAPSS_OPERATIONAL_COLUMNS,
    *CMAPSS_SENSOR_COLUMNS,
    "time_in_cycles",
]
CMAPSS_DATA_COLUMNS = [
    "unit_number",
    "time_in_cycles",
    *CMAPSS_OPERATIONAL_COLUMNS,
    *CMAPSS_SENSOR_COLUMNS,
]


@dataclass(frozen=True)
class DatabaseSettings:
    """Database connection settings loaded from the environment."""

    host: str
    port: int
    dbname: str
    user: str
    password: str
    connect_timeout: int = 10
    sslmode: str = "require"


@dataclass(frozen=True)
class CmapssExperiment:
    """Predefined experiment settings for the diploma MVP."""

    experiment_id: int
    title: str
    train_dataset_id: str
    test_dataset_id: str
    description: str


CMAPSS_EXPERIMENTS: dict[int, CmapssExperiment] = {
    1: CmapssExperiment(
        experiment_id=1,
        title="Эксперимент 1",
        train_dataset_id="FD001",
        test_dataset_id="FD001",
        description="Базовая проверка MVP на официальном разбиении FD001.",
    ),
    2: CmapssExperiment(
        experiment_id=2,
        title="Эксперимент 2",
        train_dataset_id="FD002",
        test_dataset_id="FD002",
        description="Проверка MVP на более сложном сценарии FD002.",
    ),
    3: CmapssExperiment(
        experiment_id=3,
        title="Эксперимент 3",
        train_dataset_id="FD002",
        test_dataset_id="FD001",
        description="Проверка переносимости модели, обученной на FD002, на тесте FD001.",
    ),
}


def load_database_settings() -> DatabaseSettings:
    """Load database settings from the project .env file."""

    load_dotenv(PROJECT_ROOT / ".env", override=False)

    required_keys = [
        "DB_HOST",
        "DB_PORT",
        "DB_NAME",
        "DB_USER",
        "DB_PASSWORD",
    ]
    missing_keys = [key for key in required_keys if not os.getenv(key)]
    if missing_keys:
        missing = ", ".join(missing_keys)
        raise ValueError(f"Missing database settings in .env: {missing}")

    return DatabaseSettings(
        host=os.environ["DB_HOST"],
        port=int(os.environ["DB_PORT"]),
        dbname=os.environ["DB_NAME"],
        user=os.environ["DB_USER"],
        password=os.environ["DB_PASSWORD"],
        connect_timeout=int(os.getenv("DB_CONNECT_TIMEOUT", "10")),
        sslmode="require",
    )


def ensure_reports_dir() -> Path:
    """Create the reports directory if it does not exist."""

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    return REPORTS_DIR


def ensure_models_dir() -> Path:
    """Create the models directory if it does not exist."""

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    return MODELS_DIR


def get_cmapss_experiment(experiment_id: int) -> CmapssExperiment:
    """Return one of the predefined C-MAPSS experiment presets."""

    try:
        return CMAPSS_EXPERIMENTS[int(experiment_id)]
    except (KeyError, ValueError) as exc:
        raise ValueError(f"Unsupported experiment_id: {experiment_id}") from exc


def get_cmapss_source_system(dataset_id: str = DEFAULT_CMAPSS_DATASET_ID) -> str:
    """Build the source_system marker used for C-MAPSS measurements."""

    return f"{CMAPSS_SOURCE_SYSTEM_PREFIX}_{dataset_id.lower()}"


def get_cmapss_model_version(training_dataset_id: str = DEFAULT_CMAPSS_DATASET_ID) -> str:
    """Return a readable version marker for a model trained on one dataset."""

    return f"CMAPSS-{training_dataset_id.upper()}-MVP-1.0"


def get_cmapss_model_path(dataset_id: str = DEFAULT_CMAPSS_DATASET_ID) -> Path:
    """Return the default model path for a training dataset."""

    ensure_models_dir()
    return MODELS_DIR / f"rul_model_{dataset_id.lower()}.joblib"


def get_cmapss_report_path(experiment_id: int | None = None) -> Path:
    """Return the report path for the latest run or a specific experiment."""

    ensure_reports_dir()
    if experiment_id is None:
        return REPORTS_DIR / "cmapss_mvp_report.txt"
    return REPORTS_DIR / f"cmapss_mvp_report_experiment_{int(experiment_id)}.txt"


def get_cmapss_dataset_prefix(dataset_id: str = DEFAULT_CMAPSS_DATASET_ID) -> str:
    """Return the equipment code prefix for a dataset."""

    return f"CMAPSS-{dataset_id.upper()}-"
