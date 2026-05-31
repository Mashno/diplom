from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from cmapss_preprocessing import build_training_dataset
from config import DEFAULT_CMAPSS_DATASET_ID, get_cmapss_model_path


def _build_candidate_models() -> dict[str, Pipeline]:
    """Create the candidate pipelines used in the MVP."""

    return {
        "linear_regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", LinearRegression()),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=100,
                        random_state=42,
                        n_jobs=-1,
                    ),
                )
            ]
        ),
    }


def _calculate_metrics(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Calculate regression metrics used in the report."""

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
    }


def train_rul_model(
    train_df: pd.DataFrame,
    model_key: str = "random_forest",
    training_dataset_id: str = DEFAULT_CMAPSS_DATASET_ID,
) -> dict[str, Any]:
    """Train C-MAPSS RUL models and return the selected trained artifact."""

    features, target, feature_columns = build_training_dataset(train_df)
    candidate_models = _build_candidate_models()
    if model_key not in candidate_models:
        raise ValueError(f"Unsupported model_key: {model_key}")

    trained_models: dict[str, Pipeline] = {}
    training_metrics: dict[str, dict[str, float]] = {}
    model_names = {
        "linear_regression": "LinearRegression",
        "random_forest": "RandomForestRegressor",
    }

    for candidate_key, pipeline in candidate_models.items():
        pipeline.fit(features, target)
        predictions = np.maximum(pipeline.predict(features), 0.0)
        trained_models[candidate_key] = pipeline
        training_metrics[candidate_key] = _calculate_metrics(target, predictions)

    return {
        "pipeline": trained_models[model_key],
        "model_key": model_key,
        "model_name": model_names[model_key],
        "feature_columns": feature_columns,
        "training_metrics": training_metrics,
        "training_dataset_id": training_dataset_id.upper(),
        "training_unit_count": int(train_df["unit_number"].nunique()),
        "training_row_count": int(len(train_df)),
    }


def predict_rul_for_latest_cycles(
    model: dict[str, Any],
    test_df: pd.DataFrame,
) -> pd.DataFrame:
    """Predict RUL for the latest available cycle of each test engine."""

    feature_columns = model["feature_columns"]
    latest_cycles = (
        test_df.sort_values(["unit_number", "time_in_cycles"])
        .groupby("unit_number", as_index=False)
        .tail(1)
        .copy()
    )
    predictions = model["pipeline"].predict(latest_cycles.loc[:, feature_columns])
    latest_cycles["predicted_rul"] = np.maximum(predictions, 0.0)
    return latest_cycles[["unit_number", "time_in_cycles", "predicted_rul"]].reset_index(drop=True)


def evaluate_on_test(
    model: dict[str, Any],
    test_df: pd.DataFrame,
    rul_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Evaluate a trained model on the official C-MAPSS test split."""

    predictions_df = predict_rul_for_latest_cycles(model, test_df)
    evaluation_df = predictions_df.merge(rul_df, on="unit_number", how="inner")
    metrics = _calculate_metrics(
        evaluation_df["true_rul"].to_numpy(dtype=float),
        evaluation_df["predicted_rul"].to_numpy(dtype=float),
    )
    return evaluation_df, metrics


def save_model(
    model: dict[str, Any],
    path: str | Path | None = None,
    dataset_id: str = DEFAULT_CMAPSS_DATASET_ID,
) -> Path:
    """Persist the trained model artifact to disk."""

    model_path = Path(path) if path else get_cmapss_model_path(dataset_id)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model["model_path"] = str(model_path)
    joblib.dump(model, model_path)
    return model_path


def load_model(
    path: str | Path | None = None,
    dataset_id: str = DEFAULT_CMAPSS_DATASET_ID,
) -> dict[str, Any]:
    """Load a previously saved model artifact from disk."""

    model_path = Path(path) if path else get_cmapss_model_path(dataset_id)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model_artifact = joblib.load(model_path)
    model_artifact.setdefault("training_dataset_id", dataset_id.upper())
    model_artifact.setdefault("model_path", str(model_path))
    return model_artifact
