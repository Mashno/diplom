from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import CMAPSS_DATA_COLUMNS, DATASETS_DIR, DEFAULT_CMAPSS_DATASET_ID


def _resolve_dataset_path(file_name: str) -> Path:
    """Resolve a file path inside the datasets directory."""

    file_path = DATASETS_DIR / file_name
    if not file_path.exists():
        raise FileNotFoundError(
            f"C-MAPSS file '{file_name}' was not found in {DATASETS_DIR}."
        )
    return file_path


def _load_cmapss_table(file_path: Path) -> pd.DataFrame:
    """Load a raw C-MAPSS train or test file with trailing spaces handled safely."""

    dataframe = pd.read_csv(
        file_path,
        sep=r"\s+",
        header=None,
        names=CMAPSS_DATA_COLUMNS,
        usecols=range(len(CMAPSS_DATA_COLUMNS)),
        engine="python",
    )
    dataframe["unit_number"] = dataframe["unit_number"].astype(int)
    dataframe["time_in_cycles"] = dataframe["time_in_cycles"].astype(int)
    return dataframe


def _load_cmapss_rul_table(file_path: Path) -> pd.DataFrame:
    """Load a raw C-MAPSS RUL file."""

    rul_df = pd.read_csv(
        file_path,
        sep=r"\s+",
        header=None,
        names=["true_rul"],
        usecols=[0],
        engine="python",
    )
    rul_df.insert(0, "unit_number", range(1, len(rul_df) + 1))
    rul_df["unit_number"] = rul_df["unit_number"].astype(int)
    rul_df["true_rul"] = rul_df["true_rul"].astype(float)
    return rul_df


def load_cmapss_train(
    dataset_id: str = DEFAULT_CMAPSS_DATASET_ID,
    file_path: str | Path | None = None,
) -> pd.DataFrame:
    """Load train data and calculate the target RUL for each cycle."""

    resolved_path = Path(file_path) if file_path else _resolve_dataset_path(f"train_{dataset_id.upper()}.txt")
    if not resolved_path.exists():
        raise FileNotFoundError(f"C-MAPSS train file not found: {resolved_path}")

    train_df = _load_cmapss_table(resolved_path)
    max_cycles = train_df.groupby("unit_number")["time_in_cycles"].transform("max")
    train_df["RUL"] = max_cycles - train_df["time_in_cycles"]
    return train_df


def load_cmapss_test(
    dataset_id: str = DEFAULT_CMAPSS_DATASET_ID,
    file_path: str | Path | None = None,
) -> pd.DataFrame:
    """Load raw test trajectories for a C-MAPSS dataset."""

    resolved_path = Path(file_path) if file_path else _resolve_dataset_path(f"test_{dataset_id.upper()}.txt")
    if not resolved_path.exists():
        raise FileNotFoundError(f"C-MAPSS test file not found: {resolved_path}")
    return _load_cmapss_table(resolved_path)


def load_cmapss_rul(
    dataset_id: str = DEFAULT_CMAPSS_DATASET_ID,
    file_path: str | Path | None = None,
) -> pd.DataFrame:
    """Load the true RUL values for the last cycle of each test unit."""

    resolved_path = Path(file_path) if file_path else _resolve_dataset_path(f"RUL_{dataset_id.upper()}.txt")
    if not resolved_path.exists():
        raise FileNotFoundError(f"C-MAPSS RUL file not found: {resolved_path}")
    return _load_cmapss_rul_table(resolved_path)
