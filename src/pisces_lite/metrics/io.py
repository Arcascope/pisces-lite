"""Load metrics CSVs in either format; convert between long and wide."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pandas as pd

from pisces_lite.metrics.logger import IDENTITY_COLUMNS


def _infer_identity_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in IDENTITY_COLUMNS if c in df.columns]


def load_long(
    csv_path: "Path | str",
    *,
    metric_column: str = "metric",
    value_column: str = "value",
    identity_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Read the CSV and return in long format, pivoting from wide if needed."""
    df = pd.read_csv(csv_path)
    if metric_column in df.columns and value_column in df.columns:
        return df
    return to_long(
        df,
        metric_column=metric_column,
        value_column=value_column,
        identity_columns=identity_columns,
    )


def load_wide(
    csv_path: "Path | str",
    *,
    metric_column: str = "metric",
    value_column: str = "value",
    identity_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Read the CSV and return in wide format, pivoting from long if needed."""
    df = pd.read_csv(csv_path)
    if metric_column in df.columns and value_column in df.columns:
        return to_wide(
            df,
            metric_column=metric_column,
            value_column=value_column,
            identity_columns=identity_columns,
        )
    return df


def to_long(
    df: pd.DataFrame,
    *,
    metric_column: str = "metric",
    value_column: str = "value",
    identity_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Melt a wide DataFrame to long."""
    id_cols = identity_columns or _infer_identity_columns(df)
    value_cols = [c for c in df.columns if c not in id_cols]
    return df.melt(
        id_vars=id_cols,
        value_vars=value_cols,
        var_name=metric_column,
        value_name=value_column,
    )


def to_wide(
    df: pd.DataFrame,
    *,
    metric_column: str = "metric",
    value_column: str = "value",
    identity_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Pivot a long DataFrame to wide, with one row per identity tuple."""
    id_cols = identity_columns or _infer_identity_columns(df)
    wide = df.pivot_table(
        index=id_cols,
        columns=metric_column,
        values=value_column,
        aggfunc="first",
    ).reset_index()
    wide.columns.name = None
    return wide
