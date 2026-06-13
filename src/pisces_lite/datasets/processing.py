"""Data processing functions for PSG and accelerometer data.

Ported from ``pisces2.data_sets.data_processing`` — pure numpy/pandas/scipy.
The dataset-name-sniffing hacks from ``pisces2.processing.load_data`` are NOT
ported here; they belong in per-dataset ``data_set.json`` and are applied by
the loader in :mod:`pisces_lite.datasets.data_set_object`.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from pisces_lite.datasets.constants import (
    MINIMUM_ACCEL_SAMPLES_PER_PSG,
    PSG_COL,
    PSG_DT,
    PSG_MASK,
    SLEEP_CLASS_LABEL,
    TIMESTAMP_COL,
    WAKE_CLASS_LABEL,
    X_COL,
    Y_COL,
    Z_COL,
)


def _regrid_psg_uniform(
    psg_data: pd.DataFrame,
    start_time: float,
    end_time: float,
    timestamp_col: str,
    psg_col: str,
    psg_dt: int,
) -> pd.DataFrame:
    """
    Reindex PSG onto a uniform ``psg_dt`` grid spanning ``[start_time, end_time]``,
    filling any gap positions (epochs the source PSG did not score) with
    ``PSG_MASK``.

    This is needed because the accel feature pipeline (``ProcessingConfig.apply``
    via RegulariseNUFFTGrid) produces ``X`` on a uniform 30 s grid anchored at
    ``start_time``. Source-times PSG, however, can contain rows whose timestamps
    skip multiple epochs (real scoring gaps). Without re-gridding, positional
    indexing post-gap would land ``y[i]`` at a real time later than ``X[i]``,
    silently shifting labels relative to features.
    """
    if psg_data.empty:
        return psg_data
    epoch_count = int(round((float(end_time) - float(start_time)) / float(psg_dt))) + 1
    target_epoch_idx = np.arange(epoch_count, dtype=np.int64)
    source = psg_data.copy()
    source["_psg_epoch_idx"] = np.rint(
        (source[timestamp_col].astype(float) - float(start_time)) / float(psg_dt)
    ).astype(np.int64)
    gridded = (
        source.set_index("_psg_epoch_idx")
        .reindex(target_epoch_idx)
        .reset_index()
    )
    gridded[timestamp_col] = (
        float(start_time) + gridded["_psg_epoch_idx"].to_numpy(dtype=float) * float(psg_dt)
    ).astype(psg_data[timestamp_col].dtype)
    gridded.drop(columns=["_psg_epoch_idx"], inplace=True)
    gridded[psg_col] = gridded[psg_col].fillna(PSG_MASK).astype(int)
    return gridded


def psg_map(
    psg_data: pd.DataFrame,
    mapping_dict: Dict,
    psg_col: str = PSG_COL,
    psg_mask: int = PSG_MASK,
) -> pd.DataFrame:
    """Remap PSG stage values using ``mapping_dict``; unmapped → ``psg_mask``."""
    psg_data[psg_col] = (
        psg_data[psg_col].map(mapping_dict).fillna(psg_mask).astype(int)
    )
    return psg_data


def psg_to_sleep_wake(
    psg_data: pd.DataFrame,
    psg_col: str = PSG_COL,
    psg_mask: int = PSG_MASK,
    sleep_class_label: int = SLEEP_CLASS_LABEL,
    wake_class_label: int = WAKE_CLASS_LABEL,
) -> pd.DataFrame:
    """Binarise PSG stages: any value ≥1 → sleep, 0 → wake, mask preserved."""
    out = psg_data.copy()
    out[psg_col] = np.where(
        psg_data[psg_col] >= 1,
        sleep_class_label,
        np.where(psg_data[psg_col] == 0, wake_class_label, psg_mask),
    )
    return out


def align_trim(
    accelerometer_data: pd.DataFrame,
    psg_data: pd.DataFrame,
    timestamp_col: str = TIMESTAMP_COL,
    psg_dt: int = PSG_DT,
    psg_col: str = PSG_COL,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Trim both frames to their overlapping time range, snapped to PSG epochs.

    The returned PSG frame is also re-gridded onto a uniform ``psg_dt`` grid
    anchored at ``start_time``; any source-PSG scoring gap appears as one or
    more rows filled with ``PSG_MASK``. This guarantees that ``psg[i]``
    corresponds to wall-clock time ``start_time + i * psg_dt`` — which is the
    convention assumed by the downstream feature pipeline, which lays X out
    on the same uniform grid.
    """
    accelerometer_data = accelerometer_data.sort_values(by=timestamp_col)
    psg_data = psg_data.sort_values(by=timestamp_col)
    psg_phase = float(psg_data[timestamp_col].iloc[0]) % float(psg_dt)
    accel_start = accelerometer_data[timestamp_col].iloc[0]
    accel_end = accelerometer_data[timestamp_col].iloc[-1]
    start_time = max(
        psg_phase + psg_dt * np.ceil((accel_start - psg_phase) / psg_dt),
        psg_data[timestamp_col].iloc[0],
    )
    end_time = min(
        psg_phase + psg_dt * np.floor((accel_end - psg_phase) / psg_dt),
        psg_data[timestamp_col].iloc[-1],
    )
    psg_data = psg_data[
        (psg_data[timestamp_col] >= start_time) & (psg_data[timestamp_col] <= end_time)
    ]
    accelerometer_data = accelerometer_data[
        (accelerometer_data[timestamp_col] >= start_time)
        & (accelerometer_data[timestamp_col] <= end_time)
    ]
    psg_data = _regrid_psg_uniform(
        psg_data,
        start_time=start_time,
        end_time=end_time,
        timestamp_col=timestamp_col,
        psg_col=psg_col,
        psg_dt=psg_dt,
    )
    return accelerometer_data, psg_data


def mask_data(
    accelerometer_data: pd.DataFrame,
    psg_data: pd.DataFrame,
    timestamp_col: str = TIMESTAMP_COL,
    psg_dt: int = PSG_DT,
    minimum_accel_samples_per_psg: int = MINIMUM_ACCEL_SAMPLES_PER_PSG,
    psg_col: str = PSG_COL,
    mask_value: int = -2,
    extra_mask_minutes: float = 1.5,
) -> pd.DataFrame:
    """Mark PSG epochs with insufficient accelerometer coverage as ``mask_value``.

    Expands each flagged epoch by ``extra_mask_minutes`` before and after.
    """
    for tidx, psg_time in enumerate(psg_data[timestamp_col]):
        accel_window = accelerometer_data[
            (accelerometer_data[timestamp_col] >= psg_time)
            & (accelerometer_data[timestamp_col] < psg_time + psg_dt)
        ]
        if len(accel_window) < minimum_accel_samples_per_psg:
            extra_mask_idx = int(extra_mask_minutes * 60 // psg_dt)
            start = max(0, tidx - extra_mask_idx)
            end = min(len(psg_data) - 1, tidx + extra_mask_idx)
            psg_data.loc[start:end, psg_col] = mask_value
    return psg_data


def calculate_binary_activity(
    accelerometer_data: pd.DataFrame,
    threshold: float = 50.0,
) -> np.ndarray:
    """Z-scored squared-magnitude thresholding of accelerometer vector."""
    norm = np.sqrt(
        accelerometer_data[X_COL] ** 2
        + accelerometer_data[Y_COL] ** 2
        + accelerometer_data[Z_COL] ** 2
    )
    norm = (norm - norm.mean()) / norm.std()
    return np.where(norm**2 > threshold, 1, 0)


def rescale_spectrogram(
    Sxx: np.ndarray,
    axis: int = 0,
    epsilon: float = 1e-12,
) -> np.ndarray:
    """Divide by per-axis max; clamps denominator by ``epsilon``."""
    return Sxx / np.maximum(Sxx.max(axis=axis, keepdims=True), epsilon)
