"""Data processing functions for PSG and accelerometer data.

"""
from __future__ import annotations

import warnings
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
    # Several PSG rows can snap to the same epoch when scored finer than psg_dt
    # (or with jittered timestamps); keep the first so reindex sees unique labels.
    source = source.drop_duplicates(subset="_psg_epoch_idx", keep="first")
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
    """Remap PSG stage values using ``mapping_dict``; unmapped → ``psg_mask``.

    Returns a new frame; ``psg_data`` is left unchanged.
    """
    out = psg_data.copy()
    out[psg_col] = (
        out[psg_col].map(mapping_dict).fillna(psg_mask).astype(int)
    )
    return out


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

    If either frame is empty there is nothing to align, and if the two share
    no snapped epoch the trim would be empty; both cases emit a
    ``RuntimeWarning``. In the no-overlap case both frames are returned empty.
    """
    accelerometer_data = accelerometer_data.sort_values(by=timestamp_col)
    psg_data = psg_data.sort_values(by=timestamp_col)
    if accelerometer_data.empty or psg_data.empty:
        warnings.warn(
            "align_trim got an empty accelerometer or PSG frame; "
            "returning the frames unaligned.",
            RuntimeWarning,
            stacklevel=2,
        )
        return accelerometer_data, psg_data
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
    if start_time > end_time:
        warnings.warn(
            f"align_trim: accelerometer and PSG do not overlap after snapping "
            f"to {psg_dt}s epochs (start {start_time} > end {end_time}); "
            "returning empty frames.",
            RuntimeWarning,
            stacklevel=2,
        )
        empty_accel = accelerometer_data.iloc[0:0]
        empty_psg = psg_data.iloc[0:0]
        return empty_accel, empty_psg
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
    extra_mask_minutes: float = 0.0,
) -> pd.DataFrame:
    """Mark PSG epochs with insufficient accelerometer coverage as ``mask_value``.

    Only the epochs that actually lack coverage are marked. ``extra_mask_minutes``
    optionally dilates each flagged epoch by that many minutes on either side; it
    defaults to 0 so the mask stays a faithful record of where accel data is
    missing instead of manufacturing gap labels around every short dropout.

    Coverage is evaluated against the original accelerometer data for every
    epoch before anything is written, so masked epochs can never trigger
    further masking.

    Returns a new frame; ``psg_data`` is left unchanged.
    """
    if psg_data.empty:
        return psg_data.copy()
    psg_data = psg_data.copy()
    epoch_starts = psg_data[timestamp_col].to_numpy(dtype=float)
    accel_times = np.sort(accelerometer_data[timestamp_col].to_numpy(dtype=float))
    counts = np.searchsorted(
        accel_times, epoch_starts + float(psg_dt), side="left"
    ) - np.searchsorted(accel_times, epoch_starts, side="left")
    flagged = counts < minimum_accel_samples_per_psg
    extra_mask_idx = int(extra_mask_minutes * 60 // psg_dt)
    if extra_mask_idx > 0:
        dilated = flagged.copy()
        for tidx in np.flatnonzero(flagged):
            start = max(0, int(tidx) - extra_mask_idx)
            end = min(len(flagged), int(tidx) + extra_mask_idx + 1)
            dilated[start:end] = True
        flagged = dilated
    psg_data.loc[flagged, psg_col] = mask_value
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
