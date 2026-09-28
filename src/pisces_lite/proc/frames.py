"""Frame-level bookkeeping on spectrogram features.

NumPy only, so a consumer can inspect a cached feature array without the
optional senpy extension.
"""
from __future__ import annotations

import numpy as np

from pisces_lite.proc.constants import SPECTROGRAM_PADDING_VALUE


def frame_validity(
    features: np.ndarray, padding_value: float = SPECTROGRAM_PADDING_VALUE
) -> np.ndarray:
    """Boolean ``(T,)``: ``True`` where frame ``t`` holds a real spectral estimate.

    The grid-regularisation steps fill a frame with no window behind it with
    ``padding_value`` in every bin and channel, so a frame is excluded exactly
    when all of its values equal that sentinel. ``features`` is ``(T, ...)``,
    and must be un-normalized: normalization rescales the sentinel along with
    everything else, after which excluded frames can no longer be told apart.
    """
    arr = np.asarray(features)
    if arr.ndim < 1:
        raise ValueError("features must have a leading time axis")
    if arr.ndim == 1:
        return arr != padding_value
    return ~np.all(arr == padding_value, axis=tuple(range(1, arr.ndim)))
