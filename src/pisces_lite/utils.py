"""Small array utilities used across pisces-lite."""
from __future__ import annotations

from typing import Union

import numpy as np


def pad_or_truncate(
    array: np.ndarray,
    target_length: int,
    pad_value: Union[float, np.ndarray] = 0.0,
    axis: int = 0,
    mode: str = "post",
) -> np.ndarray:
    """Pad or truncate ``array`` along ``axis`` to exactly ``target_length``.

    Args:
        array: Input array of any shape.
        target_length: Desired size along ``axis``.
        pad_value: Scalar or 1-D array of per-feature pad values. A 1-D array
            must match the last axis of ``array`` for broadcast padding.
        axis: Axis to pad/truncate along.
        mode: ``"pre"`` | ``"post"`` | ``"sym"`` (split evenly, extra goes
            after).

    """
    current_length = array.shape[axis]

    if current_length > target_length:
        slices = [slice(None)] * array.ndim
        slices[axis] = slice(0, target_length)
        return array[tuple(slices)]

    if current_length == target_length:
        return array

    total_pad = target_length - current_length
    if mode == "pre":
        pad_before, pad_after = total_pad, 0
    elif mode == "post":
        pad_before, pad_after = 0, total_pad
    elif mode == "sym":
        pad_before = total_pad // 2
        pad_after = total_pad - pad_before
    else:
        raise ValueError(f"Unsupported mode {mode!r}. Use 'pre', 'post', or 'sym'.")

    if isinstance(pad_value, np.ndarray):
        def _block(size: int):
            if size == 0:
                return None
            pad_shape = list(array.shape)
            pad_shape[axis] = size
            broadcast_shape = [1] * array.ndim
            broadcast_shape[-1] = len(pad_value)
            return np.broadcast_to(pad_value.reshape(broadcast_shape), pad_shape)

        parts = []
        pre = _block(pad_before)
        if pre is not None:
            parts.append(pre)
        parts.append(array)
        post = _block(pad_after)
        if post is not None:
            parts.append(post)
        return np.concatenate(parts, axis=axis)

    pad_width = [(0, 0)] * array.ndim
    pad_width[axis] = (pad_before, pad_after)
    return np.pad(array, pad_width, mode="constant", constant_values=pad_value)
