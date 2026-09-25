"""The uniform frame grid the regularisation steps snap window centres onto.

senpy stamps windows at their centres, ``window / 2 + k * hop`` after the first
sample. With 10 s windows every 2 s those are odd seconds; a grid anchored at 0
put every one half a hop between two frames. These tests pin the grid to the
centres' own offset so every window lands on exactly one frame.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip(
    "senpy",
    reason="processing backend is the optional [proc] extra; see pyproject.toml",
)

import senpy

from pisces_lite.proc import SPECTROGRAM_PADDING_VALUE, ProcessingConfig
from pisces_lite.proc.processing import RegulariseStackedNUFFTGrid, frame_grid

PAD = SPECTROGRAM_PADDING_VALUE


def test_grid_keeps_the_window_centres_offset() -> None:
    np.testing.assert_allclose(frame_grid(np.array([5.0, 7.0, 9.0]), 2.0), [1, 3, 5, 7, 9])


def test_grid_on_even_centres_starts_at_zero() -> None:
    np.testing.assert_allclose(frame_grid(np.array([4.0, 6.0, 10.0]), 2.0), [0, 2, 4, 6, 8, 10])


def test_grid_tolerates_float_jitter_in_the_centres() -> None:
    times = np.array([5.0, 7.0 + 1e-12, 9.0 - 1e-12, 11.0])
    np.testing.assert_allclose(frame_grid(times, 2.0), [1, 3, 5, 7, 9, 11])
    np.testing.assert_allclose(frame_grid(np.array([2.0 - 1e-13, 4.0]), 2.0), [0, 2, 4])


def _stacked(times: np.ndarray) -> "senpy.StackedSpectrogramResult":
    Sxx = np.stack([np.full((3, 2), float(t)) for t in times])
    return senpy.StackedSpectrogramResult(
        frequencies=np.arange(3.0),
        times=times,
        Sxx=Sxx,
        channels=["mag", "jerk"],
        kind="log_psd",
    )


def test_every_window_lands_on_exactly_one_frame() -> None:
    # 11 s is missing: the window there had too few samples.
    result = RegulariseStackedNUFFTGrid(hop_seconds=2.0).transform(
        _stacked(np.array([5.0, 7.0, 9.0, 13.0]))
    )

    np.testing.assert_allclose(result.times, [1, 3, 5, 7, 9, 11, 13])
    filled = result.Sxx[:, 0, 0]
    np.testing.assert_allclose(filled, [PAD, PAD, 5, 7, 9, PAD, 13])


def test_real_recordings_have_no_duplicated_or_shifted_frames() -> None:
    fs = 50.0
    t = np.arange(0.0, 180.0, 1 / fs)
    rng = np.random.default_rng(1)
    accel = np.column_stack(
        [t, rng.normal(0, 0.5, t.size), rng.normal(0, 0.5, t.size), 1 + rng.normal(0, 0.1, t.size)]
    )
    config = ProcessingConfig(
        type="nufft",
        fs=12.0,
        window_seconds=10,
        window_step_seconds=2,
        fmin=0.1,
        fmax=6.0,
        normalization_mode="none",
        spectrogram_kind="log_psd",
        spectral_channels=["mag", "jerk"],
        nufft_backend="cpu",
    )

    X = config.apply(accel, normalize=False)
    excluded = np.all(X == PAD, axis=(1, 2))

    # Only the frames centred before the first 10 s window (1 s and 3 s) are empty.
    np.testing.assert_array_equal(np.flatnonzero(excluded), [0, 1])
    assert not any(np.array_equal(X[i], X[i + 1]) for i in range(2, len(X) - 1))
