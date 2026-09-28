"""Frame ``j`` is window ``j``: the one starting ``j * hop`` after the origin.

senpy (>= 4.1) reports every window on its grid with its index and whether it
had data, and the regularise steps keep exactly that: every window with data
is a frame, in order, and a window without data is a padding frame. Nothing is
matched by nearest time, so no window is used twice and none is shifted.
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
from pisces_lite.proc.processing import (
    RegulariseNUFFTGrid,
    RegulariseStackedNUFFTGrid,
)

PAD = SPECTROGRAM_PADDING_VALUE


def _stacked(window_index, valid) -> "senpy.StackedSpectrogramResult":
    window_index = np.asarray(window_index)
    times = 5.0 + 2.0 * window_index
    Sxx = np.stack([np.full((3, 2), float(t)) for t in times])
    Sxx[~np.asarray(valid)] = np.nan
    return senpy.StackedSpectrogramResult(
        frequencies=np.arange(3.0),
        times=times,
        Sxx=Sxx,
        channels=["mag", "jerk"],
        kind="log_psd",
        window_index=window_index,
        valid=valid,
    )


def test_every_window_lands_on_exactly_one_frame() -> None:
    # Window 3 (centred at 11 s) had too few samples.
    result = RegulariseStackedNUFFTGrid(hop_seconds=2.0).transform(
        _stacked([0, 1, 2, 3, 4], [True, True, True, False, True])
    )

    np.testing.assert_allclose(result.times, [5, 7, 9, 11, 13])
    np.testing.assert_allclose(result.Sxx[:, 0, 0], [5, 7, 9, PAD, 13])
    np.testing.assert_array_equal(result.valid, [True, True, True, False, True])


def test_a_row_with_nan_in_one_channel_is_a_padding_frame() -> None:
    stacked = _stacked([0, 1], [True, True])
    stacked.Sxx[1, :, 1] = np.nan

    result = RegulariseStackedNUFFTGrid(hop_seconds=2.0).transform(stacked)

    assert np.all(result.Sxx[1] == PAD)
    assert result.valid.tolist() == [True, False]


def test_regularising_needs_window_indices() -> None:
    hand_built = senpy.SpectrogramResult(
        frequencies=np.arange(3.0), times=np.array([5.0]), Sxx=np.ones((1, 3))
    )

    with pytest.raises(ValueError, match="window_index"):
        RegulariseNUFFTGrid(hop_seconds=2.0).transform(hand_built)


def _accel(seconds: float, fs: float = 50.0, start: float = 0.0, seed: int = 1) -> np.ndarray:
    t = start + np.arange(0.0, seconds, 1 / fs)
    rng = np.random.default_rng(seed)
    return np.column_stack(
        [t, rng.normal(0, 0.5, t.size), rng.normal(0, 0.5, t.size), 1 + rng.normal(0, 0.1, t.size)]
    )


def _config(backend: str, **kwargs) -> ProcessingConfig:
    return ProcessingConfig(
        type="nufft",
        fs=12.0,
        window_seconds=10,
        window_step_seconds=2,
        fmin=0.1,
        fmax=6.0,
        normalization_mode="none",
        spectrogram_kind="log_psd",
        spectral_channels=["mag", "jerk"],
        nufft_backend=backend,
        **kwargs,
    )


@pytest.mark.parametrize("backend", ["cpu", "streaming"])
def test_real_recordings_have_no_duplicated_or_shifted_frames(backend: str) -> None:
    X = _config(backend).apply(_accel(180.0), normalize=False)

    # Every window has data, so every frame is a real FFT, starting with 0-10 s.
    assert not np.any(np.all(X == PAD, axis=(1, 2)))
    assert not any(np.array_equal(X[i], X[i + 1]) for i in range(len(X) - 1))
    # 180 s of data: a window starts every 2 s up to 178 s, the last few holding
    # only the data they reach.
    assert len(X) == 90


def test_backends_put_the_same_windows_in_the_same_frames() -> None:
    accel = _accel(240.0)
    accel = accel[(accel[:, 0] < 90.0) | (accel[:, 0] >= 125.0)]

    cpu = _config("cpu").apply(accel, normalize=False)
    streaming = _config("streaming").apply(accel, normalize=False)

    assert cpu.shape == streaming.shape
    np.testing.assert_array_equal(cpu == PAD, streaming == PAD)
    # FINUFFT's tolerance against the streaming path's direct sums, in log-PSD.
    np.testing.assert_allclose(cpu, streaming, rtol=1e-4)


def test_origin_moves_the_grid_in_the_timestamp_unit() -> None:
    # Timestamps in ms, the PSG starting 4 s before the accelerometer.
    accel = _accel(120.0, start=1_000.0)
    accel[:, 0] *= 1_000.0

    X = _config("cpu").apply(accel, normalize=False, origin=996_000.0)
    first_sample = _config("cpu").apply(accel, normalize=False)

    # Windows 0 and 1 (996-1006 s, 998-1008 s) hold 6 s and 8 s of data, so they
    # are FFTs too. Window 2 onwards is the grid anchored at the first sample.
    assert not np.any(np.all(X == PAD, axis=(1, 2)))
    assert X.shape[0] == first_sample.shape[0] + 2
    np.testing.assert_allclose(X[2:], first_sample, rtol=0, atol=1e-9)


def test_unix_origin_anchors_to_whole_steps_since_the_epoch() -> None:
    accel = _accel(60.0, start=1_700_000_001.0)

    X = _config("cpu", grid_origin="unix").apply(accel, normalize=False)
    explicit = _config("cpu").apply(accel, normalize=False, origin=1_700_000_002.0)

    np.testing.assert_allclose(X, explicit, rtol=1e-12)
