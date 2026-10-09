import numpy as np
import pytest

pytest.importorskip(
    "senpy",
    reason="processing backend is the optional [proc] extra; see pyproject.toml",
)

from pisces_lite.proc.config import ProcessingConfig
from pisces_lite.proc.processing import jerk_lowpass_taps, lowpass_axes


def _gain(taps: np.ndarray, fs: float, f: float) -> float:
    n = np.arange(len(taps)) - len(taps) // 2
    return float(np.abs(np.sum(taps * np.exp(-2j * np.pi * f / fs * n))))


def _accel(fs: float, n_seconds: float = 300.0, tone_hz: float = 15.0) -> np.ndarray:
    rng = np.random.default_rng(0)
    t = np.arange(int(n_seconds * fs)) / fs
    slow = 0.2 * np.sin(2 * np.pi * 0.3 * t)
    tone = 0.05 * np.sin(2 * np.pi * tone_hz * t)
    x = slow + tone + 0.01 * rng.standard_normal(t.size)
    y = 0.5 * slow + 0.01 * rng.standard_normal(t.size)
    z = 1.0 + 0.01 * rng.standard_normal(t.size)
    return np.column_stack([t, x, y, z])


def _config(channels, lowpass_hz, backend="streaming"):
    return ProcessingConfig.from_dict({
        "type": "nufft",
        "fs": 12.0,
        "window_seconds": 10,
        "window_step_seconds": 2,
        "fmin": 0.1,
        "fmax": 6.0,
        "spectral_channels": channels,
        "normalization_mode": "none",
        "spectrogram_kind": "log_psd",
        "nufft_backend": backend,
        "jerk_lowpass_hz": lowpass_hz,
    })


@pytest.mark.parametrize("fs", [16.7, 25.0, 50.0, 100.0])
def test_taps_pass_below_and_stop_above_the_cutoff(fs):
    taps = jerk_lowpass_taps(fs, 6.0)
    assert len(taps) % 2 == 1
    np.testing.assert_allclose(taps, taps[::-1])
    assert abs(_gain(taps, fs, 2.0) - 1.0) < 1e-2
    assert abs(_gain(taps, fs, 6.0) - 0.5) < 0.05
    assert _gain(taps, fs, 7.5) < 0.01


def test_taps_are_an_impulse_near_the_cutoff():
    np.testing.assert_array_equal(jerk_lowpass_taps(12.5, 6.0), [1.0])


def test_lowpass_keeps_a_constant_signal_constant_at_the_ends():
    t = np.arange(500) / 50.0
    (out,) = lowpass_axes(t, [np.full(t.size, 0.98)], 6.0)
    np.testing.assert_allclose(out, 0.98, rtol=1e-12)


def test_lowpass_off_returns_the_axes():
    t = np.arange(10) / 50.0
    axes = [np.arange(10.0), np.ones(10), np.zeros(10)]
    for got, want in zip(lowpass_axes(t, axes, None), axes):
        np.testing.assert_array_equal(got, want)


def test_stacked_lowpass_changes_jerk_and_leaves_magnitude():
    accel = _accel(50.0)
    off = _config(["mag", "jerk"], None).apply(accel, normalize=False)
    on = _config(["mag", "jerk"], 6.0).apply(accel, normalize=False)
    assert off.shape == on.shape
    np.testing.assert_allclose(on[..., 0], off[..., 0])
    assert not np.allclose(on[..., 1], off[..., 1])


def test_stacked_lowpass_changes_nothing_at_low_rates():
    accel = _accel(12.5, tone_hz=5.0)
    off = _config(["mag", "jerk"], None).apply(accel, normalize=False)
    on = _config(["mag", "jerk"], 6.0).apply(accel, normalize=False)
    np.testing.assert_array_equal(on, off)


def test_jerk_only_pipeline_runs_with_lowpass():
    cfg = _config(None, 6.0)
    out = cfg.apply(_accel(50.0), normalize=False)
    assert out.ndim == 2
    assert np.isfinite(out).any()


def test_jax_backend_matches_streaming_with_lowpass():
    pytest.importorskip("senpy.jax_backend")
    accel = _accel(50.0)
    streaming = _config(["mag", "jerk"], 6.0).apply(accel, normalize=False)
    jax = _config(["mag", "jerk"], 6.0, backend="jax").apply(accel, normalize=False)
    np.testing.assert_allclose(jax, streaming, atol=1e-3)


def test_cpu_backend_refuses_lowpass():
    with pytest.raises(ValueError):
        _config(["mag", "jerk"], 6.0, backend="cpu").apply(_accel(50.0), normalize=False)
