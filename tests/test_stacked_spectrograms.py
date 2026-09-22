import numpy as np
import pytest

pytest.importorskip(
    "senpy",
    reason="processing backend is the optional [proc] extra; see pyproject.toml",
)

from pisces_lite.proc.config import ProcessingConfig
from pisces_lite.proc.processing import (
    ComputeStackedSpectrogramsNUFFT,
    RegulariseStackedNUFFTGrid,
    ExtractStackedArray,
    stacked_nufft_pipeline,
)


def _make_accel_array(n_seconds: float = 120.0, fs: float = 50.0) -> np.ndarray:
    rng = np.random.default_rng(42)
    n = int(n_seconds * fs)
    t = np.arange(n, dtype=np.float64) / fs
    x = rng.standard_normal(n) * 0.5
    y = rng.standard_normal(n) * 0.5
    z = 1.0 + rng.standard_normal(n) * 0.1
    return np.column_stack([t, x, y, z])


def test_stacked_spectrogram_pipeline_shape():
    accel = _make_accel_array()
    pipeline = stacked_nufft_pipeline(
        secperseg=30.0,
        secoverlap=0.0,
        target_fs=16.0,
        channels=["x", "y", "z", "mag", "jerk"],
    )
    result = pipeline.transform(accel)
    assert result.ndim == 3
    T, F, C = result.shape
    assert C == 5
    assert T > 0
    assert F > 0


def test_stacked_spectrogram_cpu_backend_remains_available():
    accel = _make_accel_array()
    pipeline = stacked_nufft_pipeline(
        secperseg=30.0,
        secoverlap=0.0,
        target_fs=16.0,
        channels=["mag", "jerk"],
        nufft_backend="cpu",
    )

    result = pipeline.transform(accel)

    assert result.ndim == 3
    assert result.shape[2] == 2


def test_stacked_jax_backend_uses_senpy_packed_windows(monkeypatch):
    from senpy import jax_backend as senpy_jax

    calls = []

    def fake_window_batch(points, signals, valid, *, nfft_padded, median_fs, **kwargs):
        calls.append((points.shape, signals.shape, valid.shape, kwargs))
        shape = (points.shape[0], 3, nfft_padded // 2 + 1)
        return np.ones(shape, dtype=np.complex64)

    monkeypatch.setattr(senpy_jax, "compute_nustft_window_batch", fake_window_batch)
    pipeline = stacked_nufft_pipeline(
        secperseg=2.0,
        secoverlap=1.0,
        target_fs=4.0,
        channels=["x", "mag"],
        nufft_backend="jax",
        nufft_backend_kwargs={"batch_size": 8, "eps": 1e-5},
    )

    result = pipeline.transform(_make_accel_array(n_seconds=8.0, fs=4.0))

    assert calls
    assert result.ndim == 3
    assert result.shape[2] == 2
    assert np.all(result[1:] == 1.0)


def test_stacked_jax_apply_many_packs_windows_across_recordings(monkeypatch):
    from senpy import jax_backend as senpy_jax

    original_pack = senpy_jax.pack_nustft_window_batches
    packed_recording_counts = []

    def recording_pack(recordings, **kwargs):
        packed_recording_counts.append(len(recordings))
        return original_pack(recordings, **kwargs)

    def fake_window_batch(points, signals, valid, *, nfft_padded, median_fs, **kwargs):
        shape = (points.shape[0], 3, nfft_padded // 2 + 1)
        return np.ones(shape, dtype=np.complex64)

    monkeypatch.setattr(senpy_jax, "pack_nustft_window_batches", recording_pack)
    monkeypatch.setattr(senpy_jax, "compute_nustft_window_batch", fake_window_batch)
    cfg = ProcessingConfig.from_dict(
        {
            "type": "nufft",
            "fs": 4.0,
            "window_seconds": 2,
            "window_step_seconds": 1,
            "fmax": 2.0,
            "spectral_channels": ["x", "mag"],
            "normalization_mode": "none",
            "nufft_backend": "jax",
            "nufft_backend_kwargs": {"batch_size": 16},
        }
    )

    results = cfg.apply_many(
        [
            _make_accel_array(n_seconds=8.0, fs=4.0),
            _make_accel_array(n_seconds=9.0, fs=4.0),
        ],
        normalize=False,
    )

    assert packed_recording_counts == [2]
    assert len(results) == 2
    assert all(result.ndim == 3 and result.shape[2] == 2 for result in results)


def test_jerk_jax_apply_many_packs_windows_across_recordings(monkeypatch):
    from senpy import jax_backend as senpy_jax

    original_pack = senpy_jax.pack_nustft_window_batches
    packed_recording_counts = []

    def recording_pack(recordings, **kwargs):
        packed_recording_counts.append(len(recordings))
        return original_pack(recordings, **kwargs)

    def fake_window_batch(points, signals, valid, *, nfft_padded, median_fs, **kwargs):
        shape = (points.shape[0], 3, nfft_padded // 2 + 1)
        return np.ones(shape, dtype=np.complex64)

    monkeypatch.setattr(senpy_jax, "pack_nustft_window_batches", recording_pack)
    monkeypatch.setattr(senpy_jax, "compute_nustft_window_batch", fake_window_batch)
    cfg = ProcessingConfig.from_dict(
        {
            "type": "nufft",
            "fs": 4.0,
            "window_seconds": 2,
            "window_step_seconds": 1,
            "fmax": 2.0,
            "normalization_mode": "none",
            "nufft_backend": "jax",
            "nufft_backend_kwargs": {"batch_size": 16},
        }
    )

    results = cfg.apply_many(
        [
            _make_accel_array(n_seconds=8.0, fs=4.0),
            _make_accel_array(n_seconds=9.0, fs=4.0),
        ],
        normalize=False,
    )

    assert packed_recording_counts == [2]
    assert len(results) == 2
    assert all(result.ndim == 2 for result in results)


def test_processing_config_spectral_channels_produces_3d_output():
    cfg = ProcessingConfig.from_dict({
        "type": "nufft",
        "fs": 16.0,
        "window_seconds": 30,
        "window_step_seconds": 30,
        "fmin": 0.0,
        "fmax": 8.0,
        "spectral_channels": ["x", "y", "z", "mag", "jerk"],
        "normalization_mode": "none",
        "jerk_diff": True,
        "detrend": True,
        "spectrogram_kind": "magnitude",
    })

    accel = _make_accel_array()
    result = cfg.apply(accel, normalize=False)

    assert result.ndim == 3
    T, F, C = result.shape
    assert C == 5
    assert T > 0
    assert F > 0


def test_processing_config_no_spectral_channels_gives_2d_output():
    cfg = ProcessingConfig.from_dict({
        "type": "nufft",
        "fs": 16.0,
        "window_seconds": 30,
        "window_step_seconds": 30,
        "fmin": 0.0,
        "fmax": 8.0,
        "normalization_mode": "none",
        "jerk_diff": True,
        "detrend": True,
    })

    accel = _make_accel_array()
    result = cfg.apply(accel, normalize=False)

    assert result.ndim == 2


def test_stacked_config_channel_order():
    cfg = ProcessingConfig.from_dict({
        "type": "nufft",
        "spectral_channels": ["x", "y", "z", "mag", "jerk"],
    })
    assert cfg.spectral_channels == ["x", "y", "z", "mag", "jerk"]


def test_stacked_config_subset_channels():
    cfg = ProcessingConfig.from_dict({
        "type": "nufft",
        "fs": 16.0,
        "window_seconds": 30,
        "window_step_seconds": 30,
        "spectral_channels": ["x", "mag"],
        "normalization_mode": "none",
    })
    accel = _make_accel_array()
    result = cfg.apply(accel, normalize=False)
    assert result.ndim == 3
    assert result.shape[2] == 2


def test_stacked_config_frequency_bounds_respected():
    cfg = ProcessingConfig.from_dict({
        "type": "nufft",
        "fs": 16.0,
        "window_seconds": 30,
        "window_step_seconds": 30,
        "fmin": 0.0,
        "fmax": 4.0,
        "spectral_channels": ["x", "y", "z", "mag", "jerk"],
        "normalization_mode": "none",
    })
    cfg_full = ProcessingConfig.from_dict({
        "type": "nufft",
        "fs": 16.0,
        "window_seconds": 30,
        "window_step_seconds": 30,
        "fmin": 0.0,
        "fmax": 8.0,
        "spectral_channels": ["x", "y", "z", "mag", "jerk"],
        "normalization_mode": "none",
    })
    accel = _make_accel_array()
    result_half = cfg.apply(accel, normalize=False)
    result_full = cfg_full.apply(accel, normalize=False)

    assert result_half.shape[1] < result_full.shape[1]


def test_stacked_normalization_3d():
    cfg = ProcessingConfig.from_dict({
        "type": "nufft",
        "fs": 16.0,
        "window_seconds": 30,
        "window_step_seconds": 30,
        "spectral_channels": ["x", "y", "z", "mag", "jerk"],
        "normalization_mode": "znorm_axis0",
    })
    accel = _make_accel_array()
    result = cfg.apply(accel)
    assert result.ndim == 3
    assert np.isfinite(result).all()
