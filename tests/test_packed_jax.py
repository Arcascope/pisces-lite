"""The opt-in ``packing: "vectorized"`` JAX path matches senpy's packing.

The vectorized packer changes only how windows reach the device, so on the same
recordings it must give the same spectrograms as the default senpy packing.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("senpy", reason="needs the optional [proc] extra")
pytest.importorskip("jax", reason="needs the optional [jax] extra")

from pisces_lite.proc import ProcessingConfig


def _recording(seconds: float, gap: tuple[float, float] | None = None, seed: int = 0) -> np.ndarray:
    fs = 32.0
    t = np.arange(0.0, seconds, 1 / fs)
    rng = np.random.default_rng(seed)
    accel = np.column_stack(
        [t, rng.normal(0, 0.5, t.size), rng.normal(0, 0.5, t.size), 1 + rng.normal(0, 0.1, t.size)]
    )
    if gap is not None:
        accel = accel[(t < gap[0]) | (t >= gap[1])]
    return accel


def _config(**backend_kwargs) -> ProcessingConfig:
    return ProcessingConfig.from_dict(
        {
            "type": "nufft",
            "fs": 12.0,
            "window_seconds": 10,
            "window_step_seconds": 2,
            "fmin": 0.1,
            "fmax": 6.0,
            "normalization_mode": "none",
            "spectrogram_kind": "log_psd",
            "spectral_channels": ["mag", "jerk"],
            "nufft_backend": "jax",
            "nufft_backend_kwargs": backend_kwargs,
        }
    )


def test_vectorized_packing_matches_senpy_packing() -> None:
    recordings = [_recording(300.0, seed=1), _recording(420.0, gap=(100.0, 160.0), seed=2)]

    reference = _config(batch_size=64).apply_many(recordings, normalize=False)
    fast = _config(packing="vectorized", rows_per_call=256).apply_many(recordings, normalize=False)

    assert len(fast) == len(reference)
    for got, want in zip(fast, reference):
        assert got.shape == want.shape
        np.testing.assert_allclose(got, want, atol=1e-3)


def test_single_recording_apply_uses_the_same_option() -> None:
    accel = _recording(200.0, seed=3)

    np.testing.assert_allclose(
        _config(packing="vectorized").apply(accel, normalize=False),
        _config().apply(accel, normalize=False),
        atol=1e-3,
    )


def test_unknown_packing_is_refused() -> None:
    with pytest.raises(ValueError, match="packing"):
        _config(packing="turbo").apply(_recording(60.0), normalize=False)


def test_vectorized_only_options_need_vectorized_packing() -> None:
    with pytest.raises(ValueError, match="rows_per_call"):
        _config(rows_per_call=512).apply(_recording(60.0), normalize=False)


def test_vectorized_packing_bypasses_senpys_packer(monkeypatch) -> None:
    from senpy import jax_backend as senpy_jax

    def refuse(*args, **kwargs):
        raise AssertionError("senpy packer used despite packing='vectorized'")

    monkeypatch.setattr(senpy_jax, "pack_nustft_window_batches", refuse)

    assert _config(packing="vectorized").apply(_recording(60.0), normalize=False).ndim == 3
