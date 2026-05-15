from types import SimpleNamespace

import numpy as np

from pisces_lite.proc.config import ProcessingConfig
from pisces_lite.proc import processing as processing_module
from pisces_lite.proc.features import get_feature_by_name
from pisces_lite.proc.processing import ComputeSpectrogramNUFFT, GetFeatures


def _spectrogram_result():
    sxx = np.arange(12, dtype=float).reshape(2, 6)
    phase_vector = np.zeros((2, 6, 2), dtype=float)
    phase_vector[..., 0] = 1.0
    phase_weight = np.ones((2, 6), dtype=float)
    phase_weight[:, :2] = 0.0
    phase_vector[:, :2, :] = 0.0
    return SimpleNamespace(
        frequencies=np.arange(6, dtype=float),
        times=np.array([0.0, 1.0]),
        Sxx=sxx,
        phase_vector=phase_vector,
        phase_weight=phase_weight,
    )


def test_complex_nufft_feature_returns_magnitude_phase_and_weight_channels():
    feature = get_feature_by_name("c_nufft", fmin=1.0, fmax=4.0)
    result = feature.compute(_spectrogram_result())

    assert result.shape == (2, 4, 4)
    np.testing.assert_array_equal(result[..., 0], _spectrogram_result().Sxx[:, 1:5])
    np.testing.assert_array_equal(result[..., 1:3], _spectrogram_result().phase_vector[:, 1:5, :])
    np.testing.assert_array_equal(result[..., 3], _spectrogram_result().phase_weight[:, 1:5])


def test_get_features_flattens_complex_nufft_channels():
    step = GetFeatures(features=["c_nufft"], feature_kwargs={"fmin": 1.0, "fmax": 4.0})
    result = step.transform(_spectrogram_result())

    assert result.shape == (2, 16)


def test_c_nufft_config_requests_phase_from_senpy():
    pipeline = ProcessingConfig(
        type="c_nufft",
        fs=20.0,
        window_seconds=10,
        window_step_seconds=5,
        phase_magnitude_threshold=0.01,
    )._build_pipeline()

    spectrogram_step = next(
        step for step in pipeline.substeps if isinstance(step, ComputeSpectrogramNUFFT)
    )
    get_features_step = next(step for step in pipeline.substeps if isinstance(step, GetFeatures))

    assert spectrogram_step.return_phase is True
    assert spectrogram_step.phase_magnitude_threshold == 0.01
    assert get_features_step.feature_fns[0].name == "c_nufft"


def test_nufft_config_keeps_existing_magnitude_only_path():
    pipeline = ProcessingConfig(type="nufft")._build_pipeline()
    spectrogram_step = next(
        step for step in pipeline.substeps if isinstance(step, ComputeSpectrogramNUFFT)
    )
    get_features_step = next(step for step in pipeline.substeps if isinstance(step, GetFeatures))

    assert spectrogram_step.return_phase is False
    assert get_features_step.feature_fns[0].name == "spectrogram"


def test_magnitude_only_spectrogram_step_does_not_require_new_senpy_kwargs(monkeypatch):
    seen = {}

    def fake_compute_spectrogram_nufft(**kwargs):
        seen.update(kwargs)
        return object()

    monkeypatch.setattr(
        processing_module.senpy,
        "compute_spectrogram_nufft",
        fake_compute_spectrogram_nufft,
    )
    jerk = SimpleNamespace(timestamps_s=np.array([0.0, 1.0]), jerk=np.array([0.0, 1.0]))

    ComputeSpectrogramNUFFT(secperseg=10.0, secoverlap=5.0, target_fs=20.0).transform(jerk)

    assert "return_phase" not in seen
    assert "phase_magnitude_threshold" not in seen


def test_complex_spectrogram_step_requests_new_senpy_phase_kwargs(monkeypatch):
    seen = {}

    def fake_compute_spectrogram_nufft(**kwargs):
        seen.update(kwargs)
        return object()

    monkeypatch.setattr(
        processing_module.senpy,
        "compute_spectrogram_nufft",
        fake_compute_spectrogram_nufft,
    )
    jerk = SimpleNamespace(timestamps_s=np.array([0.0, 1.0]), jerk=np.array([0.0, 1.0]))

    ComputeSpectrogramNUFFT(
        secperseg=10.0,
        secoverlap=5.0,
        target_fs=20.0,
        return_phase=True,
        phase_magnitude_threshold=0.01,
    ).transform(jerk)

    assert seen["return_phase"] is True
    assert seen["phase_magnitude_threshold"] == 0.01
