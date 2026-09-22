import pytest

pytest.importorskip(
    "senpy",
    reason="processing backend is the optional [proc] extra; see pyproject.toml",
)

from pisces_lite.proc.config import ProcessingConfig
from pisces_lite.proc.processing import ComputeSpectrogramNUFFT


def test_processing_config_accepts_detrend_and_spectral_kind_aliases():
    cfg = ProcessingConfig.from_dict({"type": "nufft", "detrend": False, "mode": "mag"})

    assert cfg.detrend is False
    assert cfg.spectrogram_kind == "mag"

    step = ComputeSpectrogramNUFFT(
        secperseg=30,
        secoverlap=0,
        detrend=cfg.detrend,
        spectrogram_kind=cfg.spectrogram_kind,
    )

    assert step.detrend is False
    assert step.spectrogram_kind == "magnitude"


def test_processing_config_rejects_unknown_spectral_kind():
    with pytest.raises(ValueError, match="spectrogram_kind"):
        ComputeSpectrogramNUFFT(30, 0, spectrogram_kind="phase")


def test_processing_config_defaults_to_streaming_nufft_backend():
    cfg = ProcessingConfig.from_dict({"type": "nufft"})

    assert cfg.nufft_backend == "streaming"
    assert cfg._build_pipeline().substeps[1].nufft_backend == "streaming"


def test_processing_config_accepts_backend_alias_and_kwargs():
    cfg = ProcessingConfig.from_dict(
        {
            "type": "nufft",
            "backend": "jax_packed",
            "backend_kwargs": {"batch_size": 64, "eps": 1e-5},
        }
    )
    step = cfg._build_pipeline().substeps[1]

    assert step.nufft_backend == "jax"
    assert step.nufft_backend_kwargs == {"batch_size": 64, "eps": 1e-5}


def test_processing_config_rejects_unknown_nufft_backend():
    cfg = ProcessingConfig.from_dict({"type": "nufft", "nufft_backend": "magic"})

    with pytest.raises(ValueError, match="nufft_backend"):
        cfg._build_pipeline()
