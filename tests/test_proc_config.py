import pytest

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
