"""Minimal, self-contained subset processing for inference.

Only the ``nufft`` pipeline is vendored — feature names other than
``"spectrogram"`` and pipeline ``type``s other than ``"nufft"`` are not
supported here.
"""
from pisces_lite.proc.config import ProcessingConfig
from pisces_lite.proc.constants import SPECTROGRAM_PADDING_VALUE
from pisces_lite.proc.processing import (
    stacked_nufft_pipeline,
    ComputeStackedSpectrogramsNUFFT,
    RegulariseStackedNUFFTGrid,
    ExtractStackedArray,
)

__all__ = [
    "ProcessingConfig",
    "SPECTROGRAM_PADDING_VALUE",
    "stacked_nufft_pipeline",
    "ComputeStackedSpectrogramsNUFFT",
    "RegulariseStackedNUFFTGrid",
    "ExtractStackedArray",
]
