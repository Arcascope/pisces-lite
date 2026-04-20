"""Minimal feature extractors. Only the ``spectrogram`` feature is supported
here because that's all the inference pipeline needs."""
from __future__ import annotations

import numpy as np
import senpy


class FrequencyFeature:
    def __init__(self, **kwargs):
        pass

    @property
    def name(self) -> str:
        return "frequency_feature"

    def compute(self, spectrogram_result: "senpy.SpectrogramResult") -> np.ndarray:
        raise NotImplementedError


class SpectrogramFeature(FrequencyFeature):
    """Return the spectrogram, optionally filtered to ``[fmin, fmax]`` Hz and
    time-downsampled.
    """

    def __init__(
        self,
        fmin: float | None = None,
        fmax: float | None = None,
        time_downsample_rate: int = 1,
        **kwargs,
    ):
        self.fmin = fmin
        self.fmax = fmax
        self.time_downsample_rate = int(time_downsample_rate)

    @property
    def name(self):
        return "spectrogram"

    def compute(self, spectrogram_result: "senpy.SpectrogramResult") -> np.ndarray:
        frequency_filter = np.ones(spectrogram_result.Sxx.shape[1], dtype=bool)
        if self.fmin is not None:
            frequency_filter &= spectrogram_result.frequencies >= self.fmin
        if self.fmax is not None:
            frequency_filter &= spectrogram_result.frequencies <= self.fmax
        return spectrogram_result.Sxx[:: self.time_downsample_rate, frequency_filter]


def get_feature_by_name(name: str, **kwargs) -> FrequencyFeature | None:
    """Dispatcher for feature names used by ProcessingConfig._build_pipeline.

    Only ``"spectrogram"`` / ``"SpectrogramFeature"`` are supported here. Other
    names from the full pisces2 feature library would pull in heavier deps and
    aren't needed for inference on the shipped inputs.json.
    """
    feature_classes = {
        "SpectrogramFeature": SpectrogramFeature,
        "spectrogram": SpectrogramFeature,
    }
    if name not in feature_classes:
        raise ValueError(
            f"feature '{name}' is not vendored into pisces_lite.proc; "
            "add it if a new inputs.json needs it."
        )
    return feature_classes[name](**kwargs)
