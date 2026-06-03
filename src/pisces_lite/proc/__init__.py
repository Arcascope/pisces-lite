"""Minimal, self-contained subset of pisces2 processing for inference.

Carved out of ``pisces2.configuration.processing_configuration`` and
``pisces2.processing`` so callers can compute NUFFT spectrograms without
importing the full pisces2 package (which pulls in Keras/TensorFlow and
a large model zoo at import time).

Only the ``nufft`` pipeline is vendored — feature names other than
``"spectrogram"`` and pipeline ``type``s other than ``"nufft"`` are not
supported here.
"""
from pisces_lite.proc.config import ProcessingConfig
from pisces_lite.proc.constants import SPECTROGRAM_PADDING_VALUE

__all__ = ["ProcessingConfig", "SPECTROGRAM_PADDING_VALUE"]
