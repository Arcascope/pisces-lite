"""Minimal, self-contained subset of pisces2 processing for inference.

Carved out of ``pisces2.configuration.processing_configuration`` and
``pisces2.processing`` so callers can compute NUFFT spectrograms without
importing the full pisces2 package (which pulls in Keras/TensorFlow and
a large model zoo at import time).

Only the ``nufft`` and ``c_nufft`` pipelines are vendored here.
"""
from pisces_lite.proc.config import ProcessingConfig

__all__ = ["ProcessingConfig"]
