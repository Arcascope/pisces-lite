"""Minimal, self-contained subset processing for inference.

Only the ``nufft`` pipeline is vendored -- feature names other than
``"spectrogram"`` and pipeline ``type``s other than ``"nufft"`` are not
supported here.

The pipeline classes resolve lazily (PEP 562). ``processing`` imports senpy at
module scope, and senpy is an optional compiled extension -- see the ``proc``
extra in pyproject.toml. Deferring these names means a consumer that only
reads a :class:`ProcessingConfig`, as a training host working from a prebuilt
feature cache does, never imports senpy and so never needs it installed.
Touching any of the pipeline names without the extra raises below.
"""
from pisces_lite.proc.config import ProcessingConfig
from pisces_lite.proc.constants import SPECTROGRAM_PADDING_VALUE
from pisces_lite.proc.frames import frame_validity

_LAZY = frozenset(
    {
        "stacked_nufft_pipeline",
        "ComputeStackedSpectrogramsNUFFT",
        "RegulariseStackedNUFFTGrid",
        "ExtractStackedArray",
    }
)


def __getattr__(name: str):
    if name not in _LAZY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from pisces_lite.proc._backend import load_processing

    return getattr(load_processing(f"pisces_lite.proc.{name}"), name)


def __dir__():
    return sorted(set(globals()) | _LAZY)


__all__ = [
    "ProcessingConfig",
    "SPECTROGRAM_PADDING_VALUE",
    "frame_validity",
    "stacked_nufft_pipeline",
    "ComputeStackedSpectrogramsNUFFT",
    "RegulariseStackedNUFFTGrid",
    "ExtractStackedArray",
]
