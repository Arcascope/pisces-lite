"""NUFFT spectrogram pipeline classes, vendored from pisces2.processing.

Only the classes reachable from ``ProcessingConfig._build_pipeline`` for
``type="nufft"`` are present. No keras/sklearn/model imports.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Generic, List, TypeVar

import numpy as np
import senpy

from pisces_lite.proc import features as pf

_log = logging.getLogger(__name__)

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


class ProcessingStep(Generic[InputT, OutputT]):
    @property
    def name(self) -> str:
        return "ProcessingStep"

    def transform(self, X: InputT) -> OutputT:
        raise NotImplementedError

    def plot_transform(self, X: InputT, saveto) -> OutputT:
        return self.transform(X)


class ComputeJerkNUFFT(ProcessingStep):
    """Raw (N, 4) accel array → senpy.JerkData with non-uniform timestamps."""

    def __init__(self, use_diff: bool = True):
        self.use_diff = use_diff

    @property
    def name(self) -> str:
        return "jerk_nufft"

    def transform(self, X: np.ndarray) -> "senpy.JerkData":
        timestamps = np.ascontiguousarray(X[..., 0])
        median_dt = float(np.median(np.diff(timestamps)))
        ts_unit = "ms" if median_dt > 10 else "s"
        return senpy.compute_jerk(
            timestamps,
            np.ascontiguousarray(X[..., 1]),
            np.ascontiguousarray(X[..., 2]),
            np.ascontiguousarray(X[..., 3]),
            ts_unit=ts_unit,
            use_diff=self.use_diff,
        )


class ComputeSpectrogramNUFFT(ProcessingStep):
    """senpy.JerkData (non-uniform) → senpy.SpectrogramResult via NUFFT."""

    def __init__(self, secperseg: float, secoverlap: float, target_fs: float = 0.0):
        self.secperseg = secperseg
        self.secoverlap = secoverlap
        self.target_fs = target_fs

    @property
    def name(self) -> str:
        return f"nufft_spectrogram({self.secperseg}s,overlap={self.secoverlap}s)"

    def transform(self, jerk: "senpy.JerkData") -> "senpy.SpectrogramResult":
        return senpy.compute_spectrogram_nufft(
            timestamps=jerk.timestamps_s,
            signal=jerk.jerk,
            secperseg=self.secperseg,
            secoverlap=self.secoverlap,
            target_fs=self.target_fs,
        )


class RegulariseNUFFTGrid(ProcessingStep):
    """Sparse NUFFT spectrogram → dense uniform-time grid, zero-filling gaps."""

    def __init__(self, hop_seconds: float):
        self.hop_seconds = hop_seconds

    @property
    def name(self) -> str:
        return "regularise_nufft_grid"

    def transform(self, result: "senpy.SpectrogramResult") -> "senpy.SpectrogramResult":
        times = result.times
        Sxx = result.Sxx
        if len(times) == 0:
            return result

        n_freqs = Sxx.shape[1]
        t_end = times[-1]
        hop = self.hop_seconds
        tol = hop / 2.0

        expected_times = np.arange(0.0, t_end + tol, hop)
        dense_Sxx = np.zeros((len(expected_times), n_freqs), dtype=Sxx.dtype)
        for i, t_exp in enumerate(expected_times):
            dists = np.abs(times - t_exp)
            j = int(np.argmin(dists))
            if dists[j] <= tol:
                dense_Sxx[i] = Sxx[j]
        return senpy.SpectrogramResult(
            frequencies=result.frequencies,
            times=expected_times,
            Sxx=dense_Sxx,
        )


class GetFeatures(ProcessingStep):
    def __init__(self, features: List[str], feature_kwargs: dict | None = None):
        feature_kwargs = feature_kwargs or {}
        self.feature_fns = [pf.get_feature_by_name(name=f, **feature_kwargs) for f in features]

    def transform(self, spectrogram: "senpy.SpectrogramResult") -> np.ndarray:
        features = [fn.compute(spectrogram) for fn in self.feature_fns]
        stacked = np.stack(features, axis=-1)
        if stacked.ndim > 2:
            shape = stacked.shape
            rest = 1
            for d in shape[1:]:
                rest *= d
            stacked = stacked.reshape((shape[0], rest))
        return stacked


class CompositeStep(ProcessingStep):
    def __init__(self, substeps: List[ProcessingStep]):
        self.substeps = substeps

    @property
    def name(self) -> str:
        return "+".join(f"({s.name})" for s in self.substeps)

    def transform(self, X):
        x_out = X
        for step in self.substeps:
            t0 = time.monotonic()
            try:
                _log.info("step %s starting", step.name)
                x_out = step.transform(x_out)
            except BaseException as exc:
                _log.exception("step %s raised after %.2fs", step.name, time.monotonic() - t0)
                raise
            shape = getattr(x_out, "shape", None)
            if shape is None and hasattr(x_out, "Sxx"):
                shape = ("Sxx=", x_out.Sxx.shape)
            _log.info("step %s done in %.2fs → %s", step.name, time.monotonic() - t0, shape)
        return x_out


def nufft_based_features(
    secperseg: float,
    secoverlap: float,
    features: List[str],
    target_fs: float = 0.0,
    feature_kwargs: dict | None = None,
    use_diff: bool = True,
) -> CompositeStep:
    """Build: raw (N, 4) → JerkNUFFT → NUFFTSpectrogram → regularise → GetFeatures."""
    feature_kwargs = feature_kwargs or {}
    return CompositeStep(
        substeps=[
            ComputeJerkNUFFT(use_diff=use_diff),
            ComputeSpectrogramNUFFT(
                secperseg=secperseg, secoverlap=secoverlap, target_fs=target_fs,
            ),
            RegulariseNUFFTGrid(hop_seconds=secperseg - secoverlap),
            GetFeatures(features=features, feature_kwargs=feature_kwargs),
        ]
    )
