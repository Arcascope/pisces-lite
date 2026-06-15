"""NUFFT spectrogram pipeline classes, vendored from pisces2.processing.

Only the classes reachable from ``ProcessingConfig._build_pipeline`` for
``type="nufft"`` are present. No keras/sklearn/model imports.
"""
from __future__ import annotations

import logging
import time
from typing import Generic, List, Optional, TypeVar

import numpy as np
import senpy

from pisces_lite.proc import features as pf
from pisces_lite.proc.constants import SPECTROGRAM_PADDING_VALUE

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

    def __init__(
        self,
        secperseg: float,
        secoverlap: float,
        target_fs: float = 0.0,
        detrend: bool = True,
        spectrogram_kind: str = "magnitude",
    ):
        self.secperseg = secperseg
        self.secoverlap = secoverlap
        self.target_fs = target_fs
        self.detrend = detrend
        self.spectrogram_kind = _normalize_spectrogram_kind(spectrogram_kind)

    @property
    def name(self) -> str:
        return (
            f"nufft_spectrogram({self.secperseg}s,overlap={self.secoverlap}s,"
            f"kind={self.spectrogram_kind},detrend={self.detrend})"
        )

    def transform(self, jerk: "senpy.JerkData") -> "senpy.SpectrogramResult":
        return senpy.compute_nufft_spectrogram(
            timestamps=jerk.timestamps_s,
            signal=jerk.jerk,
            window_s=self.secperseg,
            overlap_s=self.secoverlap,
            target_fs=self.target_fs if self.target_fs > 0.0 else None,
            kind=self.spectrogram_kind,
            detrend=self.detrend,
        )


class RegulariseNUFFTGrid(ProcessingStep):
    """Sparse NUFFT spectrogram → dense uniform-time grid, sentinel-filling gaps."""

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
        dense_Sxx = np.full(
            (len(expected_times), n_freqs),
            SPECTROGRAM_PADDING_VALUE,
            dtype=Sxx.dtype,
        )
        for i, t_exp in enumerate(expected_times):
            dists = np.abs(times - t_exp)
            j = int(np.argmin(dists))
            if dists[j] <= tol:
                dense_Sxx[i] = Sxx[j]
        return senpy.SpectrogramResult(
            frequencies=result.frequencies,
            times=expected_times,
            Sxx=dense_Sxx,
            kind=result.kind,
            method=result.method,
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


def _normalize_spectrogram_kind(kind: str) -> str:
    normalized = str(kind).replace("-", "_").lower()
    aliases = {"mag": "magnitude", "magnitude": "magnitude", "power": "power", "psd": "psd"}
    if normalized not in aliases:
        raise ValueError("spectrogram_kind must be one of: 'mag', 'magnitude', 'power', 'psd'")
    return aliases[normalized]


class ComputeStackedSpectrogramsNUFFT(ProcessingStep):
    """Raw ``(N, 4)`` accel → ``StackedSpectrogramResult`` ``(T, F, C)`` via per-channel NUFFT.

    Each requested channel (x, y, z, mag, jerk) is transformed independently with
    the same NUFFT parameters, then stacked along the last axis.
    """

    def __init__(
        self,
        secperseg: float,
        secoverlap: float,
        target_fs: float = 0.0,
        detrend: bool = True,
        spectrogram_kind: str = "magnitude",
        channels: Optional[List[str]] = None,
        use_diff: bool = True,
    ):
        self.secperseg = secperseg
        self.secoverlap = secoverlap
        self.target_fs = target_fs
        self.detrend = detrend
        self.spectrogram_kind = _normalize_spectrogram_kind(spectrogram_kind)
        self.channels = channels
        self.use_diff = use_diff

    @property
    def name(self) -> str:
        ch = self.channels or senpy.STACKED_SPECTROGRAM_CHANNELS
        return (
            f"stacked_nufft({self.secperseg}s,"
            f"channels={ch},kind={self.spectrogram_kind})"
        )

    def transform(self, X: np.ndarray) -> "senpy.StackedSpectrogramResult":
        timestamps_raw = np.ascontiguousarray(X[..., 0], dtype=np.float64)
        median_dt = float(np.median(np.diff(timestamps_raw)))
        ts_unit = "ms" if median_dt >= 10 else "s"
        conversion = 1e3 if ts_unit == "ms" else 1e6
        timestamps_us = (timestamps_raw * conversion).astype(np.int64)

        accel = senpy.AccelerometerData(
            timestamps_us=timestamps_us,
            x=np.ascontiguousarray(X[..., 1], dtype=np.float64),
            y=np.ascontiguousarray(X[..., 2], dtype=np.float64),
            z=np.ascontiguousarray(X[..., 3], dtype=np.float64),
        )
        return senpy.compute_stacked_spectrograms(
            accel=accel,
            window_s=self.secperseg,
            overlap_s=self.secoverlap,
            target_fs=self.target_fs if self.target_fs > 0.0 else None,
            kind=self.spectrogram_kind,
            detrend=self.detrend,
            channels=self.channels,
            use_diff=self.use_diff,
        )


class RegulariseStackedNUFFTGrid(ProcessingStep):
    """Snap a ``StackedSpectrogramResult`` to a uniform time grid, filling gaps with the padding value."""

    def __init__(self, hop_seconds: float):
        self.hop_seconds = hop_seconds

    @property
    def name(self) -> str:
        return "regularise_stacked_nufft_grid"

    def transform(
        self, result: "senpy.StackedSpectrogramResult"
    ) -> "senpy.StackedSpectrogramResult":
        times = result.times
        Sxx = result.Sxx  # (T, F, C)
        if len(times) == 0:
            return result

        T, F, C = Sxx.shape
        t_end = times[-1]
        hop = self.hop_seconds
        tol = hop / 2.0

        expected_times = np.arange(0.0, t_end + tol, hop)
        dense_Sxx = np.full(
            (len(expected_times), F, C),
            SPECTROGRAM_PADDING_VALUE,
            dtype=Sxx.dtype,
        )
        idx_hi = np.searchsorted(times, expected_times)
        idx_lo = np.clip(idx_hi - 1, 0, len(times) - 1)
        idx_hi = np.clip(idx_hi, 0, len(times) - 1)
        d_lo = np.abs(times[idx_lo] - expected_times)
        d_hi = np.abs(times[idx_hi] - expected_times)
        best_idx = np.where(d_lo <= d_hi, idx_lo, idx_hi)
        best_dist = np.where(d_lo <= d_hi, d_lo, d_hi)
        mask = best_dist <= tol
        dense_Sxx[mask] = Sxx[best_idx[mask]]

        return senpy.StackedSpectrogramResult(
            frequencies=result.frequencies,
            times=expected_times,
            Sxx=dense_Sxx,
            channels=result.channels,
            kind=result.kind,
        )


class ExtractStackedArray(ProcessingStep):
    """Pull the ``(T, F, C)`` ndarray out of a ``StackedSpectrogramResult``.

    Optionally filters frequencies to ``[fmin, fmax]`` and time-downsamples.
    """

    def __init__(
        self,
        fmin: Optional[float] = None,
        fmax: Optional[float] = None,
        time_downsample_rate: int = 1,
    ):
        self.fmin = fmin
        self.fmax = fmax
        self.time_downsample_rate = int(time_downsample_rate)

    @property
    def name(self) -> str:
        return "extract_stacked_array"

    def transform(self, result: "senpy.StackedSpectrogramResult") -> np.ndarray:
        freq_filter = np.ones(len(result.frequencies), dtype=bool)
        if self.fmin is not None:
            freq_filter &= result.frequencies >= self.fmin
        if self.fmax is not None:
            freq_filter &= result.frequencies <= self.fmax
        return result.Sxx[:: self.time_downsample_rate, freq_filter, :]


def stacked_nufft_pipeline(
    secperseg: float,
    secoverlap: float,
    target_fs: float = 0.0,
    channels: Optional[List[str]] = None,
    feature_kwargs: Optional[dict] = None,
    use_diff: bool = True,
    detrend: bool = True,
    spectrogram_kind: str = "magnitude",
) -> CompositeStep:
    """Build: raw ``(N, 4)`` → stacked spectrograms → regularise → ``(T, F, C)`` array."""
    feature_kwargs = feature_kwargs or {}
    return CompositeStep(
        substeps=[
            ComputeStackedSpectrogramsNUFFT(
                secperseg=secperseg,
                secoverlap=secoverlap,
                target_fs=target_fs,
                detrend=detrend,
                spectrogram_kind=spectrogram_kind,
                channels=channels,
                use_diff=use_diff,
            ),
            RegulariseStackedNUFFTGrid(hop_seconds=secperseg - secoverlap),
            ExtractStackedArray(
                fmin=feature_kwargs.get("fmin"),
                fmax=feature_kwargs.get("fmax"),
                time_downsample_rate=feature_kwargs.get("time_downsample_rate", 1),
            ),
        ]
    )


def nufft_based_features(
    secperseg: float,
    secoverlap: float,
    features: List[str],
    target_fs: float = 0.0,
    feature_kwargs: dict | None = None,
    use_diff: bool = True,
    detrend: bool = True,
    spectrogram_kind: str = "magnitude",
) -> CompositeStep:
    """Build: raw (N, 4) → JerkNUFFT → NUFFTSpectrogram → regularise → GetFeatures."""
    feature_kwargs = feature_kwargs or {}
    return CompositeStep(
        substeps=[
            ComputeJerkNUFFT(use_diff=use_diff),
            ComputeSpectrogramNUFFT(
                secperseg=secperseg,
                secoverlap=secoverlap,
                target_fs=target_fs,
                detrend=detrend,
                spectrogram_kind=spectrogram_kind,
            ),
            RegulariseNUFFTGrid(hop_seconds=secperseg - secoverlap),
            GetFeatures(features=features, feature_kwargs=feature_kwargs),
        ]
    )
