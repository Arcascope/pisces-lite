"""NUFFT spectrogram pipeline classes, vendored from pisces2.processing.

Only the classes reachable from ``ProcessingConfig._build_pipeline`` for
``type="nufft"`` are present. No keras/sklearn/model imports.
"""
from __future__ import annotations

import logging
import math
import time
import warnings
from typing import Dict, Generic, List, Optional, Sequence, TypeVar

import numpy as np
import senpy

from pisces_lite.proc import features as pf
from pisces_lite.proc.constants import SPECTROGRAM_PADDING_VALUE

_log = logging.getLogger(__name__)

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")

_NUFFT_BACKEND_ALIASES = {
    "streaming": "streaming",
    "cpu": "cpu",
    "finufft": "cpu",
    "jax": "jax",
    "jax_gpu": "jax",
    "jax_packed": "jax",
}


class ProcessingStep(Generic[InputT, OutputT]):
    @property
    def name(self) -> str:
        return "ProcessingStep"

    def transform(self, X: InputT) -> OutputT:
        raise NotImplementedError

    def transform_many(self, X: Sequence[InputT]) -> List[OutputT]:
        """Transform recordings independently unless a backend overrides this."""
        return [self.transform(item) for item in X]

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
        nufft_backend: str = "streaming",
        nufft_backend_kwargs: Optional[dict] = None,
    ):
        self.secperseg = secperseg
        self.secoverlap = secoverlap
        self.target_fs = target_fs
        self.detrend = detrend
        self.spectrogram_kind = _normalize_spectrogram_kind(spectrogram_kind)
        self.nufft_backend = _normalize_nufft_backend(nufft_backend)
        self.nufft_backend_kwargs = dict(nufft_backend_kwargs or {})

    @property
    def name(self) -> str:
        return (
            f"nufft_spectrogram({self.secperseg}s,overlap={self.secoverlap}s,"
            f"kind={self.spectrogram_kind},detrend={self.detrend},"
            f"backend={self.nufft_backend})"
        )

    def transform(self, jerk: "senpy.JerkData") -> "senpy.SpectrogramResult":
        return _compute_nufft_spectrogram(
            timestamps=jerk.timestamps_s,
            signal=jerk.jerk,
            window_s=self.secperseg,
            overlap_s=self.secoverlap,
            target_fs=self.target_fs if self.target_fs > 0.0 else None,
            kind=self.spectrogram_kind,
            detrend=self.detrend,
            backend=self.nufft_backend,
            backend_kwargs=self.nufft_backend_kwargs,
        )

    def transform_many(
        self, jerks: Sequence["senpy.JerkData"]
    ) -> List["senpy.SpectrogramResult"]:
        if self.nufft_backend != "jax":
            return super().transform_many(jerks)
        grouped = _compute_packed_jax_recording_spectrograms(
            recordings=[(jerk.timestamps_s, [jerk.jerk]) for jerk in jerks],
            window_s=self.secperseg,
            overlap_s=self.secoverlap,
            target_fs=self.target_fs if self.target_fs > 0.0 else None,
            kind=self.spectrogram_kind,
            detrend=self.detrend,
            backend_kwargs=self.nufft_backend_kwargs,
        )
        return [specs[0] for specs in grouped]


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

    def transform_many(self, X: Sequence[InputT]) -> List[OutputT]:
        x_out = list(X)
        for step in self.substeps:
            t0 = time.monotonic()
            try:
                _log.info("step %s starting for %d recordings", step.name, len(x_out))
                x_out = step.transform_many(x_out)
            except BaseException:
                _log.exception(
                    "step %s raised after %.2fs for %d recordings",
                    step.name,
                    time.monotonic() - t0,
                    len(x_out),
                )
                raise
            _log.info(
                "step %s done in %.2fs for %d recordings",
                step.name,
                time.monotonic() - t0,
                len(x_out),
            )
        return x_out


def _normalize_spectrogram_kind(kind: str) -> str:
    normalized = str(kind).replace("-", "_").lower()
    aliases = {"mag": "magnitude", "magnitude": "magnitude", "power": "power", "psd": "psd"}
    if normalized not in aliases:
        raise ValueError("spectrogram_kind must be one of: 'mag', 'magnitude', 'power', 'psd'")
    return aliases[normalized]


def _normalize_nufft_backend(backend: str) -> str:
    normalized = str(backend).replace("-", "_").lower()
    if normalized not in _NUFFT_BACKEND_ALIASES:
        raise ValueError("nufft_backend must be one of: 'streaming', 'cpu', 'jax'")
    return _NUFFT_BACKEND_ALIASES[normalized]


def _take_backend_kwargs(
    backend: str,
    backend_kwargs: Optional[dict],
    allowed: Sequence[str],
) -> Dict[str, object]:
    kwargs = dict(backend_kwargs or {})
    unknown = sorted(set(kwargs) - set(allowed))
    if unknown:
        raise ValueError(
            f"Unsupported {backend} NUFFT backend option(s): {', '.join(unknown)}"
        )
    return kwargs


def _default_streaming_subwindow_s(window_s: float, overlap_s: float) -> float:
    """Choose a <=1 s granularity that exactly divides the window and hop."""
    hop_s = window_s - overlap_s
    scale = 1_000_000
    window_us = int(round(window_s * scale))
    hop_us = int(round(hop_s * scale))
    return min(scale, math.gcd(window_us, hop_us)) / scale


def _compute_nufft_spectrogram(
    *,
    timestamps: np.ndarray,
    signal: np.ndarray,
    window_s: float,
    overlap_s: float,
    target_fs: Optional[float],
    kind: str,
    detrend: bool,
    backend: str,
    backend_kwargs: Optional[dict],
) -> "senpy.SpectrogramResult":
    backend = _normalize_nufft_backend(backend)
    if backend == "cpu":
        _take_backend_kwargs(backend, backend_kwargs, ())
        return senpy.compute_nufft_spectrogram(
            timestamps=timestamps,
            signal=signal,
            window_s=window_s,
            overlap_s=overlap_s,
            target_fs=target_fs,
            kind=kind,
            detrend=detrend,
        )
    if backend == "streaming":
        kwargs = _take_backend_kwargs(
            backend, backend_kwargs, ("subwindow_s", "chunk")
        )
        subwindow_s = float(
            kwargs.pop(
                "subwindow_s",
                _default_streaming_subwindow_s(window_s, overlap_s),
            )
        )
        result = senpy.compute_nustft_streaming(
            timestamps=timestamps,
            signal=signal,
            window_s=window_s,
            overlap_s=overlap_s,
            subwindow_s=subwindow_s,
            fmax=target_fs / 2.0 if target_fs is not None else None,
            detrend=detrend,
            **kwargs,
        )
        return result.spectrogram(kind)

    return _compute_packed_jax_spectrograms(
        timestamps=timestamps,
        signals=[signal],
        window_s=window_s,
        overlap_s=overlap_s,
        target_fs=target_fs,
        kind=kind,
        detrend=detrend,
        backend_kwargs=backend_kwargs,
    )[0]


def _spectral_surface(coefficients: np.ndarray, kind: str) -> np.ndarray:
    magnitude = np.abs(coefficients)
    return magnitude if kind == "magnitude" else magnitude * magnitude


def _compute_packed_jax_spectrograms(
    *,
    timestamps: np.ndarray,
    signals: Sequence[np.ndarray],
    window_s: float,
    overlap_s: float,
    target_fs: Optional[float],
    kind: str,
    detrend: bool,
    backend_kwargs: Optional[dict],
) -> List["senpy.SpectrogramResult"]:
    """Pack the channels of one recording and restore their time order."""
    return _compute_packed_jax_recording_spectrograms(
        recordings=[(timestamps, signals)],
        window_s=window_s,
        overlap_s=overlap_s,
        target_fs=target_fs,
        kind=kind,
        detrend=detrend,
        backend_kwargs=backend_kwargs,
    )[0]


def _compute_packed_jax_recording_spectrograms(
    *,
    recordings: Sequence[tuple[np.ndarray, Sequence[np.ndarray]]],
    window_s: float,
    overlap_s: float,
    target_fs: Optional[float],
    kind: str,
    detrend: bool,
    backend_kwargs: Optional[dict],
) -> List[List["senpy.SpectrogramResult"]]:
    """Pack channels and windows across recordings, then restore both orders."""
    kwargs = _take_backend_kwargs("jax", backend_kwargs, ("batch_size", "eps"))
    batch_size = int(kwargs.pop("batch_size", 128))
    eps = float(kwargs.pop("eps", 1e-6))

    from senpy import jax_backend as senpy_jax

    packed_recordings = []
    group_metadata = []
    for recording_index, (timestamps, signals) in enumerate(recordings):
        signals = [np.asarray(signal) for signal in signals]
        if not signals:
            raise ValueError("each JAX recording requires at least one signal")
        for channel_start in range(0, len(signals), 3):
            group = signals[channel_start : channel_start + 3]
            group_size = len(group)
            group.extend(np.zeros_like(group[0]) for _ in range(3 - group_size))
            packed_recordings.append((timestamps, np.column_stack(group)))
            group_metadata.append((recording_index, channel_start, group_size))

    batches = senpy_jax.pack_nustft_window_batches(
        packed_recordings,
        window_s=window_s,
        overlap_s=overlap_s,
        batch_size=batch_size,
    )
    if not batches:
        raise ValueError("JAX packed NUSTFT requires enough data for at least one window")

    rows_by_group: List[Dict[int, tuple[float, np.ndarray]]] = [
        {} for _ in packed_recordings
    ]
    frequencies_by_group: List[Optional[np.ndarray]] = [None] * len(packed_recordings)
    for batch in batches:
        coefficients = np.asarray(
            senpy_jax.compute_nustft_window_batch(
                batch.points,
                batch.signals,
                batch.valid,
                nfft_padded=batch.nfft_padded,
                median_fs=batch.median_fs,
                detrend=detrend,
                eps=eps,
            )
        )
        batch_frequencies = np.arange(
            batch.nfft_padded // 2 + 1, dtype=np.float64
        ) / window_s
        if target_fs is not None:
            keep = batch_frequencies <= target_fs / 2.0 + np.finfo(float).eps
            batch_frequencies = batch_frequencies[keep]
            coefficients = coefficients[..., keep]
        for row in np.flatnonzero(batch.row_valid):
            group_index = int(batch.recording_indices[row])
            window_index = int(batch.window_indices[row])
            group_frequencies = frequencies_by_group[group_index]
            if group_frequencies is None:
                frequencies_by_group[group_index] = batch_frequencies
            elif not np.array_equal(group_frequencies, batch_frequencies):
                raise ValueError("JAX packed NUSTFT produced an incompatible frequency grid")
            rows_by_group[group_index][window_index] = (
                float(batch.times[row]),
                coefficients[row],
            )

    results: List[List[Optional["senpy.SpectrogramResult"]]] = [
        [None] * len(signals) for _, signals in recordings
    ]
    for group_index, (recording_index, channel_start, group_size) in enumerate(
        group_metadata
    ):
        ordered = [rows_by_group[group_index][i] for i in sorted(rows_by_group[group_index])]
        if not ordered:
            raise ValueError(
                "JAX packed NUSTFT requires enough data for at least one window "
                f"in recording {recording_index}"
            )
        times = np.array([row[0] for row in ordered], dtype=np.float64)
        group_coefficients = np.stack([row[1] for row in ordered])
        for channel_index in range(group_size):
            results[recording_index][channel_start + channel_index] = senpy.SpectrogramResult(
                frequencies=frequencies_by_group[group_index],
                times=times,
                Sxx=_spectral_surface(group_coefficients[:, channel_index, :], kind),
                kind=kind,
                method="jax_finufft_packed",
            )
    if any(spec is None for recording in results for spec in recording):
        raise RuntimeError("JAX packed NUSTFT did not reconstruct every signal")
    return [[spec for spec in recording if spec is not None] for recording in results]


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
        nufft_backend: str = "streaming",
        nufft_backend_kwargs: Optional[dict] = None,
    ):
        self.secperseg = secperseg
        self.secoverlap = secoverlap
        self.target_fs = target_fs
        self.detrend = detrend
        self.spectrogram_kind = _normalize_spectrogram_kind(spectrogram_kind)
        self.channels = channels
        self.use_diff = use_diff
        self.nufft_backend = _normalize_nufft_backend(nufft_backend)
        self.nufft_backend_kwargs = dict(nufft_backend_kwargs or {})

    @property
    def name(self) -> str:
        ch = self.channels or senpy.STACKED_SPECTROGRAM_CHANNELS
        return (
            f"stacked_nufft({self.secperseg}s,"
            f"channels={ch},kind={self.spectrogram_kind},"
            f"backend={self.nufft_backend})"
        )

    def transform(self, X: np.ndarray) -> "senpy.StackedSpectrogramResult":
        if self.nufft_backend == "cpu":
            _take_backend_kwargs("cpu", self.nufft_backend_kwargs, ())
            return senpy.compute_stacked_spectrograms(
                accel=self._prepare_accelerometer(X),
                window_s=self.secperseg,
                overlap_s=self.secoverlap,
                target_fs=self.target_fs if self.target_fs > 0.0 else None,
                kind=self.spectrogram_kind,
                detrend=self.detrend,
                channels=self.channels,
                use_diff=self.use_diff,
            )
        timestamps, signals, channels = self._prepare_signals(X)
        target_fs = self.target_fs if self.target_fs > 0.0 else None
        if self.nufft_backend == "jax":
            specs = _compute_packed_jax_spectrograms(
                timestamps=timestamps,
                signals=signals,
                window_s=self.secperseg,
                overlap_s=self.secoverlap,
                target_fs=target_fs,
                kind=self.spectrogram_kind,
                detrend=self.detrend,
                backend_kwargs=self.nufft_backend_kwargs,
            )
        else:
            specs = [
                _compute_nufft_spectrogram(
                    timestamps=timestamps,
                    signal=signal,
                    window_s=self.secperseg,
                    overlap_s=self.secoverlap,
                    target_fs=target_fs,
                    kind=self.spectrogram_kind,
                    detrend=self.detrend,
                    backend="streaming",
                    backend_kwargs=self.nufft_backend_kwargs,
                )
                for signal in signals
            ]
        return _stack_spectrogram_results(specs, channels, self.secperseg - self.secoverlap)

    @staticmethod
    def _prepare_accelerometer(X: np.ndarray) -> "senpy.AccelerometerData":
        timestamps_raw = np.ascontiguousarray(X[..., 0], dtype=np.float64)
        median_dt = float(np.median(np.diff(timestamps_raw)))
        ts_unit = "ms" if median_dt >= 10 else "s"
        conversion = 1e3 if ts_unit == "ms" else 1e6
        timestamps_us = (timestamps_raw * conversion).astype(np.int64)
        return senpy.AccelerometerData(
            timestamps_us=timestamps_us,
            x=np.ascontiguousarray(X[..., 1], dtype=np.float64),
            y=np.ascontiguousarray(X[..., 2], dtype=np.float64),
            z=np.ascontiguousarray(X[..., 3], dtype=np.float64),
        )

    def _prepare_signals(
        self, X: np.ndarray
    ) -> tuple[np.ndarray, List[np.ndarray], List[str]]:
        accel = self._prepare_accelerometer(X)

        channels = list(self.channels or senpy.STACKED_SPECTROGRAM_CHANNELS)
        signal_by_channel = {
            "x": accel.x,
            "y": accel.y,
            "z": accel.z,
            "mag": senpy.compute_magnitude(accel.x, accel.y, accel.z),
        }
        if "jerk" in channels:
            jerk = senpy.compute_jerk(
                accel.timestamps_s,
                accel.x,
                accel.y,
                accel.z,
                use_diff=self.use_diff,
            )
            signal_by_channel["jerk"] = jerk.jerk
        unknown = [channel for channel in channels if channel not in signal_by_channel]
        if unknown:
            raise ValueError(
                f"Unknown channel {unknown[0]!r}. Must be one of: "
                f"{senpy.STACKED_SPECTROGRAM_CHANNELS}"
            )

        signals = [np.ascontiguousarray(signal_by_channel[ch]) for ch in channels]
        return accel.timestamps_s, signals, channels

    def transform_many(
        self, arrays: Sequence[np.ndarray]
    ) -> List["senpy.StackedSpectrogramResult"]:
        if self.nufft_backend != "jax":
            return super().transform_many(arrays)
        prepared = [self._prepare_signals(X) for X in arrays]
        grouped_specs = _compute_packed_jax_recording_spectrograms(
            recordings=[(timestamps, signals) for timestamps, signals, _ in prepared],
            window_s=self.secperseg,
            overlap_s=self.secoverlap,
            target_fs=self.target_fs if self.target_fs > 0.0 else None,
            kind=self.spectrogram_kind,
            detrend=self.detrend,
            backend_kwargs=self.nufft_backend_kwargs,
        )
        return [
            _stack_spectrogram_results(
                specs, channels, self.secperseg - self.secoverlap
            )
            for specs, (_, _, channels) in zip(grouped_specs, prepared)
        ]


def _stack_spectrogram_results(
    specs: Sequence["senpy.SpectrogramResult"],
    channels: List[str],
    hop_seconds: float,
) -> "senpy.StackedSpectrogramResult":
    ref = specs[0]
    Sxx = np.full(
        (len(ref.times), len(ref.frequencies), len(channels)),
        np.nan,
        dtype=np.float64,
    )
    unmatched = {}
    for channel_index, (channel, spec) in enumerate(zip(channels, specs)):
        if not np.array_equal(spec.frequencies, ref.frequencies):
            raise ValueError(f"NUFFT backend produced an incompatible grid for {channel!r}")
        if np.array_equal(spec.times, ref.times):
            Sxx[:, :, channel_index] = spec.Sxx
            continue
        indices = np.searchsorted(spec.times, ref.times)
        left = np.clip(indices - 1, 0, len(spec.times) - 1)
        right = np.clip(indices, 0, len(spec.times) - 1)
        use_left = np.abs(spec.times[left] - ref.times) <= np.abs(
            spec.times[right] - ref.times
        )
        best = np.where(use_left, left, right)
        distances = np.abs(spec.times[best] - ref.times)
        matched = distances <= hop_seconds / 2.0
        Sxx[matched, :, channel_index] = spec.Sxx[best[matched]]
        if not np.all(matched):
            unmatched[channel] = int(np.count_nonzero(~matched))
    if unmatched:
        detail = ", ".join(
            f"{channel}: {count}/{len(ref.times)} time bins"
            for channel, count in unmatched.items()
        )
        warnings.warn(
            "NUFFT backend produced NaN-filled time bins for channels that could not "
            f"be aligned ({detail}).",
            RuntimeWarning,
            stacklevel=2,
        )
    return senpy.StackedSpectrogramResult(
        frequencies=ref.frequencies,
        times=ref.times,
        Sxx=Sxx,
        channels=channels,
        kind=ref.kind,
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
    nufft_backend: str = "streaming",
    nufft_backend_kwargs: Optional[dict] = None,
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
                nufft_backend=nufft_backend,
                nufft_backend_kwargs=nufft_backend_kwargs,
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
    nufft_backend: str = "streaming",
    nufft_backend_kwargs: Optional[dict] = None,
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
                nufft_backend=nufft_backend,
                nufft_backend_kwargs=nufft_backend_kwargs,
            ),
            RegulariseNUFFTGrid(hop_seconds=secperseg - secoverlap),
            GetFeatures(features=features, feature_kwargs=feature_kwargs),
        ]
    )
