"""NUFFT spectrogram pipeline classes.

Only the classes reachable from ``ProcessingConfig._build_pipeline`` for
``type="nufft"`` are present. No keras/sklearn/model imports.
"""
from __future__ import annotations

import logging
import math
import time
from typing import Dict, Generic, List, Optional, Sequence, TypeVar, Union

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


#: Where window 0 of the NUSTFT grid starts: ``None`` (the first sample), a
#: number of seconds on the timestamps' clock, or ``"unix"`` (a whole number of
#: hops since the Unix epoch). See ``senpy.window_grid``.
OriginSpec = Union[None, float, str]


class ProcessingStep(Generic[InputT, OutputT]):
    #: Steps that place windows on the grid take ``origin_s`` in ``transform``.
    takes_origin = False

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


def timestamp_unit(timestamps: np.ndarray) -> str:
    """``"ms"`` or ``"s"``: the unit of a raw accelerometer timestamp column.

    Inferred from the median sample spacing; no wearable samples slower than
    one sample per 10 s, so a spacing of 10 or more must be milliseconds.
    """
    median_dt = float(np.median(np.diff(np.asarray(timestamps, dtype=np.float64))))
    return "ms" if median_dt >= 10 else "s"


def origin_seconds(timestamps: np.ndarray, origin: OriginSpec) -> OriginSpec:
    """Convert an origin given on a raw timestamp column's clock to seconds.

    ``origin`` is in the column's own unit (see :func:`timestamp_unit`), so a
    PSG start read from the same data can be passed as it is. ``None`` and
    ``"unix"`` pass through.
    """
    if origin is None or isinstance(origin, str):
        return origin
    scale = 1e-3 if timestamp_unit(timestamps) == "ms" else 1.0
    return float(origin) * scale


class ComputeJerkNUFFT(ProcessingStep):
    """Raw (N, 4) accel array → senpy.JerkData with non-uniform timestamps."""

    def __init__(self, use_diff: bool = True):
        self.use_diff = use_diff

    @property
    def name(self) -> str:
        return "jerk_nufft"

    def transform(self, X: np.ndarray) -> "senpy.JerkData":
        timestamps = np.ascontiguousarray(X[..., 0])
        ts_unit = timestamp_unit(timestamps)
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

    takes_origin = True

    def transform(
        self, jerk: "senpy.JerkData", origin_s: OriginSpec = None
    ) -> "senpy.SpectrogramResult":
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
            origin_s=origin_s,
        )

    def transform_many(
        self,
        jerks: Sequence["senpy.JerkData"],
        origin_s: Union[OriginSpec, Sequence[OriginSpec]] = None,
    ) -> List["senpy.SpectrogramResult"]:
        origins = _per_recording(origin_s, len(jerks))
        if self.nufft_backend != "jax":
            return [self.transform(jerk, origin) for jerk, origin in zip(jerks, origins)]
        grouped = _compute_packed_jax_recording_spectrograms(
            recordings=[(jerk.timestamps_s, [jerk.jerk]) for jerk in jerks],
            window_s=self.secperseg,
            overlap_s=self.secoverlap,
            target_fs=self.target_fs if self.target_fs > 0.0 else None,
            kind=self.spectrogram_kind,
            detrend=self.detrend,
            backend_kwargs=self.nufft_backend_kwargs,
            origins_s=origins,
        )
        return [specs[0] for specs in grouped]


def _regularise(result, hop: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One frame per window on senpy's grid; windows without data hold the padding value.

    Frame ``j`` is window ``j``: ``[origin + j * hop, origin + j * hop + window)``,
    timestamped at its centre. A window fills its frame when senpy marked it
    valid and every value in it is finite. Returns
    ``(frame_times, dense_rows, frame_valid)``.
    """
    if result.window_index is None:
        raise ValueError(
            "regularising needs each row's window_index; compute the spectrogram "
            "with empty_windows='keep'"
        )
    window_index = np.asarray(result.window_index, dtype=np.int64)
    rows = result.Sxx
    n_frames = int(window_index[-1]) + 1
    hop = float(hop)
    half_window = float(result.times[0]) - int(window_index[0]) * hop
    frame_times = half_window + hop * np.arange(n_frames, dtype=np.float64)
    usable = np.asarray(result.valid, dtype=bool) & np.all(
        np.isfinite(rows.reshape(rows.shape[0], -1)), axis=1
    )
    dense = np.full((n_frames,) + rows.shape[1:], SPECTROGRAM_PADDING_VALUE, dtype=rows.dtype)
    dense[window_index[usable]] = rows[usable]
    frame_valid = np.zeros(n_frames, dtype=bool)
    frame_valid[window_index[usable]] = True
    return frame_times, dense, frame_valid


class RegulariseNUFFTGrid(ProcessingStep):
    """NUFFT spectrogram on senpy's window grid → one row per frame, padding where empty.

    Frame ``j`` is window ``j`` of the grid; see :func:`_regularise`.
    """

    def __init__(self, hop_seconds: float):
        self.hop_seconds = hop_seconds

    @property
    def name(self) -> str:
        return "regularise_nufft_grid"

    def transform(self, result: "senpy.SpectrogramResult") -> "senpy.SpectrogramResult":
        if len(result.times) == 0:
            return result
        frame_times, dense_Sxx, frame_valid = _regularise(result, self.hop_seconds)
        return senpy.SpectrogramResult(
            frequencies=result.frequencies,
            times=frame_times,
            Sxx=dense_Sxx,
            kind=result.kind,
            method=result.method,
            valid=frame_valid,
            origin_s=result.origin_s,
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

    def transform(self, X, origin_s: OriginSpec = None):
        """Run every step; ``origin_s`` reaches the steps that place windows."""
        x_out = X
        for step in self.substeps:
            t0 = time.monotonic()
            try:
                _log.info("step %s starting", step.name)
                if step.takes_origin:
                    x_out = step.transform(x_out, origin_s=origin_s)
                else:
                    x_out = step.transform(x_out)
            except BaseException:
                _log.exception("step %s raised after %.2fs", step.name, time.monotonic() - t0)
                raise
            shape = getattr(x_out, "shape", None)
            if shape is None and hasattr(x_out, "Sxx"):
                shape = ("Sxx=", x_out.Sxx.shape)
            _log.info("step %s done in %.2fs → %s", step.name, time.monotonic() - t0, shape)
        return x_out

    def transform_many(
        self,
        X: Sequence[InputT],
        origin_s: Union[OriginSpec, Sequence[OriginSpec]] = None,
    ) -> List[OutputT]:
        """Run every step over all recordings; ``origin_s`` is one origin or one per recording."""
        x_out = list(X)
        origins = _per_recording(origin_s, len(x_out))
        for step in self.substeps:
            t0 = time.monotonic()
            try:
                _log.info("step %s starting for %d recordings", step.name, len(x_out))
                if step.takes_origin:
                    x_out = step.transform_many(x_out, origin_s=origins)
                else:
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


def _per_recording(
    origin_s: Union[OriginSpec, Sequence[OriginSpec]], n: int
) -> List[OriginSpec]:
    """One origin per recording from a single origin or a sequence of them."""
    if origin_s is None or isinstance(origin_s, (str, int, float, np.number)):
        return [origin_s] * n
    origins = list(origin_s)
    if len(origins) != n:
        raise ValueError(f"origin_s must be one origin or one per recording ({n}), got {len(origins)}")
    return origins


def _normalize_spectrogram_kind(kind: str) -> str:
    normalized = str(kind).replace("-", "_").lower()
    aliases = {
        "mag": "magnitude",
        "magnitude": "magnitude",
        "power": "power",
        "psd": "psd",
        "log_psd": "log_psd",
    }
    if normalized not in aliases:
        raise ValueError(
            "spectrogram_kind must be one of: 'mag', 'magnitude', 'power', 'psd', 'log_psd'"
        )
    return aliases[normalized]


# log(PSD + eps) uses the float64 machine epsilon, matching the log-power
# spectrograms of Olsen et al. 2022 (SleepStagePrediction), whose code adds
# sys.float_info.epsilon before the log.
_LOG_PSD_EPSILON = float(np.finfo(np.float64).eps)


def _senpy_kind(kind: str) -> str:
    """The kind to request from senpy; 'log_psd' is derived from 'psd' here."""
    return "psd" if kind == "log_psd" else kind


def _grid_metadata(result) -> Dict[str, object]:
    """The per-row window metadata a senpy result carries, to copy onto a derived one."""
    return {
        "window_index": result.window_index,
        "sample_count": result.sample_count,
        "valid": result.valid,
        "origin_s": result.origin_s,
    }


def _log_psd_spectrogram(result: "senpy.SpectrogramResult") -> "senpy.SpectrogramResult":
    return senpy.SpectrogramResult(
        frequencies=result.frequencies,
        times=result.times,
        Sxx=np.log(result.Sxx + _LOG_PSD_EPSILON),
        kind="log_psd",
        method=result.method,
        **_grid_metadata(result),
    )


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


#: Every NUFFT call here asks senpy for the full window grid: windows with too
#: few samples come back as NaN rows with ``valid`` False, and the regularise
#: steps turn them into padding frames.
_EMPTY_WINDOWS = "keep"


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
    origin_s: OriginSpec = None,
) -> "senpy.SpectrogramResult":
    backend = _normalize_nufft_backend(backend)
    if backend == "cpu":
        _take_backend_kwargs(backend, backend_kwargs, ())
        result = senpy.compute_nufft_spectrogram(
            timestamps=timestamps,
            signal=signal,
            window_s=window_s,
            overlap_s=overlap_s,
            target_fs=target_fs,
            kind=_senpy_kind(kind),
            detrend=detrend,
            origin_s=origin_s,
            empty_windows=_EMPTY_WINDOWS,
        )
        return _log_psd_spectrogram(result) if kind == "log_psd" else result
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
            origin_s=origin_s,
            empty_windows=_EMPTY_WINDOWS,
            **kwargs,
        )
        spectrogram = result.spectrogram(_senpy_kind(kind))
        return _log_psd_spectrogram(spectrogram) if kind == "log_psd" else spectrogram

    return _compute_packed_jax_spectrograms(
        timestamps=timestamps,
        signals=[signal],
        window_s=window_s,
        overlap_s=overlap_s,
        target_fs=target_fs,
        kind=kind,
        detrend=detrend,
        backend_kwargs=backend_kwargs,
        origin_s=origin_s,
    )[0]


def _spectral_surface(coefficients: np.ndarray, kind: str) -> np.ndarray:
    magnitude = np.abs(coefficients)
    if kind == "magnitude":
        return magnitude
    power = magnitude * magnitude
    if kind == "log_psd":
        return np.log(power + _LOG_PSD_EPSILON)
    return power


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
    origin_s: OriginSpec = None,
) -> List["senpy.SpectrogramResult"]:
    """Transform the channels of one recording on the JAX backend."""
    return _compute_packed_jax_recording_spectrograms(
        recordings=[(timestamps, signals)],
        window_s=window_s,
        overlap_s=overlap_s,
        target_fs=target_fs,
        kind=kind,
        detrend=detrend,
        backend_kwargs=backend_kwargs,
        origins_s=[origin_s],
    )[0]


#: ``nufft_backend_kwargs`` the JAX backend accepts, passed to
#: ``senpy.jax_backend.compute_nustft_many``.
_JAX_BACKEND_OPTIONS = ("eps", "rows_per_call", "max_in_flight", "build_threads")


def _compute_packed_jax_recording_spectrograms(
    *,
    recordings: Sequence[tuple[np.ndarray, Sequence[np.ndarray]]],
    window_s: float,
    overlap_s: float,
    target_fs: Optional[float],
    kind: str,
    detrend: bool,
    backend_kwargs: Optional[dict],
    origins_s: Optional[Sequence[OriginSpec]] = None,
) -> List[List["senpy.SpectrogramResult"]]:
    """Transform every channel of every recording in one senpy ``compute_nustft_many`` call.

    Returns ``[recording][channel]`` spectrograms on senpy's window grid.
    """
    kwargs = _take_backend_kwargs("jax", backend_kwargs, _JAX_BACKEND_OPTIONS)

    from senpy import jax_backend as senpy_jax

    recordings = list(recordings)
    origins = _per_recording(origins_s, len(recordings))
    for recording_index, (_, signals) in enumerate(recordings):
        if len(signals) == 0:
            raise ValueError(f"JAX recording {recording_index} has no signals")

    results = senpy_jax.compute_nustft_many(
        [(timestamps, np.column_stack(signals)) for timestamps, signals in recordings],
        window_s=window_s,
        overlap_s=overlap_s,
        target_fs=target_fs,
        detrend=detrend,
        origin_s=origins,
        empty_windows=_EMPTY_WINDOWS,
        **kwargs,
    )

    spectrograms: List[List["senpy.SpectrogramResult"]] = []
    for recording_index, channels in enumerate(results):
        if len(channels[0].times) == 0:
            raise ValueError(
                "JAX NUSTFT requires enough data for at least one window "
                f"in recording {recording_index}"
            )
        spectrograms.append(
            [
                senpy.SpectrogramResult(
                    frequencies=channel.frequencies,
                    times=channel.times,
                    Sxx=_spectral_surface(channel.coefficients, kind),
                    kind=kind,
                    method="jax_finufft_packed",
                    **_grid_metadata(channel),
                )
                for channel in channels
            ]
        )
    return spectrograms


class ComputeStackedSpectrogramsNUFFT(ProcessingStep):
    """Raw ``(N, 4)`` accel → ``StackedSpectrogramResult`` ``(T, F, C)`` via per-channel NUFFT.

    Each requested channel (x, y, z, mag, jerk) is transformed independently with
    the same NUFFT parameters and window grid, then stacked along the last axis.
    """

    takes_origin = True

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

    def transform(
        self, X: np.ndarray, origin_s: OriginSpec = None
    ) -> "senpy.StackedSpectrogramResult":
        if self.nufft_backend == "cpu":
            _take_backend_kwargs("cpu", self.nufft_backend_kwargs, ())
            stacked = senpy.compute_stacked_spectrograms(
                accel=self._prepare_accelerometer(X),
                window_s=self.secperseg,
                overlap_s=self.secoverlap,
                target_fs=self.target_fs if self.target_fs > 0.0 else None,
                kind=_senpy_kind(self.spectrogram_kind),
                detrend=self.detrend,
                channels=self.channels,
                use_diff=self.use_diff,
                origin_s=origin_s,
                empty_windows=_EMPTY_WINDOWS,
            )
            if self.spectrogram_kind == "log_psd":
                stacked = senpy.StackedSpectrogramResult(
                    frequencies=stacked.frequencies,
                    times=stacked.times,
                    Sxx=np.log(stacked.Sxx + _LOG_PSD_EPSILON),
                    channels=stacked.channels,
                    kind="log_psd",
                    **_grid_metadata(stacked),
                )
            return stacked
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
                origin_s=origin_s,
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
                    origin_s=origin_s,
                )
                for signal in signals
            ]
        return _stack_spectrogram_results(specs, channels)

    @staticmethod
    def _prepare_accelerometer(X: np.ndarray) -> "senpy.AccelerometerData":
        timestamps_raw = np.ascontiguousarray(X[..., 0], dtype=np.float64)
        conversion = 1e3 if timestamp_unit(timestamps_raw) == "ms" else 1e6
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
        self,
        arrays: Sequence[np.ndarray],
        origin_s: Union[OriginSpec, Sequence[OriginSpec]] = None,
    ) -> List["senpy.StackedSpectrogramResult"]:
        origins = _per_recording(origin_s, len(arrays))
        if self.nufft_backend != "jax":
            return [self.transform(X, origin) for X, origin in zip(arrays, origins)]
        prepared = [self._prepare_signals(X) for X in arrays]
        grouped_specs = _compute_packed_jax_recording_spectrograms(
            recordings=[(timestamps, signals) for timestamps, signals, _ in prepared],
            window_s=self.secperseg,
            overlap_s=self.secoverlap,
            target_fs=self.target_fs if self.target_fs > 0.0 else None,
            kind=self.spectrogram_kind,
            detrend=self.detrend,
            backend_kwargs=self.nufft_backend_kwargs,
            origins_s=origins,
        )
        return [
            _stack_spectrogram_results(specs, channels)
            for specs, (_, _, channels) in zip(grouped_specs, prepared, strict=True)
        ]


def _stack_spectrogram_results(
    specs: Sequence["senpy.SpectrogramResult"],
    channels: List[str],
) -> "senpy.StackedSpectrogramResult":
    """Stack per-channel spectrograms that share one window grid, row by window index.

    Every channel is on the same grid (same timestamps, same origin), so a
    window index names the same window in each. A grid row is valid only when
    every channel has data there.
    """
    ref = max(specs, key=lambda spec: len(spec.window_index))
    n_windows = len(ref.window_index)
    Sxx = np.full(
        (n_windows, len(ref.frequencies), len(channels)),
        np.nan,
        dtype=np.float64,
    )
    valid = np.ones(n_windows, dtype=bool)
    for channel_index, (channel, spec) in enumerate(zip(channels, specs, strict=True)):
        if not np.array_equal(spec.frequencies, ref.frequencies):
            raise ValueError(f"NUFFT backend produced an incompatible grid for {channel!r}")
        Sxx[spec.window_index, :, channel_index] = spec.Sxx
        channel_valid = np.zeros(n_windows, dtype=bool)
        channel_valid[spec.window_index] = spec.valid
        valid &= channel_valid
    return senpy.StackedSpectrogramResult(
        frequencies=ref.frequencies,
        times=ref.times,
        Sxx=Sxx,
        channels=channels,
        kind=ref.kind,
        window_index=ref.window_index,
        sample_count=ref.sample_count,
        valid=valid,
        origin_s=ref.origin_s,
    )


class RegulariseStackedNUFFTGrid(ProcessingStep):
    """Stacked spectrogram on senpy's window grid → one ``(F, C)`` slab per frame, padding where empty.

    Frame ``j`` is window ``j`` of the grid; see :func:`_regularise`.
    """

    def __init__(self, hop_seconds: float):
        self.hop_seconds = hop_seconds

    @property
    def name(self) -> str:
        return "regularise_stacked_nufft_grid"

    def transform(
        self, result: "senpy.StackedSpectrogramResult"
    ) -> "senpy.StackedSpectrogramResult":
        if len(result.times) == 0:
            return result
        frame_times, dense_Sxx, frame_valid = _regularise(result, self.hop_seconds)
        return senpy.StackedSpectrogramResult(
            frequencies=result.frequencies,
            times=frame_times,
            Sxx=dense_Sxx,
            channels=result.channels,
            kind=result.kind,
            valid=frame_valid,
            origin_s=result.origin_s,
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
