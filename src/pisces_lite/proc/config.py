"""Vendored subset of pisces2.configuration.processing_configuration.ProcessingConfig.

Only the inference path (``apply`` / ``extract_features`` / ``normalize``) is
preserved. Training-only entry points (``fit_transform`` / ``transform``)
are intentionally omitted — they depended on pisces2.utils and are not used
by the inference path.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class ProcessingConfig:
    type: str
    fs: float = 32.0
    window_seconds: int = 32
    window_step_seconds: int = 15
    fmin: float = 0.0
    fmax: Optional[float] = None
    time_downsample_rate: int = 1
    normalization_mode: str = "znorm_axis0"
    feature_index: int = 0
    normalize_below: float = 4.0
    target_X_length: Optional[int] = None
    target_Y_length: Optional[int] = None
    norm_stats: Optional[Dict[str, Dict[str, list]]] = field(default=None)
    make_plots: bool = True
    jerk_diff: bool = True
    detrend: bool = True
    spectrogram_kind: str = "magnitude"
    spectral_channels: Optional[List[str]] = None
    nufft_backend: str = "streaming"
    nufft_backend_kwargs: Dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_json(cls, path: "Path | str") -> "ProcessingConfig":
        with open(path) as f:
            return cls.from_dict(json.load(f))

    @classmethod
    def from_dict(cls, d: dict) -> "ProcessingConfig":
        d = dict(d)
        if "kwargs" in d:
            d.update(d.pop("kwargs"))
        for alias in ("spectrogram_mode", "stft_mode", "stft_kind", "kind", "mode"):
            if alias in d and "spectrogram_kind" not in d:
                d["spectrogram_kind"] = d.pop(alias)
        if "backend" in d and "nufft_backend" not in d:
            d["nufft_backend"] = d.pop("backend")
        if "backend_kwargs" in d and "nufft_backend_kwargs" not in d:
            d["nufft_backend_kwargs"] = d.pop("backend_kwargs")
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})

    @property
    def nfft(self) -> int:
        return int(self.fs * self.window_seconds)

    @property
    def hop_length(self) -> int:
        return int(self.fs * self.window_step_seconds)

    def get_frequencies(self) -> np.ndarray:
        all_freq = np.fft.rfftfreq(n=self.nfft, d=1 / self.fs)
        filt_freq = all_freq >= self.fmin
        if self.fmax is not None:
            filt_freq &= all_freq <= self.fmax
        return all_freq[filt_freq]

    def _build_pipeline(self):
        if self.type != "nufft":
            raise NotImplementedError(
                f"pisces_lite.proc only vendors the 'nufft' pipeline; got type={self.type!r}."
            )

        secoverlap = self.window_seconds - self.window_step_seconds
        feature_kwargs = {
            "fmin": self.fmin,
            "fmax": self.fmax,
            "time_downsample_rate": self.time_downsample_rate,
        }

        if self.spectral_channels is not None:
            from pisces_lite.proc.processing import stacked_nufft_pipeline

            return stacked_nufft_pipeline(
                secperseg=self.window_seconds,
                secoverlap=secoverlap,
                target_fs=self.fs,
                channels=self.spectral_channels,
                feature_kwargs=feature_kwargs,
                use_diff=self.jerk_diff,
                detrend=self.detrend,
                spectrogram_kind=self.spectrogram_kind,
                nufft_backend=self.nufft_backend,
                nufft_backend_kwargs=self.nufft_backend_kwargs,
            )

        from pisces_lite.proc.processing import nufft_based_features

        return nufft_based_features(
            secperseg=self.window_seconds,
            secoverlap=secoverlap,
            features=["spectrogram"],
            target_fs=self.fs,
            feature_kwargs=feature_kwargs,
            use_diff=self.jerk_diff,
            detrend=self.detrend,
            spectrogram_kind=self.spectrogram_kind,
            nufft_backend=self.nufft_backend,
            nufft_backend_kwargs=self.nufft_backend_kwargs,
        )

    def extract_features(self, accel: np.ndarray) -> np.ndarray:
        pipeline = self._build_pipeline()
        features = pipeline.transform(accel)
        if features.ndim == 1:
            features = features[:, np.newaxis]
        return features

    def extract_features_many(
        self, accel_arrays: Sequence[np.ndarray]
    ) -> List[np.ndarray]:
        """Extract several recordings, allowing accelerator backends to pack work."""
        pipeline = self._build_pipeline()
        features_many = pipeline.transform_many(list(accel_arrays))
        return [
            features[:, np.newaxis] if features.ndim == 1 else features
            for features in features_many
        ]

    def normalize(
        self,
        features: np.ndarray,
        data_set_name: Optional[str] = None,
    ) -> np.ndarray:
        features = features.copy()
        features[np.isnan(features)] = 0.0

        if self.normalization_mode == "none":
            return features

        if features.ndim == 3:
            # Per-channel z-normalization for stacked (T, F, C) spectrograms.
            # keepdims broadcasts over T or F; the C axis is normalised independently.
            if self.normalization_mode == "znorm_axis1":
                mean = np.mean(features, axis=1, keepdims=True)
                denom = np.std(features, axis=1, keepdims=True)
            elif data_set_name is not None:
                # Per-channel scalar stats, stored shape (C,) -> broadcast (1, 1, C).
                mean, denom = self._lookup_stats(data_set_name)
                mean = mean.reshape(1, 1, -1)
                denom = denom.reshape(1, 1, -1)
            else:
                # Per-channel scalar self-stats: reduce over both T and F.
                mean = np.mean(features, axis=(0, 1), keepdims=True)
                denom = np.std(features, axis=(0, 1), keepdims=True)
            return (features - mean) / (denom + 1e-7)

        if self.normalization_mode == "znorm_axis1":
            norm_axis = 0 if features.shape[1] == 1 else 1
            mean = np.mean(features, axis=norm_axis, keepdims=True)
            denom = np.std(features, axis=norm_axis, keepdims=True)
            return (features - mean) / (denom + 1e-7)

        if data_set_name is not None:
            mean, denom = self._lookup_stats(data_set_name)
        else:
            mean, denom = self._compute_stats(features)
        return (features - mean) / (denom + 1e-7)

    def _lookup_stats(self, data_set_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load saved (mean, denom) for a dataset, tolerating the legacy ``std`` key."""
        if self.norm_stats is None or data_set_name not in self.norm_stats:
            raise RuntimeError(
                f"No normalization stats for dataset {data_set_name!r}."
            )
        stats = self.norm_stats[data_set_name]
        mean = np.array(stats["mean"])
        denom = np.array(stats["denom"] if "denom" in stats else stats["std"])
        return mean, denom

    def _compute_stats(self, all_X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        region = all_X[:, self.feature_index] < self.normalize_below
        if not np.any(region):
            region = np.ones(len(all_X), dtype=bool)
        mean = np.mean(all_X[region], axis=0)
        if self.normalization_mode == "original_norm":
            denom = np.mean(all_X[region], axis=0)
        else:
            denom = np.std(all_X[region], axis=0)
        return mean,denom 

    def apply(
        self,
        accel_array: np.ndarray,
        data_set_name: Optional[str] = None,
        normalize: bool = True,
        padding: bool = False,
    ) -> np.ndarray:
        features = self.extract_features(accel_array)
        if normalize:
            features = self.normalize(features, data_set_name=data_set_name)
        if padding:
            raise NotImplementedError(
                "padding=True was part of training; inference path does not pad."
            )
        return features

    def apply_many(
        self,
        accel_arrays: Sequence[np.ndarray],
        data_set_names: Optional[Sequence[Optional[str]]] = None,
        normalize: bool = True,
        padding: bool = False,
    ) -> List[np.ndarray]:
        """Apply one config to several recordings without changing their order.

        Accelerator-backed pipeline steps may override ``transform_many`` to
        pack work across recordings. CPU-backed steps retain the established
        per-recording path.
        """
        accel_arrays = list(accel_arrays)
        if data_set_names is None:
            data_set_names = [None] * len(accel_arrays)
        else:
            data_set_names = list(data_set_names)
            if len(data_set_names) != len(accel_arrays):
                raise ValueError("data_set_names must match accel_arrays length")
        if padding:
            raise NotImplementedError(
                "padding=True was part of training; inference path does not pad."
            )

        features_many = self.extract_features_many(accel_arrays)
        if normalize:
            return [
                self.normalize(features, data_set_name=data_set_name)
                for features, data_set_name in zip(features_many, data_set_names)
            ]
        return features_many
