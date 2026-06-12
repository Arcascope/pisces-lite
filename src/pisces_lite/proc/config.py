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
from typing import Dict, List, Optional, Tuple

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
        )

    def extract_features(self, accel: np.ndarray) -> np.ndarray:
        pipeline = self._build_pipeline()
        features = pipeline.transform(accel)
        if features.ndim == 1:
            features = features[:, np.newaxis]
        return features

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
            result = np.empty_like(features)
            for c in range(features.shape[2]):
                ch = features[:, :, c]
                if self.normalization_mode == "znorm_axis1":
                    mean = np.mean(ch, axis=1, keepdims=True)
                    std = np.std(ch, axis=1, keepdims=True)
                else:
                    mean = np.mean(ch, axis=0, keepdims=True)
                    std = np.std(ch, axis=0, keepdims=True)
                result[:, :, c] = (ch - mean) / (std + 1e-7)
            return result

        if self.normalization_mode == "znorm_axis1":
            norm_axis = 0 if features.shape[1] == 1 else 1
            mean = np.mean(features, axis=norm_axis, keepdims=True)
            std = np.std(features, axis=norm_axis, keepdims=True)
            return (features - mean) / (std + 1e-7)

        if data_set_name is not None:
            if self.norm_stats is None or data_set_name not in self.norm_stats:
                raise RuntimeError(
                    f"No normalization stats for dataset {data_set_name!r}."
                )
            mean = np.array(self.norm_stats[data_set_name]["mean"])
            std = np.array(self.norm_stats[data_set_name]["std"])
        else:
            mean, std = self._compute_stats(features)
        return (features - mean) / (std + 1e-7)

    def _compute_stats(self, all_X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        region = all_X[:, self.feature_index] < self.normalize_below
        if not np.any(region):
            region = np.ones(len(all_X), dtype=bool)
        mean = np.mean(all_X[region], axis=0)
        if self.normalization_mode == "original_norm":
            std = np.mean(all_X[region], axis=0)
        else:
            std = np.std(all_X[region], axis=0)
        return mean, std

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
