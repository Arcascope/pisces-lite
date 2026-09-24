"""ModelIOBundle + feature-cache path helpers.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

from pisces_lite.datasets.constants import (
    PSG_5C_MAPPING_TO_WLDR,
    PSG_5C_MAPPING_TO_WNR,
    PSG_5C_MAPPING_TO_WS,
)


@dataclass
class ModelIOBundle:
    """Container for model inputs/outputs keyed by subject.

    Shape conventions:
      - ``X``: ``(B, T, F)`` (or ``(1, T, F)`` for a single subject)
      - ``y``: ``(B, T)``
      - ``datasets``, ``subject_ids``: ``(B,)`` of dtype ``<U...``

    ``time`` and ``freq`` are optional per-bundle metadata shared across the
    batch dimension.
    """

    X: np.ndarray
    y: np.ndarray
    datasets: np.ndarray
    subject_ids: np.ndarray
    time: Optional[np.ndarray] = None
    freq: Optional[np.ndarray] = None

    @classmethod
    def from_npz(cls, npz_path: "Path | str") -> "ModelIOBundle":
        data = np.load(npz_path, allow_pickle=True)
        return cls(
            X=data["X"],
            y=data["y"],
            datasets=data["datasets"],
            subject_ids=data["subject_ids"],
            time=data["time"] if "time" in data.files else None,
            freq=data["freq"] if "freq" in data.files else None,
        )

    def save_npz(self, npz_path: "Path | str") -> None:
        np.savez_compressed(
            npz_path,
            X=self.X,
            y=self.y,
            datasets=self.datasets,
            subject_ids=self.subject_ids,
            time=self.time if self.time is not None else np.array([]),
            freq=self.freq if self.freq is not None else np.array([]),
        )

    @staticmethod
    def _ensure_batch(arr: np.ndarray, expected_ndim: int) -> np.ndarray:
        arr = np.asarray(arr)
        if arr.ndim > expected_ndim:
            raise ValueError(
                f"Cannot coerce {arr.ndim}-D array into {expected_ndim}-D; "
                f"shape={arr.shape}"
            )
        while arr.ndim < expected_ndim:
            arr = arr[np.newaxis, ...]
        return arr

    def __getitem__(self, key) -> "ModelIOBundle":
        return ModelIOBundle(
            X=self._ensure_batch(self.X[key], self.X.ndim),
            y=self._ensure_batch(self.y[key], self.y.ndim),
            datasets=self._ensure_batch(self.datasets[key], self.datasets.ndim),
            subject_ids=self._ensure_batch(self.subject_ids[key], self.subject_ids.ndim),
            time=self.time,
            freq=self.freq,
        )

    def __len__(self) -> int:
        return len(self.subject_ids)

    @classmethod
    def combine(cls, bundles: List["ModelIOBundle"]) -> "ModelIOBundle":
        if not bundles:
            raise ValueError("No bundles to combine")
        return cls(
            X=np.concatenate([b.X for b in bundles], axis=0),
            y=np.concatenate([b.y for b in bundles], axis=0),
            datasets=np.concatenate([b.datasets for b in bundles], axis=0),
            subject_ids=np.concatenate([b.subject_ids for b in bundles], axis=0),
            time=bundles[0].time,
            freq=bundles[0].freq,
        )

    def y_for_n_classes(self, num_classes: int) -> np.ndarray:
        """Return ``self.y`` remapped into ``num_classes`` sleep stages."""
        mapping = {
            2: PSG_5C_MAPPING_TO_WS,
            3: PSG_5C_MAPPING_TO_WNR,
            4: PSG_5C_MAPPING_TO_WLDR,
        }.get(num_classes, {})
        y = np.asarray(self.y)
        if not mapping or y.size == 0:
            return y
        return np.vectorize(lambda x: mapping.get(x, x), otypes=[y.dtype])(y)

    @property
    def has_freq(self) -> bool:
        return self.freq is not None and np.asarray(self.freq).size > 0


def get_features_cache_name(dataset_name: str, subject_id: Optional[str]) -> str:
    """``{dataset}[_subject_{id}]_features.npz``"""
    base = dataset_name if subject_id is None else f"{dataset_name}_subject_{subject_id}"
    return f"{base}_features.npz"


def resolve_feature_cache_dir(
    processing_config,
    feature_cache_dir: "Path | str | None",
    cache_prefix: str | None = None,
) -> Optional[Path]:
    """``{feature_cache_dir}/{cache_prefix}/`` — the per-run cache directory.
    If `cache_prefix` is `None`, defaults to `{type}/{fs}Hz`.

    Returns ``None`` when ``feature_cache_dir`` is ``None``.
    """
    if feature_cache_dir is None:
        return None
    if cache_prefix is not None:
        subdirectory = cache_prefix
    else:
        subdirectory = f"{processing_config.type}/{processing_config.fs}Hz"
    return Path(feature_cache_dir) / subdirectory

