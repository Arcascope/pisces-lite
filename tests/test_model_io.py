"""Tests for pisces_lite.model_io: cache paths, bundles, and class remapping."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from pisces_lite.model_io import (
    ModelIOBundle,
    get_features_cache_name,
    resolve_feature_cache_dir,
)


class _FakeProcessingConfig:
    type = "nufft"
    fs = 32.0


def test_resolve_feature_cache_dir_with_prefix(tmp_path):
    out = resolve_feature_cache_dir(_FakeProcessingConfig(), tmp_path, "12Hz_log_PSD")
    assert out == Path(tmp_path) / "12Hz_log_PSD"


def test_resolve_feature_cache_dir_defaults_to_type_and_fs(tmp_path):
    # cache_prefix=None must fall back to "{type}/{fs}Hz" without blowing up.
    out = resolve_feature_cache_dir(_FakeProcessingConfig(), tmp_path)
    assert out == Path(tmp_path) / "nufft" / "32.0Hz"


def test_resolve_feature_cache_dir_none_root():
    assert resolve_feature_cache_dir(_FakeProcessingConfig(), None) is None


def test_get_features_cache_name():
    assert get_features_cache_name("DREAMT", "S001") == "DREAMT_subject_S001_features.npz"
    assert get_features_cache_name("DREAMT", None) == "DREAMT_features.npz"


def _bundle(n: int, num_stages: int = 5) -> ModelIOBundle:
    return ModelIOBundle(
        X=np.zeros((n, 3, 2)),
        y=np.tile(np.arange(num_stages, dtype=int), (n, 1)),
        datasets=np.array([f"d{i}" for i in range(n)]),
        subject_ids=np.array([f"s{i}" for i in range(n)]),
    )


def test_y_for_n_classes_remaps_stages():
    bundle = _bundle(1)
    # 5-class source (0..4) -> 2-class sleep/wake maps 1..4 to sleep.
    assert bundle.y_for_n_classes(2).tolist() == [[0, 1, 1, 1, 1]]


def test_y_for_n_classes_empty_bundle_does_not_raise():
    bundle = _bundle(0)
    out = bundle.y_for_n_classes(4)
    assert out.shape == (0, 5)
