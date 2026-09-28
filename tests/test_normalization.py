"""Padding frames take no part in normalization and stay padding afterwards."""
from __future__ import annotations

import numpy as np
import pytest

from pisces_lite.proc import SPECTROGRAM_PADDING_VALUE, ProcessingConfig, frame_validity

PAD = SPECTROGRAM_PADDING_VALUE


def _features(shape, seed=0) -> np.ndarray:
    return np.random.default_rng(seed).normal(-8.0, 2.0, shape)


@pytest.mark.parametrize("mode", ["znorm_axis0", "znorm_axis1", "original_norm"])
@pytest.mark.parametrize("shape", [(40, 6), (40, 6, 3), (40, 1)])
def test_padding_frames_do_not_move_the_statistics(mode, shape) -> None:
    config = ProcessingConfig(type="nufft", normalization_mode=mode, normalize_below=100.0)
    clean = _features(shape)
    padded = clean.copy()
    padded[[0, 1, 17]] = PAD

    got = config.normalize(padded)
    want = config.normalize(np.delete(clean, [0, 1, 17], axis=0))

    np.testing.assert_allclose(np.delete(got, [0, 1, 17], axis=0), want)
    assert np.all(got[[0, 1, 17]] == PAD)
    np.testing.assert_array_equal(frame_validity(got), frame_validity(padded))


def test_saved_stats_also_keep_padding() -> None:
    config = ProcessingConfig(
        type="nufft",
        norm_stats={"ds": {"mean": [1.0, 2.0], "denom": [2.0, 2.0]}},
    )
    features = np.array([[3.0, 4.0], [PAD, PAD]])

    np.testing.assert_allclose(
        config.normalize(features, data_set_name="ds"), [[1.0, 1.0], [PAD, PAD]], atol=1e-6
    )


def test_all_padding_and_none_mode_are_left_alone() -> None:
    features = np.full((5, 3), PAD)

    np.testing.assert_array_equal(ProcessingConfig(type="nufft").normalize(features), features)
    mixed = _features((5, 3))
    mixed[2] = PAD
    mixed[3, 1] = np.nan
    out = ProcessingConfig(type="nufft", normalization_mode="none").normalize(mixed)
    assert np.all(out[2] == PAD) and out[3, 1] == 0.0
