"""The opt-in gap mode: epochs whose spectrogram frames are mostly excluded.

The mode is off by default, so every existing config keeps its labels; these
tests pin that down alongside the masking rule itself.
"""
from __future__ import annotations

import numpy as np
import pytest

from pisces_lite.datasets import PAD_CLASS_LABEL, mask_labels_by_frame_coverage
from pisces_lite.proc import SPECTROGRAM_PADDING_VALUE, ProcessingConfig, frame_validity

PAD = SPECTROGRAM_PADDING_VALUE


def test_frame_validity_flags_only_all_sentinel_frames() -> None:
    features = np.full((4, 3, 2), -12.0)
    features[1] = PAD
    features[2, 0, 0] = PAD  # one sentinel value is not an excluded frame

    np.testing.assert_array_equal(frame_validity(features), [True, False, True, True])


def test_epochs_over_the_threshold_become_gap() -> None:
    labels = np.array([2, 2, 5])
    valid = np.ones(9, dtype=bool)
    valid[3:5] = False  # epoch 1: 2 of 3 excluded
    valid[6] = False  # epoch 2: 1 of 3 excluded

    masked = mask_labels_by_frame_coverage(labels, valid, frames_per_epoch=3)

    np.testing.assert_array_equal(masked, [2, PAD_CLASS_LABEL, 5])
    np.testing.assert_array_equal(labels, [2, 2, 5])  # input untouched


def test_exactly_half_excluded_is_not_gap() -> None:
    valid = np.array([True, False, True, False])

    masked = mask_labels_by_frame_coverage(np.array([1, 1]), valid, frames_per_epoch=2)

    np.testing.assert_array_equal(masked, [1, 1])


def test_frames_missing_past_the_end_count_as_excluded() -> None:
    masked = mask_labels_by_frame_coverage(
        np.array([0, 0]), np.ones(4, dtype=bool), frames_per_epoch=3
    )

    np.testing.assert_array_equal(masked, [0, PAD_CLASS_LABEL])


@pytest.mark.parametrize("fraction", [-0.1, 1.0, 1.5])
def test_threshold_outside_zero_one_is_refused(fraction: float) -> None:
    with pytest.raises(ValueError, match="max_excluded_fraction"):
        mask_labels_by_frame_coverage(np.array([0]), np.ones(2, bool), 2, fraction)
    with pytest.raises(ValueError, match="gap_max_excluded_frame_fraction"):
        ProcessingConfig(type="nufft", gap_max_excluded_frame_fraction=fraction)


def test_gap_mode_is_off_by_default() -> None:
    config = ProcessingConfig.from_dict({"type": "nufft", "window_step_seconds": 2})
    features = np.full((30, 3, 2), PAD)
    labels = np.array([2, 3])

    assert config.gap_max_excluded_frame_fraction is None
    np.testing.assert_array_equal(config.mask_gap_epochs(labels, features), labels)


def test_gap_mode_reads_from_json_config() -> None:
    config = ProcessingConfig.from_dict(
        {"type": "nufft", "window_step_seconds": 2, "gap_max_excluded_frame_fraction": 0.5}
    )
    features = np.full((30, 3, 2), -12.0)
    features[15:24] = PAD  # 9 of epoch 1's 15 frames

    np.testing.assert_array_equal(
        config.mask_gap_epochs(np.array([2, 3]), features), [2, PAD_CLASS_LABEL]
    )


def test_frames_per_epoch_follows_hop_and_downsampling() -> None:
    assert ProcessingConfig(type="nufft", window_step_seconds=2).frames_per_epoch() == 15
    assert (
        ProcessingConfig(type="nufft", window_step_seconds=2, time_downsample_rate=3)
        .frames_per_epoch()
        == 5
    )
    with pytest.raises(ValueError, match="whole number"):
        ProcessingConfig(type="nufft", window_step_seconds=4).frames_per_epoch()


def test_a_recording_dropout_marks_its_epoch_as_gap() -> None:
    pytest.importorskip("senpy", reason="needs the optional [proc] extra")
    fs = 50.0
    t = np.arange(0.0, 240.0, 1 / fs)
    keep = (t < 90.0) | (t >= 125.0)  # 35 s with no samples at all
    rng = np.random.default_rng(0)
    accel = np.column_stack(
        [t, rng.normal(0, 0.5, t.size), rng.normal(0, 0.5, t.size), 1 + rng.normal(0, 0.1, t.size)]
    )[keep]
    config = ProcessingConfig(
        type="nufft",
        fs=12.0,
        window_seconds=10,
        window_step_seconds=2,
        fmin=0.1,
        fmax=6.0,
        normalization_mode="none",
        spectrogram_kind="log_psd",
        spectral_channels=["mag", "jerk"],
        nufft_backend="cpu",
        gap_max_excluded_frame_fraction=0.5,
    )

    features = config.apply(accel, normalize=False)
    labels = config.mask_gap_epochs(np.full(8, 2), features)

    # Epoch 3 (90-120 s) sits inside the dropout; its neighbours keep data.
    assert labels[3] == PAD_CLASS_LABEL
    assert labels[1] == 2 and labels[5] == 2

    # Normalizing keeps the padding frames, so the mask is the same afterwards.
    normalized = config.normalize(features)
    np.testing.assert_array_equal(config.mask_gap_epochs(np.full(8, 2), normalized), labels)


def test_the_psg_origin_puts_each_epochs_frames_inside_it() -> None:
    pytest.importorskip("senpy", reason="needs the optional [proc] extra")
    fs = 50.0
    psg_start = 1_000.0
    # The accelerometer starts 7 s into the first epoch and drops out for
    # the whole of epoch 4 (1120-1150 s).
    t = psg_start + 7.0 + np.arange(int(233.0 * fs)) / fs
    t = t[(t < psg_start + 120.0) | (t >= psg_start + 150.0)]
    rng = np.random.default_rng(0)
    accel = np.column_stack(
        [t, rng.normal(0, 0.5, t.size), rng.normal(0, 0.5, t.size), 1 + rng.normal(0, 0.1, t.size)]
    )
    config = ProcessingConfig(
        type="nufft",
        fs=12.0,
        window_seconds=10,
        window_step_seconds=2,
        fmin=0.1,
        fmax=6.0,
        spectrogram_kind="log_psd",
        spectral_channels=["mag", "jerk"],
        nufft_backend="cpu",
        gap_max_excluded_frame_fraction=0.5,
    )

    features = config.apply(accel, origin=psg_start)
    valid = frame_validity(features)

    # Frame j is the window starting 2 j s after the PSG start. The last
    # one to fit the data starts at 230 s.
    assert features.shape[0] == 116
    # The first windows start before the accelerometer but hold data: all FFTs.
    assert valid[:4].all()
    # The dropout is 120-150 s. Windows starting at 120..140 s (frames 60..70)
    # lie inside it; the one starting at 118 s ends with 2 s of data before it,
    # and the one at 142 s begins 2 s before the data comes back.
    assert not valid[60:71].any()
    assert valid[59] and valid[71]
    # Epoch 4 (frames 60..74) has 11 of 15 frames without data. Epoch 7 (frames
    # 105..119) has the 11 windows that end before the data does.
    labels = config.mask_gap_epochs(np.full(8, 2), features)
    assert labels.tolist() == [2, 2, 2, 2, PAD_CLASS_LABEL, 2, 2, 2]
