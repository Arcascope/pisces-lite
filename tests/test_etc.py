"""Tests for the ETC ("easy to classify wake") model and scoring."""
from __future__ import annotations

import numpy as np
import pytest

from pisces_lite.metrics import ETCModel, auroc_numpy, etc_score


def test_kernel_spec_example():
    """The spec's worked example: 70 timestamps, width 0.5 -> std 17."""
    m = ETCModel(n_timestamps=70, kernel_width=0.5)
    assert m.std == 17
    k = m.kernel
    assert len(k) == 70
    assert pytest.approx(k.sum()) == 1.0
    center = 70 // 2
    assert np.argmax(k) == center
    # +/-1 sigma lands on indices 18 and 52, at exp(-0.5) of the peak
    peak = k[center]
    assert pytest.approx(k[center + m.std] / peak, abs=1e-9) == np.exp(-0.5)
    assert pytest.approx(k[center - m.std] / peak, abs=1e-9) == np.exp(-0.5)


def test_wake_proba_shapes_and_normalisation():
    m = ETCModel(n_timestamps=70, kernel_width=0.5)
    rng = np.random.default_rng(0)
    batch = rng.random((4, 70, 12))
    out = m.wake_proba(batch)
    assert out.shape == (4, 70)
    assert np.allclose(out.max(axis=1), 1.0)  # max 1.0 over the time axis
    # single recording (T, F) -> (T,)
    single = m.wake_proba(batch[0])
    assert single.shape == (70,)


def test_wake_proba_shorter_than_kernel():
    """A recording shorter than the kernel still returns exactly T samples."""
    m = ETCModel(n_timestamps=70, kernel_width=0.5)
    out = m.wake_proba(np.random.default_rng(0).random((40, 12)))
    assert out.shape == (40,)


def test_auroc_matches_sklearn():
    sklearn_metrics = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, size=500)
    s = rng.random(500)
    assert pytest.approx(auroc_numpy(y, s), abs=1e-9) == sklearn_metrics.roc_auc_score(y, s)


def test_auroc_handles_ties_and_empty_class():
    # all positives -> undefined
    assert np.isnan(auroc_numpy(np.ones(5), np.arange(5)))
    # tied scores -> 0.5
    assert pytest.approx(auroc_numpy([0, 1, 0, 1], [1.0, 1.0, 1.0, 1.0])) == 0.5


def _block_recording(rng, T=300, F=12):
    """Wake (label 0) blocks have high energy; sleep (label 1) is quiet."""
    labels = np.ones(T, dtype=int)
    labels[:100] = 0
    labels[250:] = 0
    specs = rng.random((T, F)) * 0.2
    specs[labels == 0] += 4.0
    return specs, labels


def test_etc_score_separates_clean_blocks():
    m = ETCModel(n_timestamps=70, kernel_width=0.5)
    specs, labels = _block_recording(np.random.default_rng(1))
    assert etc_score(m, specs, labels) > 0.99


def test_etc_score_padding_and_list_forms():
    m = ETCModel(n_timestamps=70, kernel_width=0.5)
    rng = np.random.default_rng(3)
    s1, l1 = _block_recording(rng)
    s2, l2 = _block_recording(rng)
    # padding (negative labels) is ignored
    l1 = l1.copy()
    l1[:10] = -1
    score_list = etc_score(m, [s1, s2], [l1, l2])
    # batched array of equal-length recordings is equivalent
    score_batch = etc_score(m, np.stack([s1, s2]), np.stack([l1, l2]))
    assert pytest.approx(score_list) == score_batch
    assert score_list > 0.99


def test_etc_score_macro_is_default_and_averages_per_recording():
    m = ETCModel(n_timestamps=70, kernel_width=0.5)
    rng = np.random.default_rng(11)
    # recordings of differing length / separability
    s1, l1 = _block_recording(rng, T=300)
    s2, l2 = _block_recording(rng, T=180)
    S, L = [s1, s2], [l1, l2]

    # default (macro) == mean of independently scored batches-of-1
    per = [etc_score(m, s, l) for s, l in zip(S, L)]
    assert pytest.approx(etc_score(m, S, L)) == float(np.mean(per))
    # a single recording scores the same either way
    assert pytest.approx(etc_score(m, s1, l1)) == per[0]


def test_etc_score_macro_skips_undefined_recordings():
    m = ETCModel(n_timestamps=70, kernel_width=0.5)
    rng = np.random.default_rng(12)
    s1, l1 = _block_recording(rng)
    # an all-sleep recording has undefined AUROC and must be skipped
    s2 = rng.random((120, 12))
    l2 = np.ones(120, dtype=int)
    assert pytest.approx(etc_score(m, [s1, s2], [l1, l2])) == etc_score(m, s1, l1)


def test_etc_score_micro_pools_timestamps():
    m = ETCModel(n_timestamps=70, kernel_width=0.5)
    rng = np.random.default_rng(13)
    s1, l1 = _block_recording(rng, T=300)
    s2, l2 = _block_recording(rng, T=180)
    # micro == concatenating all valid timestamps into one AUROC
    s_all = np.concatenate([1.0 - m.wake_proba(s1), 1.0 - m.wake_proba(s2)])
    y_all = np.concatenate([(l1 > 0).astype(int), (l2 > 0).astype(int)])
    assert pytest.approx(etc_score(m, [s1, s2], [l1, l2], reduce="micro")) == auroc_numpy(
        y_all, s_all
    )
