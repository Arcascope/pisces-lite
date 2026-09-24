"""Tests for pisces_lite.roc: binary and one-vs-rest ROC computation."""
from __future__ import annotations

import numpy as np
import sklearn.metrics

from pisces_lite.roc import ROCResult, ROCResultCollection


def _result(y_true, y_proba, wake_class: int = 0) -> ROCResult:
    return ROCResult.from_predictions(
        y_true=y_true,
        y_proba=y_proba,
        unique_id="u",
        test_id="t",
        test_data_set="d",
        train_data_set="tr",
        legend_key="k",
        wake_class=wake_class,
    )


def test_from_predictions_drops_masked_epochs_only():
    y_true = np.array([-2, -1, 0, 1, 2, 3])
    y_proba = np.eye(4)[[0, 0, 0, 1, 2, 3]]

    result = _result(y_true, y_proba)

    # Negatives (masks) are dropped; all scored non-wake labels are sleep.
    assert result.auroc == 1.0


def test_from_predictions_nonzero_wake_class_keeps_lower_labels():
    # wake_class=1: label 0 is *not* the wake class, so it must survive as
    # sleep rather than being filtered out as if it were a mask.
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_proba = np.array([
        [0.1, 0.7, 0.1, 0.1],
        [0.2, 0.6, 0.1, 0.1],
        [0.6, 0.1, 0.2, 0.1],
        [0.7, 0.1, 0.1, 0.1],
        [0.1, 0.1, 0.7, 0.1],
        [0.1, 0.1, 0.6, 0.2],
    ])

    result = _result(y_true, y_proba, wake_class=1)

    expected = sklearn.metrics.roc_auc_score(
        (y_true != 1).astype(int), 1 - y_proba[:, 1]
    )
    assert result.auroc == expected


def test_from_predictions_ovr_matches_sklearn():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 3, size=200)
    logits = rng.normal(size=(200, 3))
    y_proba = np.exp(logits - logits.max(axis=-1, keepdims=True))
    y_proba /= y_proba.sum(axis=-1, keepdims=True)

    result = ROCResult.from_predictions_ovr(
        y_true=y_true,
        y_proba=y_proba,
        class_idx=2,
        unique_id="u",
        test_id="t",
        test_data_set="d",
        train_data_set="tr",
        legend_key="k",
    )

    expected = sklearn.metrics.roc_auc_score(
        (y_true == 2).astype(int), y_proba[:, 2]
    )
    assert result.auroc == expected


def test_collection_npz_roundtrip(tmp_path):
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]])
    collection = ROCResultCollection([_result(y_true, y_proba)])

    path = tmp_path / "roc_curves.npz"
    collection.to_npz(path)
    loaded = ROCResultCollection.from_npz(path)

    assert len(loaded) == 1
    assert loaded.roc_list[0].auroc == collection.roc_list[0].auroc
