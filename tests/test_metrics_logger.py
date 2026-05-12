"""Smoke tests for MetricsLogger, MetricsConfig, and io helpers."""
from __future__ import annotations

import numpy as np
import pytest

from pisces_lite.metrics import (
    MetricsConfig,
    MetricsLogger,
    load_long,
    load_wide,
    to_long,
    to_wide,
)


def _fake_predictions(n: int = 200, num_classes: int = 4, seed: int = 0):
    rng = np.random.default_rng(seed)
    y_true = rng.integers(low=0, high=num_classes, size=n)
    logits = rng.normal(size=(n, num_classes))
    y_proba = np.exp(logits - logits.max(axis=-1, keepdims=True))
    y_proba = y_proba / y_proba.sum(axis=-1, keepdims=True)
    return y_true, y_proba


def _common_config(**overrides) -> MetricsConfig:
    base = dict(
        num_classes=4,
        metrics=["tst_mape", "waso_mape", "balanced_accuracy"],
        per_class_counts=True,
        per_class_mape=True,
        confusion_matrix=False,
        oura_gap=False,
        format="long",
    )
    base.update(overrides)
    return MetricsConfig(**base)


def test_long_format_record(tmp_path):
    cfg = _common_config(format="long")
    logger = MetricsLogger(cfg)
    csv_path = tmp_path / "cv_results.csv"

    y_true, y_proba = _fake_predictions()
    rows = logger.record(
        csv_path,
        y_true=y_true,
        y_pred_proba=y_proba,
        run_id="r1",
        fold=0,
        epoch=1,
        subject_id="S001",
        dataset="healthy",
        training_dataset="healthy",
    )

    assert csv_path.exists()
    assert len(rows) == len(logger.specs)
    df = load_long(csv_path)
    assert set(["run_id", "fold", "epoch", "metric", "value"]).issubset(df.columns)
    assert df["metric"].nunique() == len(logger.specs)


def test_wide_format_record(tmp_path):
    cfg = _common_config(format="wide")
    logger = MetricsLogger(cfg)
    csv_path = tmp_path / "cv_results.csv"

    y_true, y_proba = _fake_predictions()
    logger.record(
        csv_path,
        y_true=y_true, y_pred_proba=y_proba,
        run_id="r1", fold=0, epoch=1,
        subject_id="S001", dataset="healthy", training_dataset="healthy",
    )

    df = load_wide(csv_path)
    assert "tst_mape" in df.columns
    assert "balanced_accuracy" in df.columns
    assert "metric" not in df.columns
    assert len(df) == 1


def test_off_wrist_class_is_not_sleep_for_tst_and_waso(tmp_path):
    cfg = MetricsConfig(
        num_classes=4,
        class_names=["wake", "light", "deep", "rem", "off_wrist"],
        psg_dt_minutes=1.0,
        min_stage_minutes=0.0,
        metrics=[
            "true_tst_minutes",
            "pred_tst_minutes",
            "true_waso_minutes",
            "pred_waso_minutes",
            "tst_mape",
            "waso_mape",
        ],
        format="wide",
    )
    logger = MetricsLogger(cfg)
    csv_path = tmp_path / "cv_results.csv"

    y_true = np.array([4, 0, 1, 0, 1, 4])
    y_pred = np.array([4, 0, 1, 0, 1, 4])
    y_proba = np.eye(5, dtype=float)[y_pred]
    logger.record(
        csv_path,
        y_true=y_true,
        y_pred_proba=y_proba,
        y_pred=y_pred,
        run_id="r1",
        fold=0,
        epoch=1,
    )

    row = load_wide(csv_path).iloc[0]
    assert row["true_tst_minutes"] == pytest.approx(2.0)
    assert row["pred_tst_minutes"] == pytest.approx(2.0)
    assert row["true_waso_minutes"] == pytest.approx(1.0)
    assert row["pred_waso_minutes"] == pytest.approx(1.0)
    assert row["tst_mape"] == pytest.approx(0.0)
    assert row["waso_mape"] == pytest.approx(0.0)


def test_per_record_append_and_callback(tmp_path):
    cfg = _common_config(format="long")
    logger = MetricsLogger(cfg)
    csv_path = tmp_path / "cv_results.csv"

    callback_calls = []

    def on_write(rows, ctx):
        callback_calls.append((len(rows), ctx.identity["fold"]))

    y_true, y_proba = _fake_predictions()
    for fold in range(3):
        logger.record(
            csv_path,
            y_true=y_true, y_pred_proba=y_proba,
            run_id="r1", fold=fold, epoch=0,
            subject_id=f"S{fold:03d}", dataset="healthy", training_dataset="healthy",
            on_write=on_write,
        )

    assert len(callback_calls) == 3
    df = load_long(csv_path)
    assert df["fold"].nunique() == 3
    # each call appended len(specs) rows
    assert len(df) == 3 * len(logger.specs)


def test_long_wide_roundtrip(tmp_path):
    cfg = _common_config(format="long")
    logger = MetricsLogger(cfg)
    csv_path = tmp_path / "cv_results.csv"

    y_true, y_proba = _fake_predictions()
    logger.record(
        csv_path,
        y_true=y_true, y_pred_proba=y_proba,
        run_id="r1", fold=0, epoch=0,
        subject_id="S001", dataset="healthy", training_dataset="healthy",
    )

    long_df = load_long(csv_path)
    wide = to_wide(long_df)
    back = to_long(wide)

    # metric values should round-trip
    assert set(long_df["metric"]) == set(back["metric"])
    for m in long_df["metric"].unique():
        a = float(long_df.loc[long_df["metric"] == m, "value"].iloc[0])
        b = float(back.loc[back["metric"] == m, "value"].iloc[0])
        if np.isnan(a):
            assert np.isnan(b)
        else:
            assert a == pytest.approx(b)


def test_auroc_writes_npz(tmp_path):
    cfg = _common_config(metrics=["auroc"], per_class_counts=False, per_class_mape=False)
    logger = MetricsLogger(cfg)
    csv_path = tmp_path / "cv_results.csv"

    y_true, y_proba = _fake_predictions()
    logger.record(
        csv_path,
        y_true=y_true, y_pred_proba=y_proba,
        run_id="r1", fold=0, epoch=0,
        subject_id="S001", dataset="healthy", training_dataset="healthy",
    )
    logger.record(
        csv_path,
        y_true=y_true, y_pred_proba=y_proba,
        run_id="r1", fold=1, epoch=0,
        subject_id="S002", dataset="healthy", training_dataset="healthy",
    )

    npz_path = csv_path.parent / "roc_curves.npz"
    assert npz_path.exists()

    from pisces_lite.roc import ROCResultCollection
    collection = ROCResultCollection.from_npz(npz_path)
    assert len(collection) == 2
