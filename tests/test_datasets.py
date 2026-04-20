"""Smoke tests for pisces_lite.datasets: find + parse + load + config transforms."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pisces_lite.datasets import (
    DataSetConfig,
    DataSetObject,
    PSG_COL,
    TIMESTAMP_COL,
    X_COL,
    Y_COL,
    Z_COL,
    get_subject_data,
    load_subject,
)


def _make_dataset_dir(
    root: Path,
    name: str,
    subjects: list[str],
    *,
    ts_unit: str = "s",
    psg_values: list = None,
    accel_factor: float = 1.0,
    data_set_json: dict | None = None,
    psg_delimiter: str = ",",
) -> Path:
    """Create a minimal 'cleaned_accelerometer' + 'cleaned_psg' layout."""
    ds_root = root / name
    accel_dir = ds_root / "cleaned_accelerometer"
    psg_dir = ds_root / "cleaned_psg"
    accel_dir.mkdir(parents=True)
    psg_dir.mkdir(parents=True)

    psg_values = psg_values if psg_values is not None else [0, 1, 2, 3]

    for s in subjects:
        n_accel = 1200
        ts_seconds_accel = np.linspace(0, 120, n_accel)
        if ts_unit == "ms":
            ts_accel = ts_seconds_accel * 1000.0
        elif ts_unit == "us":
            ts_accel = ts_seconds_accel * 1_000_000.0
        else:
            ts_accel = ts_seconds_accel
        pd.DataFrame({
            "time": ts_accel,
            "x": np.ones(n_accel) * accel_factor,
            "y": np.zeros(n_accel),
            "z": np.zeros(n_accel),
        }).to_csv(accel_dir / f"subject_{s}.csv", index=False)

        n_psg = 4
        ts_seconds_psg = np.arange(n_psg) * 30.0
        if ts_unit == "ms":
            ts_psg = ts_seconds_psg * 1000.0
        elif ts_unit == "us":
            ts_psg = ts_seconds_psg * 1_000_000.0
        else:
            ts_psg = ts_seconds_psg
        pd.DataFrame({
            "time": ts_psg,
            "stage": psg_values,
        }).to_csv(psg_dir / f"subject_{s}.csv", index=False, sep=psg_delimiter)

    if data_set_json is not None:
        (ds_root / "data_set.json").write_text(json.dumps(data_set_json))

    return ds_root


def test_find_parse_get_no_config_uses_prefix_tree(tmp_path):
    # no data_set.json ⇒ IdExtractor's prefix-tree fallback: ids retain the
    # shared prefix the algorithm couldn't strip ("subject_001" not "001").
    _make_dataset_dir(tmp_path, "walch_et_al", ["001", "002"])
    datasets = DataSetObject.find_data_sets(tmp_path)
    assert set(datasets) == {"walch_et_al"}
    ds = datasets["walch_et_al"]
    assert set(ds.features) == {"accelerometer", "psg"}
    assert ds.ids == ["subject_001", "subject_002"]
    assert ds.config is None

    df = ds.get_feature_data("accelerometer", "subject_001")
    assert df is not None and len(df) == 1200


def test_find_parse_get_with_id_pattern(tmp_path):
    # data_set.json's id_pattern drives exact id extraction.
    _make_dataset_dir(
        tmp_path, "walch_et_al", ["001", "002"],
        data_set_json={"name": "walch_et_al", "id_pattern": "subject_<<ID>>.csv"},
    )
    ds = DataSetObject.find_data_sets(tmp_path)["walch_et_al"]
    assert ds.ids == ["001", "002"]


def test_config_loads_from_json(tmp_path):
    cfg_dict = {
        "name": "dreamt",
        "accel": {"gravity_divisor": 64.0, "nominal_hz": 64},
        "psg": {"mapping_preset": "dreamt"},
        "csv": {"delimiter": ","},
        "timestamp": {"unit": "ms"},
        "id_pattern": "subject_<<ID>>.csv",
    }
    _make_dataset_dir(
        tmp_path, "dreamt",
        ["S001", "S002"],
        ts_unit="ms",
        psg_values=["W", "N1", "N2", "R"],
        accel_factor=64.0,
        data_set_json=cfg_dict,
    )
    datasets = DataSetObject.find_data_sets(tmp_path)
    ds = datasets["dreamt"]
    assert ds.config is not None
    assert ds.config.accel.gravity_divisor == 64.0
    assert ds.config.timestamp.unit == "ms"
    assert ds.config.effective_psg_mapping["N1"] == 1


def test_load_subject_applies_transforms(tmp_path):
    cfg_dict = {
        "name": "dreamt",
        "accel": {"gravity_divisor": 64.0},
        "psg": {"mapping_preset": "dreamt"},
        "timestamp": {"unit": "ms"},
        "id_pattern": "subject_<<ID>>.csv",
    }
    _make_dataset_dir(
        tmp_path, "dreamt",
        ["S001"],
        ts_unit="ms",
        psg_values=["W", "N1", "N2", "R"],
        accel_factor=64.0,  # raw values are 64; /64 → 1.0
        data_set_json=cfg_dict,
    )
    ds = DataSetObject.find_data_sets(tmp_path)["dreamt"]
    sd = load_subject(ds, "S001")

    # timestamps converted ms → s
    assert sd.accel_df[TIMESTAMP_COL].iloc[-1] == pytest.approx(120.0, rel=1e-6)
    # gravity divisor applied (raw was 64, result should be 1.0)
    assert sd.accel_df[X_COL].iloc[0] == pytest.approx(1.0, rel=1e-6)
    # PSG mapping applied: "W"→0, "N1"→1, "N2"→2, "R"→5
    assert list(sd.psg_df[PSG_COL]) == [0, 1, 2, 5]


def test_load_subject_no_config_noop(tmp_path):
    _make_dataset_dir(
        tmp_path, "walch_et_al", ["001", "002"],
        data_set_json={"name": "walch_et_al", "id_pattern": "subject_<<ID>>.csv"},
    )
    ds = DataSetObject.find_data_sets(tmp_path)["walch_et_al"]
    # Only id_pattern is set (no gravity/psg/ts transforms) ⇒ frames unchanged.
    sd = load_subject(ds, "001")
    assert sd.accel_df[X_COL].iloc[0] == 1.0
    assert list(sd.psg_df[PSG_COL]) == [0, 1, 2, 3]


def test_space_delimited_csv_via_config(tmp_path):
    # walch_et_al uses space-delimited CSVs historically
    cfg_dict = {
        "name": "walch_et_al",
        "csv": {"delimiter_by_feature": {"psg": " "}},
        "id_pattern": "subject_<<ID>>.csv",
    }
    _make_dataset_dir(
        tmp_path, "walch_et_al",
        ["001"],
        psg_delimiter=" ",
        data_set_json=cfg_dict,
    )
    ds = DataSetObject.find_data_sets(tmp_path)["walch_et_al"]
    psg = ds.get_feature_data("psg", "001")
    assert psg is not None
    assert psg.shape[1] == 2


def test_config_missing_name_raises(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"accel": {"gravity_divisor": 2.0}}))
    with pytest.raises(ValueError, match="must contain a 'name'"):
        DataSetConfig.from_json(path)
