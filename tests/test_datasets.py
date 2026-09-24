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
    align_trim,
    load_data_sets_from_adapter,
    load_subject,
    mask_data,
    psg_map,
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


@pytest.mark.parametrize("use_id_pattern", [False, True])
def test_unpadded_ids_map_to_their_own_files(tmp_path, use_id_pattern):
    # Lexicographic id order diverges from filename order ("sen1" < "sen10"
    # but "sen10_accel.csv" < "sen1_accel.csv"). Each id must still map to the
    # file it was extracted from, not merely to the same index. Covered for
    # both the prefix-tree fallback and an explicit id_pattern.
    ds_root = tmp_path / "SENSE"
    accel_dir = ds_root / "cleaned_accelerometer"
    accel_dir.mkdir(parents=True)
    n_by_sid = {"sen1": 5, "sen2": 6, "sen10": 7}
    for sid, n in n_by_sid.items():
        pd.DataFrame({
            "time": np.arange(n, dtype=float),
            "x": np.full(n, float(n)),
            "y": np.zeros(n),
            "z": np.zeros(n),
        }).to_csv(accel_dir / f"{sid}_accel.csv", index=False)
    if use_id_pattern:
        (ds_root / "data_set.json").write_text(json.dumps({
            "name": "SENSE",
            "id_pattern": "<<ID>>_accel.csv",
        }))

    ds = DataSetObject.find_data_sets(tmp_path)["SENSE"]

    assert ds.ids == ["sen1", "sen10", "sen2"]
    for sid, n in n_by_sid.items():
        assert ds.get_filename("accelerometer", sid).name == f"{sid}_accel.csv"
        assert len(ds.get_feature_data("accelerometer", sid)) == n


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
    assert ds.config.timestamp.unit_by_feature is None
    assert ds.config.effective_psg_mapping["N1"] == 1


def test_load_subject_applies_per_feature_timestamp_units(tmp_path):
    ds_root = tmp_path / "SENSE"
    accel_dir = ds_root / "cleaned_accelerometer"
    psg_dir = ds_root / "cleaned_psg"
    accel_dir.mkdir(parents=True)
    psg_dir.mkdir(parents=True)

    pd.DataFrame({
        "t": [1_700_000_000_000.0, 1_700_000_000_010.0, 1_700_000_000_020.0],
        "x": [0.0, 0.1, 0.2],
        "y": [0.0, 0.0, 0.0],
        "z": [1.0, 1.0, 1.0],
    }).to_csv(accel_dir / "sen003_accel.csv", index=False)
    pd.DataFrame({
        "time": [1_700_000_000.0, 1_700_000_030.0],
        "stage": [0, 1],
    }).to_csv(psg_dir / "sen003_psg.csv", index=False)
    (ds_root / "data_set.json").write_text(json.dumps({
        "name": "SENSE",
        "timestamp": {
            "unit": "s",
            "unit_by_feature": {
                "accelerometer": "ms",
                "psg": "s",
            },
        },
        "id_pattern_by_feature": {
            "accelerometer": "<<ID>>_accel.csv",
            "psg": "<<ID>>_psg.csv",
        },
    }))

    ds = DataSetObject.find_data_sets(tmp_path)["SENSE"]
    sd = load_subject(ds, "sen003")

    assert sd.accel_df[TIMESTAMP_COL].iloc[0] == pytest.approx(1_700_000_000.0)
    assert sd.psg_df[TIMESTAMP_COL].iloc[0] == pytest.approx(1_700_000_000.0)


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


def test_align_trim_regrids_psg_gaps() -> None:
    accel = pd.DataFrame({
        TIMESTAMP_COL: np.arange(0.0, 121.0, 1.0),
        X_COL: np.zeros(121),
        Y_COL: np.zeros(121),
        Z_COL: np.ones(121),
    })
    psg = pd.DataFrame({
        TIMESTAMP_COL: [0.0, 30.0, 90.0, 120.0],
        PSG_COL: [0, 1, 3, 5],
    })

    _accel_aligned, psg_aligned = align_trim(accel, psg)

    assert list(psg_aligned[TIMESTAMP_COL]) == [0.0, 30.0, 60.0, 90.0, 120.0]
    assert list(psg_aligned[PSG_COL]) == [0, 1, -1, 3, 5]


def test_align_trim_handles_subepoch_psg_cadence() -> None:
    # PSG scored finer than psg_dt: multiple rows snap to the same epoch.
    # Must not raise "cannot reindex on an axis with duplicate labels".
    accel = pd.DataFrame({
        TIMESTAMP_COL: np.arange(0.0, 121.0, 1.0),
        X_COL: np.zeros(121),
        Y_COL: np.zeros(121),
        Z_COL: np.ones(121),
    })
    psg = pd.DataFrame({
        TIMESTAMP_COL: [0.0, 10.0, 30.0, 60.0, 90.0, 120.0],
        PSG_COL: [0, 9, 1, 2, 3, 5],
    })

    _accel_aligned, psg_aligned = align_trim(accel, psg)

    assert list(psg_aligned[TIMESTAMP_COL]) == [0.0, 30.0, 60.0, 90.0, 120.0]
    # 0.0 and 10.0 both snap to epoch 0; keep="first" wins.
    assert list(psg_aligned[PSG_COL]) == [0, 1, 2, 3, 5]


def test_align_trim_preserves_fractional_psg_phase() -> None:
    phase = 0.03125
    accel = pd.DataFrame({
        TIMESTAMP_COL: np.arange(phase, 120.0 + phase + 1.0, 1.0),
        X_COL: np.zeros(121),
        Y_COL: np.zeros(121),
        Z_COL: np.ones(121),
    })
    psg = pd.DataFrame({
        TIMESTAMP_COL: phase + np.arange(4) * 30.0,
        PSG_COL: [0, 1, 2, 5],
    })

    _accel_aligned, psg_aligned = align_trim(accel, psg)

    assert list(psg_aligned[TIMESTAMP_COL]) == pytest.approx(list(psg[TIMESTAMP_COL]))
    assert list(psg_aligned[PSG_COL]) == [0, 1, 2, 5]


def _coverage_frames(missing_epochs: list[int], n_epochs: int = 12):
    """PSG on a 30 s grid plus 50 Hz accel with whole epochs dropped out."""
    psg = pd.DataFrame({
        TIMESTAMP_COL: np.arange(n_epochs, dtype=float) * 30.0,
        PSG_COL: np.ones(n_epochs, dtype=int),
    })
    times = np.arange(0.0, n_epochs * 30.0, 0.02)
    keep = ~np.isin((times // 30.0).astype(int), missing_epochs)
    accel = pd.DataFrame({
        TIMESTAMP_COL: times[keep],
        X_COL: np.zeros(keep.sum()),
        Y_COL: np.zeros(keep.sum()),
        Z_COL: np.ones(keep.sum()),
    })
    return accel, psg


def test_mask_data_masks_only_uncovered_epochs() -> None:
    accel, psg = _coverage_frames([5])

    masked = mask_data(accel, psg)

    assert list(np.flatnonzero(masked[PSG_COL].to_numpy() == -2)) == [5]
    assert (masked[PSG_COL].to_numpy() != -1).all()


def test_mask_data_does_not_mutate_input() -> None:
    accel, psg = _coverage_frames([5])
    before = psg[PSG_COL].tolist()

    mask_data(accel, psg)

    assert psg[PSG_COL].tolist() == before


def test_psg_map_does_not_mutate_input() -> None:
    psg = pd.DataFrame({PSG_COL: [0, 1, 2, 3]})
    before = psg[PSG_COL].tolist()

    mapped = psg_map(psg, {0: 0, 1: 1, 2: 2, 3: 5})

    assert psg[PSG_COL].tolist() == before
    assert mapped[PSG_COL].tolist() == [0, 1, 2, 5]


def test_align_trim_no_overlap_warns_and_empties() -> None:
    accel = pd.DataFrame({
        TIMESTAMP_COL: [0.0, 1.0],
        X_COL: [0.0, 0.0], Y_COL: [0.0, 0.0], Z_COL: [1.0, 1.0],
    })
    psg = pd.DataFrame({TIMESTAMP_COL: [100.0, 130.0], PSG_COL: [0, 1]})

    with pytest.warns(RuntimeWarning, match="do not overlap"):
        accel_aligned, psg_aligned = align_trim(accel, psg)

    assert len(accel_aligned) == 0
    assert len(psg_aligned) == 0


def test_mask_data_does_not_dilate_by_default() -> None:
    # Two dropouts 3 epochs apart: under the old +/-1.5 min rule their
    # expansions merged and swallowed everything in between.
    accel, psg = _coverage_frames([4, 7])

    masked = mask_data(accel, psg)

    assert list(np.flatnonzero(masked[PSG_COL].to_numpy() == -2)) == [4, 7]


def test_mask_data_dilates_when_asked() -> None:
    accel, psg = _coverage_frames([5])

    masked = mask_data(accel, psg, extra_mask_minutes=1.0)

    assert list(np.flatnonzero(masked[PSG_COL].to_numpy() == -2)) == [3, 4, 5, 6, 7]


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


def test_custom_feature_loader_is_lazy_and_cached(tmp_path):
    ds = DataSetObject("custom", tmp_path)
    calls = []

    def load_accel(subject_id):
        calls.append(subject_id)
        return pd.DataFrame({
            "t": [2.0, 1.0],
            "x": [0.2, 0.1],
            "y": [0.0, 0.0],
            "z": [1.0, 1.0],
        })

    ds.set_feature_loader("accelerometer", load_accel, ids=["subject_001"])

    first = ds.get_feature_data("accelerometer", "subject_001")
    second = ds.get_feature_data("accelerometer", "subject_001")

    assert calls == ["subject_001"]
    assert first is second
    assert list(first.iloc[:, 0]) == [1.0, 2.0]


def test_load_data_sets_from_adapter_file(tmp_path):
    adapter = tmp_path / "pisces_lite_adapter.py"
    adapter.write_text(
        "import pandas as pd\n"
        "from pisces_lite.datasets import DataSetObject\n"
        "\n"
        "def load_data_set(root):\n"
        "    ds = DataSetObject('adapter_ds', root)\n"
        "    ds.set_feature_loader(\n"
        "        'psg',\n"
        "        lambda subject_id: pd.DataFrame({'time': [0.0], 'stage': [1]}),\n"
        "        ids=['s1'],\n"
        "    )\n"
        "    return ds\n"
    )

    loaded = load_data_sets_from_adapter(tmp_path)

    assert set(loaded) == {"adapter_ds"}
    ds = loaded["adapter_ds"]
    assert ds.ids == ["s1"]
    assert ds.get_feature_data("psg", "s1").iloc[0, 1] == 1
