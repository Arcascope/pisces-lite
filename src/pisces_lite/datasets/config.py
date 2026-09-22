"""Per-dataset configuration parsed from ``<dataset_dir>/data_set.json``.

"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import pandas as pd

from pisces_lite.datasets.constants import (
    PSG_MAPPING_PRESETS,
    TIMESTAMP_COL,
    X_COL,
    Y_COL,
    Z_COL,
)
from pisces_lite.datasets.processing import psg_map


TimestampUnit = Literal["s", "ms", "us"]


@dataclass
class AccelConfig:
    gravity_divisor: float = 1.0
    nominal_hz: Optional[float] = None


@dataclass
class PSGConfig:
    mapping: Optional[Dict] = None
    mapping_preset: Optional[str] = None
    dt_seconds: float = 30.0


@dataclass
class CSVConfig:
    delimiter: Optional[str] = None
    delimiter_by_feature: Optional[Dict[str, str]] = None


@dataclass
class TimestampConfig:
    unit: TimestampUnit = "s"
    unit_by_feature: Optional[Dict[str, TimestampUnit]] = None


@dataclass
class DataSetConfig:
    """Parsed ``data_set.json``.

    Example ``data_set.json``::

        {
          "name": "dreamt",
          "accel": { "gravity_divisor": 64.0, "nominal_hz": 64 },
          "psg":   { "mapping_preset": "dreamt" },
          "csv":   { "delimiter": "," },
          "timestamp": { "unit": "s" },
          "timestamp": {
            "unit": "s",
            "unit_by_feature": { "accelerometer": "ms", "psg": "s" }
          },
          "id_pattern": "<<ID>>.csv"
        }
    """

    name: str
    accel: AccelConfig = field(default_factory=AccelConfig)
    psg: PSGConfig = field(default_factory=PSGConfig)
    csv: CSVConfig = field(default_factory=CSVConfig)
    timestamp: TimestampConfig = field(default_factory=TimestampConfig)
    feature_prefix: Optional[str] = None
    id_pattern: Optional[str] = None
    id_pattern_by_feature: Optional[Dict[str, str]] = None
    id_symbol: str = "<<ID>>"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json(cls, path: "Path | str") -> "DataSetConfig":
        with open(path) as f:
            return cls.from_dict(json.load(f))

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DataSetConfig":
        def _section(key: str, subcls):
            section = d.get(key, {}) or {}
            known = {f for f in subcls.__dataclass_fields__}
            return subcls(**{k: v for k, v in section.items() if k in known})

        if "name" not in d:
            raise ValueError("data_set.json must contain a 'name' field")

        return cls(
            name=d["name"],
            accel=_section("accel", AccelConfig),
            psg=_section("psg", PSGConfig),
            csv=_section("csv", CSVConfig),
            timestamp=_section("timestamp", TimestampConfig),
            feature_prefix=d.get("feature_prefix"),
            id_pattern=d.get("id_pattern"),
            id_pattern_by_feature=d.get("id_pattern_by_feature"),
            id_symbol=d.get("id_symbol", "<<ID>>"),
            metadata=dict(d.get("metadata") or {}),
        )

    @property
    def effective_psg_mapping(self) -> Optional[Dict]:
        """Resolve PSG mapping: inline dict wins; else named preset; else None."""
        if self.psg.mapping is not None:
            return self.psg.mapping
        if self.psg.mapping_preset is not None:
            preset = PSG_MAPPING_PRESETS.get(self.psg.mapping_preset)
            if preset is None:
                raise KeyError(
                    f"Unknown PSG mapping preset {self.psg.mapping_preset!r}. "
                    f"Available: {sorted(PSG_MAPPING_PRESETS)}"
                )
            return preset
        return None


@dataclass
class SubjectData:
    """Accelerometer + PSG for one subject, with config transformations applied."""

    subject_id: str
    dataset_name: str
    accel_df: pd.DataFrame
    psg_df: pd.DataFrame


def _apply_timestamp_unit(df: pd.DataFrame, unit: TimestampUnit) -> pd.DataFrame:
    if unit == "s" or TIMESTAMP_COL not in df.columns:
        return df
    divisor = {"ms": 1_000.0, "us": 1_000_000.0}.get(unit)
    if divisor is None:
        return df
    df = df.copy()
    df[TIMESTAMP_COL] = df[TIMESTAMP_COL] / divisor
    return df


def _timestamp_unit_for_feature(cfg: DataSetConfig, feature: str) -> TimestampUnit:
    per_feature = cfg.timestamp.unit_by_feature or {}
    return per_feature.get(feature, cfg.timestamp.unit)


def load_subject(data_set, subject_id: str) -> SubjectData:
    """Load one subject, applying ``data_set.config`` transforms if present.

    Transforms applied when ``data_set.config`` is set:

    - Timestamps scaled from ``config.timestamp.unit`` into seconds.
    - Accelerometer columns divided by ``config.accel.gravity_divisor``.
    - PSG stages remapped via ``config.effective_psg_mapping`` (inline dict
      or named preset from ``PSG_MAPPING_PRESETS``).
    """
    from pisces_lite.datasets.data_set_object import get_subject_data

    accel_df, psg_df = get_subject_data(data_set, subject_id)
    if accel_df is None or psg_df is None:
        raise RuntimeError(
            f"missing accel or psg for {data_set.name}/{subject_id}"
        )

    cfg: Optional[DataSetConfig] = getattr(data_set, "config", None)
    if cfg is not None:
        accel_df = _apply_timestamp_unit(
            accel_df, _timestamp_unit_for_feature(cfg, "accelerometer")
        )
        psg_df = _apply_timestamp_unit(psg_df, _timestamp_unit_for_feature(cfg, "psg"))

        if cfg.accel.gravity_divisor != 1.0:
            accel_df = accel_df.copy()
            accel_df[[X_COL, Y_COL, Z_COL]] = (
                accel_df[[X_COL, Y_COL, Z_COL]] / cfg.accel.gravity_divisor
            )

        mapping = cfg.effective_psg_mapping
        if mapping is not None:
            psg_df = psg_df.copy()
            psg_df = psg_map(psg_df, mapping)

    return SubjectData(
        subject_id=subject_id,
        dataset_name=data_set.name,
        accel_df=accel_df,
        psg_df=psg_df,
    )
