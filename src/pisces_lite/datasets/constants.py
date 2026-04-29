"""Shared constants for dataset loading and processing.

Ported from ``pisces2.constants`` minus values that belong in a per-dataset
``data_set.json`` (gravity factors, dataset-specific mappings) or that were
unused by the dataset loading path.
"""
from __future__ import annotations

from typing import List

TIMESTAMP_COL = "TIMESTAMP"
X_COL = "ACC_X"
Y_COL = "ACC_Y"
Z_COL = "ACC_Z"
DELTA_T_COL = "DELTA_T"
PSG_COL = "PSG"

PSG_DT = 30  # seconds per PSG epoch (default)
ACC_HZ = 50  # expected accelerometer sampling rate (default)
END_HZ = 32  # resampled accelerometer sampling rate
MINIMUM_ACCEL_SAMPLES_PER_PSG = END_HZ

PSG_MASK = -1
SLEEP_CLASS_LABEL = 1
WAKE_CLASS_LABEL = 0

MINUTES_TO_SECONDS = 60


# ---- legacy PSG mapping presets (kept so existing configs can reference them)

PSG_MAPPING_DREAMT = {
    "Missing": PSG_MASK,
    "P": 0,
    "W": 0,
    "N1": 1,
    "N2": 2,
    "N3": 3,
    "R": 5,
}

PSG_MAPPING_WEAVER = {
    "L": PSG_MASK,
    "W": 0,
    "N1": 1,
    "N2": 2,
    "N3": 3,
    "N4": 4,
    "R": 5,
}

PSG_MAPPING_NO_N4 = {
    PSG_MASK: PSG_MASK,
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 5,
}

PSG_MAPPING_5C = {
    PSG_MASK: PSG_MASK,
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 3,
    5: 5
}

PSG_MAPPING_WLDR = {
    PSG_MASK: PSG_MASK,
    0: 0,
    1: 1,
    2: 1,
    3: 2,
    4: 2,
    5: 3,
}

PSG_MAPPING_WNR = {
    PSG_MASK: PSG_MASK,
    0: 0,
    1: 1,
    2: 1,
    3: 1,
    4: 1,
    5: 2,
}

PSG_MAPPING_WS = {
    PSG_MASK: PSG_MASK,
    0: 0,
    1: 1,
    2: 1,
    3: 1,
    4: 1,
    5: 1,
}

PSG_STAGING_LEGEND = {
    "Mask": PSG_MASK,
    "Wake": 0,
    "Light": 1,
    "Deep": 2,
    "REM": 3,
}

PSG_CLASS_TO_NAME = {
    PSG_MASK: "Mask",
    0: "Wake",
    1: "Light",
    2: "Deep",
    3: "REM",
}

WALCH_PSG_LABELS = {
    "Mask": -1,
    "Wake": 0,
    "N1": 1,
    "N2": 2,
    "N3": 3,
    "N4": 4,
    "REM": 5,
}

SET_NAMES_PRETTY = {
    "walch_et_al": "Healthy",
    "hf_cleaned": "Clinical",
    "dreamt": "DREAMT",
    "weaver": "Weaver",
    "hybrid_motion": "Hybrid",
    "chop": "CHOP",
    "buzz_walch_et_al": "Simulated Noise",
    "hf_cleaned+walch_et_al": "Healthy+Clinical",
    "walch_et_al+hf_cleaned": "Clinical+Healthy",
}


def get_class_names(n_classes: int) -> List[str]:
    if n_classes == 2:
        return ["Wake", "Sleep"]
    if n_classes == 3:
        return ["Wake", "Non-REM", "REM"]
    if n_classes == 4:
        return ["Wake", "Light", "Deep", "REM"]
    raise ValueError(f"Unsupported number of classes: {n_classes}")


PSG_MAPPING_PRESETS: dict[str, dict] = {
    "dreamt": PSG_MAPPING_DREAMT,
    "weaver": PSG_MAPPING_WEAVER,
    "no_n4": PSG_MAPPING_NO_N4,
    "wldr": PSG_MAPPING_WLDR,
    "wnr": PSG_MAPPING_WNR,
    "ws": PSG_MAPPING_WS,
}
