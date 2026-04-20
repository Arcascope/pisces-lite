"""Metric spec registry.

Importing this package triggers registration of all built-in specs.
"""
from pisces_lite.metrics.specs.base import (
    MetricFn,
    MetricSpec,
    get_metric,
    hard_preds,
    metric_spec_from_name,
    min_stage_minutes,
    psg_dt,
    register_metric,
    waso_minutes,
)
from pisces_lite.metrics.specs import auroc as _auroc  # noqa: F401 — registers
from pisces_lite.metrics.specs import builtin as _builtin  # noqa: F401 — registers
from pisces_lite.metrics.specs.builtin import (
    make_class_count_metric,
    make_cm_cell,
    make_oura_gap,
    make_stage_mape,
)

__all__ = [
    "MetricFn",
    "MetricSpec",
    "get_metric",
    "hard_preds",
    "make_class_count_metric",
    "make_cm_cell",
    "make_oura_gap",
    "make_stage_mape",
    "metric_spec_from_name",
    "min_stage_minutes",
    "psg_dt",
    "register_metric",
    "waso_minutes",
]
