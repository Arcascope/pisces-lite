"""Metrics CSV logger.

Central artifact of pisces-lite. ROC/AUROC analysis and plotting are downstream
consumers of what this module writes.

Config (``metrics.json``) picks the format — ``"long"`` (default) or ``"wide"``
— and the set of active metric specs. The logger is initialised once, then
:meth:`MetricsLogger.record` is called per evaluation; each call appends and
flushes immediately so crashes can't lose logged rows.

Long-format columns: ``run_id, fold, epoch, metric, value, subject_id,
dataset, training_dataset``.
"""
from pisces_lite.metrics.config import MetricsConfig, default_class_names
from pisces_lite.metrics.context import EvalContext
from pisces_lite.metrics.io import load_long, load_wide, to_long, to_wide
from pisces_lite.metrics.logger import IDENTITY_COLUMNS, MetricsLogger
from pisces_lite.metrics.specs import (
    MetricFn,
    MetricSpec,
    get_metric,
    metric_spec_from_name,
    register_metric,
)

__all__ = [
    "EvalContext",
    "IDENTITY_COLUMNS",
    "MetricFn",
    "MetricSpec",
    "MetricsConfig",
    "MetricsLogger",
    "default_class_names",
    "get_metric",
    "load_long",
    "load_wide",
    "metric_spec_from_name",
    "register_metric",
    "to_long",
    "to_wide",
]
