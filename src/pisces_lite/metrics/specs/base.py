"""MetricSpec, the metric registry, and shared helpers for built-in specs."""
from __future__ import annotations

from typing import Callable, Dict

import numpy as np

from pisces_lite.metrics.context import EvalContext


MetricFn = Callable[[EvalContext], float | int | str | Dict[str, float]]
"""A metric function: takes an EvalContext, returns a scalar or a
``{suffix: scalar}`` dict for per-class expansion."""


class MetricSpec:
    """A named metric function that produces one (or several) CSV rows.

    If ``func(ctx)`` returns a scalar, one row with ``metric=name`` is emitted.
    If it returns a ``dict[str, scalar]``, one row per key is emitted with
    ``metric = f"{name}.{key}"``.
    """

    __slots__ = ("name", "func")

    def __init__(self, name: str, func: MetricFn):
        self.name = name
        self.func = func

    def __repr__(self) -> str:
        return f"MetricSpec(name={self.name!r})"


_METRIC_REGISTRY: Dict[str, MetricFn] = {}


def register_metric(name: str):
    """Decorator to register a metric function by name."""

    def decorator(func: MetricFn) -> MetricFn:
        _METRIC_REGISTRY[name] = func
        return func

    return decorator


def get_metric(name: str) -> MetricFn:
    """Look up a registered metric by name."""
    if name not in _METRIC_REGISTRY:
        raise KeyError(
            f"Metric {name!r} is not registered. "
            f"Available: {sorted(_METRIC_REGISTRY)}"
        )
    return _METRIC_REGISTRY[name]


def metric_spec_from_name(name: str) -> MetricSpec:
    """Build a MetricSpec from a registered metric name."""
    return MetricSpec(name=name, func=get_metric(name))


# ---- helpers shared by built-in specs --------------------------------------

def hard_preds(ctx: EvalContext) -> np.ndarray:
    """Return hard class predictions: ``ctx.y_pred`` if given, else argmax."""
    if ctx.y_pred is not None:
        return ctx.y_pred
    return np.argmax(ctx.y_pred_proba, axis=-1)


def psg_dt(ctx: EvalContext) -> float:
    return float(ctx.extra.get("psg_dt_minutes", 0.5))


def min_stage_minutes(ctx: EvalContext) -> float:
    return float(ctx.extra.get("min_stage_minutes", 5.0))


def waso_minutes(labels: np.ndarray, dt: float) -> float:
    sleep_mask = labels > 0
    if np.sum(sleep_mask) == 0:
        return 0.0
    idxs = np.where(sleep_mask)[0]
    wake_in_span = int(np.sum(labels[idxs[0] : idxs[-1] + 1] == 0))
    return wake_in_span * dt
