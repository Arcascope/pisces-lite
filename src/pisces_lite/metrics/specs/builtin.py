"""Built-in metric specs. 

Extensible through registering new metrics via the @register_metric decoration."""
from __future__ import annotations

from typing import Callable, List

import numpy as np
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, f1_score

from pisces_lite.metrics.context import EvalContext
from pisces_lite.metrics.specs.base import (
    MetricFn,
    MetricSpec,
    hard_preds,
    min_stage_minutes,
    psg_dt,
    register_metric,
    sleep_mask,
    waso_minutes,
)


# ---- scalar metrics --------------------------------------------------------

@register_metric("balanced_accuracy")
def _balanced_accuracy(ctx: EvalContext) -> float:
    return float(balanced_accuracy_score(ctx.y_true, hard_preds(ctx)))


@register_metric("cohen_kappa")
def _cohen_kappa(ctx: EvalContext) -> float:
    return float(cohen_kappa_score(ctx.y_true, hard_preds(ctx)))


@register_metric("f1_sleep_wake")
def _f1_sleep_wake(ctx: EvalContext) -> float:
    y_true_bin = sleep_mask(ctx.y_true, ctx).astype(int)
    y_pred_bin = sleep_mask(hard_preds(ctx), ctx).astype(int)
    return float(f1_score(y_true_bin, y_pred_bin, zero_division=0))


@register_metric("true_tst_minutes")
def _true_tst_minutes(ctx: EvalContext) -> float:
    return float(np.sum(sleep_mask(ctx.y_true, ctx))) * psg_dt(ctx)


@register_metric("pred_tst_minutes")
def _pred_tst_minutes(ctx: EvalContext) -> float:
    return float(np.sum(sleep_mask(hard_preds(ctx), ctx))) * psg_dt(ctx)


@register_metric("true_waso_minutes")
def _true_waso_minutes(ctx: EvalContext) -> float:
    return waso_minutes(ctx.y_true, psg_dt(ctx), ctx)


@register_metric("pred_waso_minutes")
def _pred_waso_minutes(ctx: EvalContext) -> float:
    return waso_minutes(hard_preds(ctx), psg_dt(ctx), ctx)


@register_metric("tst_mape")
def _tst_mape(ctx: EvalContext) -> float:
    dt = psg_dt(ctx)
    true_sleep = float(np.sum(sleep_mask(ctx.y_true, ctx))) * dt
    pred_sleep = float(np.sum(sleep_mask(hard_preds(ctx), ctx))) * dt
    if true_sleep < min_stage_minutes(ctx):
        return float("nan")
    return abs(pred_sleep - true_sleep) / max(true_sleep, 1e-6) * 100


@register_metric("waso_mape")
def _waso_mape(ctx: EvalContext) -> float:
    dt = psg_dt(ctx)
    true_w = waso_minutes(ctx.y_true, dt, ctx)
    pred_w = waso_minutes(hard_preds(ctx), dt, ctx)
    if true_w < min_stage_minutes(ctx):
        return float("nan")
    return abs(pred_w - true_w) / max(true_w, 1e-6) * 100


# ---- per-class factories (used by config flags, not registered directly) ---

def make_class_count_metric(class_idx: int, true: bool) -> MetricFn:
    """Factory for ``true_<class>_count`` / ``pred_<class>_count`` specs."""
    if true:
        def fn(ctx: EvalContext) -> int:
            return int(np.sum(ctx.y_true == class_idx))
    else:
        def fn(ctx: EvalContext) -> int:
            return int(np.sum(hard_preds(ctx) == class_idx))
    return fn


def make_stage_mape(class_idx: int) -> MetricFn:
    def fn(ctx: EvalContext) -> float:
        dt = psg_dt(ctx)
        true_count = int(np.sum(ctx.y_true == class_idx))
        pred_count = int(np.sum(hard_preds(ctx) == class_idx))
        true_min = true_count * dt
        pred_min = pred_count * dt
        if true_min < min_stage_minutes(ctx):
            return float("nan")
        return abs(pred_min - true_min) / max(true_min, 1e-6) * 100
    return fn


def make_cm_cell(true_idx: int, pred_idx: int) -> MetricFn:
    def fn(ctx: EvalContext) -> int:
        y_pred = hard_preds(ctx)
        return int(np.sum((ctx.y_true == true_idx) & (y_pred == pred_idx)))
    return fn


# ---- oura_gap (worst sleep-stage MAPE shortfall vs Oura targets) ----------

_OURA_THRESHOLDS = {
    "tst": 0.13,
    "waso": 0.56,
    "light": 0.27,
    "deep": 0.38,
    "rem": 0.21,
}


def make_oura_gap(class_names: List[str]) -> MetricFn:
    """Worst shortfall vs Oura MAPE thresholds. Expects 4-class-style names."""
    name_to_threshold = {n.lower(): t for n, t in _OURA_THRESHOLDS.items()}

    def fn(ctx: EvalContext) -> float:
        dt = psg_dt(ctx)
        min_min = min_stage_minutes(ctx)

        def _mape(true_min: float, pred_min: float) -> float:
            if true_min < min_min:
                return float("nan")
            return abs(pred_min - true_min) / max(true_min, 1e-6)

        y_pred = hard_preds(ctx)
        true_sleep = float(np.sum(sleep_mask(ctx.y_true, ctx))) * dt
        pred_sleep = float(np.sum(sleep_mask(y_pred, ctx))) * dt
        tst = _mape(true_sleep, pred_sleep)

        true_w = waso_minutes(ctx.y_true, dt, ctx)
        pred_w = waso_minutes(y_pred, dt, ctx)
        waso = _mape(true_w, pred_w)

        gaps = [
            max(0, tst - _OURA_THRESHOLDS["tst"]),
            max(0, waso - _OURA_THRESHOLDS["waso"]),
        ]
        for ci in range(1, len(class_names)):
            cn = class_names[ci].lower()
            if cn in name_to_threshold:
                true_min_c = float(np.sum(ctx.y_true == ci)) * dt
                pred_min_c = float(np.sum(y_pred == ci)) * dt
                m = _mape(true_min_c, pred_min_c)
                gaps.append(max(0, m - name_to_threshold[cn]))

        valid = [g for g in gaps if not np.isnan(g)]
        if not valid:
            return float("nan")
        return float(max(valid))

    return fn
