"""AUROC metric spec.

Computes a sleep-vs-wake ROC curve via ``pisces_lite.roc.ROCResult``, appends
it to the run's ``roc_curves.npz``, and returns the scalar AUROC to the logger.

Expects ``ctx.extra["run_dir"]`` (set by the logger from ``csv_path.parent``)
and optional ``ctx.extra["wake_class"]`` (defaults to 0).
"""
from __future__ import annotations

from pathlib import Path

from pisces_lite.metrics.context import EvalContext
from pisces_lite.metrics.specs.base import register_metric
from pisces_lite.roc import ROCResult, ROCResultCollection


@register_metric("auroc")
def _auroc(ctx: EvalContext) -> float:
    run_dir = ctx.extra.get("run_dir")
    if run_dir is None:
        raise RuntimeError(
            "auroc spec requires ctx.extra['run_dir']; the metrics logger "
            "sets this automatically from csv_path.parent."
        )
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    wake_class = int(ctx.extra.get("wake_class", 0))

    roc = ROCResult.from_predictions(
        y_true=ctx.y_true,
        y_proba=ctx.y_pred_proba,
        unique_id=str(ctx.identity.get("subject_id", "") or ""),
        test_id=str(ctx.identity.get("subject_id", "") or ""),
        test_data_set=str(ctx.identity.get("dataset", "") or ""),
        train_data_set=str(ctx.identity.get("training_dataset", "") or ""),
        legend_key=str(ctx.identity.get("run_id", "") or ""),
        wake_class=wake_class,
    )

    npz_path = run_dir / "roc_curves.npz"
    collection = (
        ROCResultCollection.from_npz(npz_path)
        if npz_path.exists()
        else ROCResultCollection([])
    )
    collection.add(roc)

    tmp_path = npz_path.with_name(npz_path.stem + ".tmp.npz")
    collection.to_npz(tmp_path)
    tmp_path.replace(npz_path)

    return float(roc.auroc)
