"""MetricsLogger: config-driven CSV logging with per-record flush."""
from __future__ import annotations

import csv
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from pisces_lite.metrics.config import MetricsConfig
from pisces_lite.metrics.context import EvalContext
from pisces_lite.metrics.specs import MetricSpec


IDENTITY_COLUMNS: List[str] = [
    "run_id",
    "fold",
    "epoch",
    "subject_id",
    "dataset",
    "training_dataset",
]


OnWrite = Callable[[List[Dict[str, Any]], EvalContext], None]


class MetricsLogger:
    """Config-driven metrics logger.

    The logger is initialised once from a ``MetricsConfig`` (typically parsed
    from ``metrics.json``). Each call to :meth:`record` computes every
    configured metric against ``(y_true, y_pred_proba)``, appends the
    resulting row(s) to ``csv_path`` (in either long or wide format depending
    on the config), flushes immediately, and fires ``on_write`` once with the
    appended row list.
    """

    def __init__(
        self,
        config: MetricsConfig,
        specs: Optional[List[MetricSpec]] = None,
    ):
        self.config = config
        self.specs: List[MetricSpec] = (
            list(specs) if specs is not None else config.build_specs()
        )

    def record(
        self,
        csv_path: "Path | str",
        *,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        y_pred: Optional[np.ndarray] = None,
        run_id: str,
        fold: int,
        epoch: int,
        subject_id: Optional[str] = None,
        dataset: Optional[str] = None,
        training_dataset: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
        on_write: Optional[OnWrite] = None,
    ) -> List[Dict[str, Any]]:
        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        identity: Dict[str, Any] = {
            "run_id": run_id,
            "fold": fold,
            "epoch": epoch,
            "subject_id": subject_id,
            "dataset": dataset,
            "training_dataset": training_dataset,
        }
        if self._has_identity(csv_path, identity):
            return []

        full_extra: Dict[str, Any] = dict(extra or {})
        full_extra.setdefault("run_dir", csv_path.parent)
        full_extra.setdefault("psg_dt_minutes", self.config.psg_dt_minutes)
        full_extra.setdefault("min_stage_minutes", self.config.min_stage_minutes)
        full_extra.setdefault("wake_class", self.config.wake_class)

        y_true_arr = np.asarray(y_true)
        mask = y_true_arr >= 0
        y_true_m = y_true_arr[mask]
        y_pred_proba_m = np.asarray(y_pred_proba)[mask]
        y_pred_m = np.asarray(y_pred)[mask] if y_pred is not None else None

        metric_values: Dict[str, Any] = {}
        ctx = EvalContext(
            y_true=y_true_m,
            y_pred_proba=y_pred_proba_m,
            y_pred=y_pred_m,
            identity=identity,
            row=metric_values,
            extra=full_extra,
        )

        for spec in self.specs:
            try:
                result = spec.func(ctx)
            except Exception as exc:
                warnings.warn(
                    f"Metric {spec.name!r} failed: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                metric_values[spec.name] = float("nan")
                continue
            if isinstance(result, dict):
                for key, value in result.items():
                    metric_values[f"{spec.name}.{key}"] = value
            else:
                metric_values[spec.name] = result

        rows = self._rows_from_values(identity, metric_values)
        self._append(csv_path, rows)

        if on_write is not None:
            on_write(rows, ctx)
        return rows

    def _rows_from_values(
        self,
        identity: Dict[str, Any],
        metric_values: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        if self.config.format == "wide":
            return [{**identity, **metric_values}]
        mcol = self.config.metric_column
        vcol = self.config.value_column
        return [
            {**identity, mcol: name, vcol: value}
            for name, value in metric_values.items()
        ]

    def _append(self, csv_path: Path, rows: List[Dict[str, Any]]) -> None:
        if not rows:
            return
        fieldnames = list(rows[0].keys())
        write_header = not csv_path.exists() or csv_path.stat().st_size == 0
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def _has_identity(self, csv_path: Path, identity: Dict[str, Any]) -> bool:
        if not csv_path.exists() or csv_path.stat().st_size == 0:
            return False

        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                return False
            id_cols = [col for col in IDENTITY_COLUMNS if col in reader.fieldnames]
            if not id_cols:
                return False
            target = {col: self._csv_identity_value(identity.get(col)) for col in id_cols}
            for row in reader:
                if all(row.get(col, "") == target[col] for col in id_cols):
                    return True
        return False

    @staticmethod
    def _csv_identity_value(value: Any) -> str:
        return "" if value is None else str(value)
