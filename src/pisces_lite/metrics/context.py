"""EvalContext: the data bag passed to every MetricSpec during a record() call."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np


@dataclass
class EvalContext:
    """Everything a metric function might need, in one place.

    Attributes:
        y_true: Ground-truth labels, 1-D, already masked (no negatives).
        y_pred_proba: Class probabilities, shape ``(T, C)``, already masked.
        y_pred: Optional hard class predictions, 1-D, already masked.
        identity: Row-identifying columns (``run_id``, ``fold``, ``epoch``,
            ``subject_id``, ``dataset``, ``training_dataset``).
        row: Metric values computed so far (mutable; metrics can read others).
        extra: Caller-supplied bag. The logger pre-populates
            ``run_dir``, ``psg_dt_minutes``, ``min_stage_minutes``, and
            ``wake_class`` so specs don't have to plumb them in.
    """

    y_true: np.ndarray
    y_pred_proba: np.ndarray
    y_pred: np.ndarray | None
    identity: Dict[str, Any]
    row: Dict[str, Any]
    extra: Dict[str, Any] = field(default_factory=dict)
