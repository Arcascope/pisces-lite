"""ROC curve computation and aggregation for cross-validation experiments.

Ported from ``pisces2.roc_analysis`` minus the CLI tooling and the
``pisces2.utils`` dependency. Only numpy/scipy/sklearn are required.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List

import numpy as np
import scipy
import sklearn.metrics


@dataclass
class ROCResult:
    """Per-subject ROC curve data (FPR, TPR, thresholds, and AUROC)."""

    unique_id: str
    test_id: str
    test_data_set: str
    train_data_set: str
    legend_key: str
    fpr: np.ndarray
    tpr: np.ndarray
    thresholds: np.ndarray
    auroc: float

    @staticmethod
    def from_predictions_ovr(
        y_true,
        y_proba,
        class_idx: int,
        unique_id: str,
        test_id: str,
        test_data_set: str,
        train_data_set: str,
        legend_key: str,
    ) -> "ROCResult":
        """Compute one-vs-rest ROC curve for a single class."""
        y_true_flat = np.asarray(y_true).flatten()
        mask = y_true_flat >= 0
        y_true_filtered = y_true_flat[mask]

        n_classes = y_proba.shape[-1]
        y_score = y_proba.reshape(-1, n_classes)[mask, class_idx]

        y_true_ovr = (y_true_filtered == class_idx).astype(int)
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(
            y_true=y_true_ovr,
            y_score=y_score,
            pos_label=1,
        )
        auroc = float(scipy.integrate.trapezoid(y=tpr, x=fpr))

        return ROCResult(
            unique_id=unique_id,
            test_id=test_id,
            test_data_set=test_data_set,
            train_data_set=train_data_set,
            legend_key=legend_key,
            fpr=fpr,
            tpr=tpr,
            thresholds=thresholds,
            auroc=auroc,
        )

    @staticmethod
    def from_predictions(
        y_true,
        y_proba,
        unique_id: str,
        test_id: str,
        test_data_set: str,
        train_data_set: str,
        legend_key: str,
        wake_class: int = 0,
    ) -> "ROCResult":
        """Binary sleep-vs-wake ROC curve.

        All non-``wake_class`` labels are treated as sleep, and the sleep
        probability is ``1 - P(wake)``.
        """
        y_true_arr = np.asarray(y_true)
        y_score_select = y_true_arr >= wake_class
        y_score = y_true_arr[y_score_select]
        y_pred_score = np.asarray(y_proba)[y_score_select]

        sleep_proba = 1 - y_pred_score[..., wake_class]
        y_score_flat = y_score.flatten()
        sleep_proba_flat = sleep_proba.flatten()
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(
            y_true=(y_score_flat != wake_class).astype(int),
            y_score=sleep_proba_flat,
            pos_label=1,
        )

        return ROCResult(
            unique_id=unique_id,
            test_id=test_id,
            test_data_set=test_data_set,
            train_data_set=train_data_set,
            legend_key=legend_key,
            fpr=fpr,
            tpr=tpr,
            thresholds=thresholds,
            auroc=float(scipy.integrate.trapezoid(x=fpr, y=tpr)),
        )


class ROCResultCollection:
    """Collection of ``ROCResult`` objects for computing aggregate ROC statistics."""

    def __init__(self, roc_list: List[ROCResult]):
        self.roc_list = list(roc_list)

    def filter(self, filter_function: Callable[[ROCResult], bool]):
        self.roc_list = [r for r in self.roc_list if filter_function(r)]

    def where(self, filter_function: Callable[[ROCResult], bool]):
        return self.filter(filter_function)

    def add(self, result: ROCResult):
        self.roc_list.append(result)

    def to_npz(self, saveto: Path | str):
        saveto = Path(saveto)
        np.savez(saveto, roc_list=self.roc_list)

    @staticmethod
    def from_npz(npz_path: Path | str) -> "ROCResultCollection":
        dataz = np.load(npz_path, allow_pickle=True)
        if "roc_list" in dataz:
            return ROCResultCollection(list(dataz["roc_list"]))
        raise KeyError("Saved ROC data must have an 'roc_list' key.")

    def __len__(self):
        return len(self.roc_list)


def stages_roc_to_npz(
    stages_roc: dict,
    saveto: "Path | str",
) -> None:
    """Save a per-class ROC dict (``{class_idx: ROCResultCollection}``) to one .npz.

    Each class is stored under the key ``class_<idx>``.
    """
    saveto = Path(saveto)
    data = {
        f"class_{class_idx}": collection.roc_list
        for class_idx, collection in stages_roc.items()
    }
    np.savez(saveto, **data)


def stages_roc_from_npz(path: "Path | str") -> dict:
    """Inverse of ``stages_roc_to_npz``."""
    path = Path(path)
    dataz = np.load(path, allow_pickle=True)
    result = {}
    for key in dataz.files:
        if key.startswith("class_"):
            class_idx = int(key.split("_", 1)[1])
            result[class_idx] = ROCResultCollection(list(dataz[key]))
    return result


def load_roc(result_dirs: List[Path]) -> ROCResultCollection:
    """Load and concatenate ``roc_curves.npz`` from multiple run directories."""
    roc_list = []
    for result_dir in result_dirs:
        roc_path = Path(result_dir) / "roc_curves.npz"
        if not roc_path.exists():
            continue
        try:
            collection = ROCResultCollection.from_npz(roc_path)
            roc_list.extend(collection.roc_list)
        except Exception:
            continue
    return ROCResultCollection(roc_list)
