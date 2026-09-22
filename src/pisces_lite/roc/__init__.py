"""ROC and AUROC analysis.

Exposes ``ROCResult`` and ``ROCResultCollection``. The ``auroc`` metric spec
(``pisces_lite.metrics.specs.auroc``) uses these to write
``roc_curves.npz`` next to the metrics CSV and return the scalar AUROC to
the logger.
"""
from pisces_lite.roc.roc_analysis import (
    ROCResult,
    ROCResultCollection,
    load_roc,
    stages_roc_from_npz,
    stages_roc_to_npz,
)

__all__ = [
    "ROCResult",
    "ROCResultCollection",
    "load_roc",
    "stages_roc_from_npz",
    "stages_roc_to_npz",
]
