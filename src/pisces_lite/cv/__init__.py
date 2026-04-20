"""Cross-validation orchestration (framework-agnostic).

Two modes:

- ``loso`` — leave-one-subject-out across the union of ``test_sets``.
- ``transfer`` — train once on ``train_sets``, evaluate every subject in
  ``test_sets`` (which must be disjoint from ``train_sets``).

Both modes hold out ``validation_sets`` from training. See
:class:`CVConfig` for the exact invariants.
"""
from pisces_lite.cv.config import CVConfig, CVMode
from pisces_lite.cv.fold import Fold, SubjectRef, iter_folds

__all__ = ["CVConfig", "CVMode", "Fold", "SubjectRef", "iter_folds"]
