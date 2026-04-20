"""Smoke tests for pisces_lite.cv: mode dispatch, invariants, fold shape."""
from __future__ import annotations

import pytest

from pisces_lite.cv import CVConfig, SubjectRef, iter_folds


class FakeDataSet:
    def __init__(self, name: str, ids: list[str]):
        self.name = name
        self.ids = list(ids)


def _datasets(**kwargs) -> list[FakeDataSet]:
    return [FakeDataSet(name, ids) for name, ids in kwargs.items()]


def test_transfer_yields_single_fold():
    cfg = CVConfig(mode="transfer", train_sets=["A", "B"], test_sets=["C"])
    datasets = _datasets(A=["a1", "a2"], B=["b1"], C=["c1", "c2", "c3"])

    folds = list(iter_folds(cfg, datasets))
    assert len(folds) == 1
    fold = folds[0]
    assert fold.mode == "transfer"
    assert fold.fold_id == 0
    assert fold.total_folds == 1
    assert len(fold.train_subjects) == 3   # 2 from A + 1 from B
    assert len(fold.test_subjects) == 3
    assert fold.val_subjects == []
    assert "transfer: 1 model on 3 train subj" in fold.description


def test_loso_yields_n_folds():
    cfg = CVConfig(mode="loso", train_sets=["A", "B"], test_sets=["A", "B"])
    datasets = _datasets(A=["a1", "a2"], B=["b1"])

    folds = list(iter_folds(cfg, datasets))
    assert len(folds) == 3
    for i, fold in enumerate(folds):
        assert fold.mode == "loso"
        assert fold.fold_id == i
        assert fold.total_folds == 3
        assert len(fold.test_subjects) == 1
        assert fold.test_subjects[0] not in fold.train_subjects
        assert len(fold.train_subjects) == 2
        assert fold.description.startswith("LOSO fold")


def test_loso_test_subset_of_train():
    # train=[A,B], test=[B] → iterate only B's subjects
    cfg = CVConfig(mode="loso", train_sets=["A", "B"], test_sets=["B"])
    datasets = _datasets(A=["a1", "a2"], B=["b1", "b2"])

    folds = list(iter_folds(cfg, datasets))
    assert len(folds) == 2
    assert {f.test_subjects[0].dataset_name for f in folds} == {"B"}
    # train pool spans A+B minus the held-out
    assert all(len(f.train_subjects) == 3 for f in folds)


def test_validation_held_out_from_training():
    cfg = CVConfig(
        mode="transfer",
        train_sets=["A"],
        test_sets=["B"],
        validation_sets=["C"],
    )
    datasets = _datasets(A=["a1", "a2"], B=["b1"], C=["c1", "c2"])
    fold = next(iter_folds(cfg, datasets))
    assert len(fold.val_subjects) == 2
    val_set = set(fold.val_subjects)
    assert val_set.isdisjoint(set(fold.train_subjects))


def test_max_folds_caps_loso():
    cfg = CVConfig(mode="loso", train_sets=["A"], test_sets=["A"], max_folds=2)
    datasets = _datasets(A=["a1", "a2", "a3", "a4"])
    folds = list(iter_folds(cfg, datasets))
    assert len(folds) == 2
    # total_folds reflects the uncapped count so callers see what was skipped
    assert folds[0].total_folds == 4


def test_invariants():
    # LOSO but test not a subset of train → error
    cfg = CVConfig(mode="loso", train_sets=["A"], test_sets=["B"])
    with pytest.raises(ValueError, match="test_sets ⊆ train_sets"):
        list(iter_folds(cfg, _datasets(A=["a1"], B=["b1"])))

    # transfer with overlap → error
    cfg = CVConfig(mode="transfer", train_sets=["A", "B"], test_sets=["B"])
    with pytest.raises(ValueError, match="train_sets ∩ test_sets"):
        list(iter_folds(cfg, _datasets(A=["a1"], B=["b1"])))

    # train ∩ val → error
    cfg = CVConfig(
        mode="transfer", train_sets=["A"], test_sets=["B"], validation_sets=["A"]
    )
    with pytest.raises(ValueError, match="train_sets and validation_sets overlap"):
        list(iter_folds(cfg, _datasets(A=["a1"], B=["b1"])))

    # missing dataset → error
    cfg = CVConfig(mode="transfer", train_sets=["A"], test_sets=["Z"])
    with pytest.raises(KeyError, match="Z"):
        list(iter_folds(cfg, _datasets(A=["a1"])))


def test_subjects_are_sorted_and_deterministic():
    cfg = CVConfig(mode="transfer", train_sets=["A"], test_sets=["B"])
    datasets = _datasets(A=["a3", "a1", "a2"], B=["b2", "b1"])
    fold = next(iter_folds(cfg, datasets))
    assert fold.train_subjects == [
        SubjectRef("A", "a1"),
        SubjectRef("A", "a2"),
        SubjectRef("A", "a3"),
    ]
    assert fold.test_subjects == [SubjectRef("B", "b1"), SubjectRef("B", "b2")]
