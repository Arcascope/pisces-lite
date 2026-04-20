"""Fold iteration over datasets according to a CVConfig."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, NamedTuple, Sequence, Union

from pisces_lite.cv.config import CVConfig


class SubjectRef(NamedTuple):
    """A reference to one subject. Cheap; consumers pass through ``load_subject``."""

    dataset_name: str
    subject_id: str


@dataclass(frozen=True)
class Fold:
    """One fold of a CV run.

    For ``mode='transfer'``, always ``fold_id=0`` and ``total_folds=1``.
    For ``mode='loso'``, ``fold_id`` ranges ``0..total_folds-1`` and each fold
    holds out exactly one ``test_subjects`` entry.
    """

    fold_id: int
    total_folds: int
    mode: str
    description: str
    train_subjects: List[SubjectRef]
    val_subjects: List[SubjectRef]
    test_subjects: List[SubjectRef]


DatasetsArg = Union[Dict[str, "object"], Sequence["object"]]


def _datasets_by_name(datasets: DatasetsArg) -> Dict[str, "object"]:
    if isinstance(datasets, dict):
        return datasets
    return {d.name: d for d in datasets}


def _subjects_from(
    set_names: Iterable[str],
    by_name: Dict[str, "object"],
    exclude: Iterable[SubjectRef] = (),
) -> List[SubjectRef]:
    excluded = set(exclude)
    out: List[SubjectRef] = []
    for name in set_names:
        ds = by_name[name]
        for sid in ds.ids:
            ref = SubjectRef(name, sid)
            if ref in excluded:
                continue
            out.append(ref)
    return sorted(out)


def _require_all_loaded(cv_config: CVConfig, by_name: Dict[str, "object"]) -> None:
    needed = set(cv_config.train_sets) | set(cv_config.test_sets) | set(cv_config.validation_sets)
    missing = needed - set(by_name)
    if missing:
        raise KeyError(
            f"iter_folds: datasets referenced by CVConfig are not loaded: "
            f"{sorted(missing)}. Loaded: {sorted(by_name)}"
        )


def iter_folds(cv_config: CVConfig, datasets: DatasetsArg) -> Iterator[Fold]:
    """Yield one :class:`Fold` per CV split, dispatching on ``cv_config.mode``.

    ``datasets`` may be a list of ``DataSetObject`` or a pre-built ``{name: ds}``
    mapping. The caller is responsible for calling ``parse_data`` on each
    dataset before passing it here.
    """
    cv_config.validate()
    by_name = _datasets_by_name(datasets)
    _require_all_loaded(cv_config, by_name)

    val_subjects = _subjects_from(cv_config.validation_sets, by_name)

    if cv_config.mode == "transfer":
        train_subjects = _subjects_from(
            cv_config.train_sets, by_name, exclude=val_subjects
        )
        test_subjects = _subjects_from(cv_config.test_sets, by_name)
        yield Fold(
            fold_id=0,
            total_folds=1,
            mode="transfer",
            description=(
                f"transfer: 1 model on {len(train_subjects)} train subj "
                f"from {list(cv_config.train_sets)} → "
                f"{len(test_subjects)} test subj "
                f"from {list(cv_config.test_sets)} "
                f"(val: {len(val_subjects)})"
            ),
            train_subjects=train_subjects,
            val_subjects=val_subjects,
            test_subjects=test_subjects,
        )
        return

    # LOSO
    test_pool = _subjects_from(
        cv_config.test_sets, by_name, exclude=val_subjects
    )
    train_pool = _subjects_from(
        cv_config.train_sets, by_name, exclude=val_subjects
    )

    total = len(test_pool)
    if cv_config.max_folds is not None:
        test_pool = test_pool[: cv_config.max_folds]

    for i, held_out in enumerate(test_pool):
        train_subjects = [s for s in train_pool if s != held_out]
        yield Fold(
            fold_id=i,
            total_folds=total,
            mode="loso",
            description=(
                f"LOSO fold {i + 1}/{total}: holding out "
                f"{held_out.subject_id} from {held_out.dataset_name} "
                f"({len(train_subjects)} train subj, "
                f"{len(val_subjects)} val subj)"
            ),
            train_subjects=train_subjects,
            val_subjects=val_subjects,
            test_subjects=[held_out],
        )
