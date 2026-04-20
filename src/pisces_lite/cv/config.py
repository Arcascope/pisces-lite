"""CVConfig: cross-validation splitting configuration (no ML-framework deps)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Optional


CVMode = Literal["loso", "transfer"]


@dataclass
class CVConfig:
    """Subject-splitting configuration.

    Modes:
      - ``loso`` (leave-one-subject-out): requires ``test_sets ⊆ train_sets``.
        Iterates over every subject in the union of ``test_sets``, holding one
        out per fold; trains on the remaining pooled train subjects (minus any
        validation subjects). Produces N folds, N models.
      - ``transfer``: requires ``train_sets ∩ test_sets = ∅``. Trains one
        model on the pooled train subjects, evaluates every subject in the
        union of ``test_sets``. Produces 1 fold, 1 model, |test| evaluations.

    In both modes, ``train_sets ∩ validation_sets`` must be empty at the
    dataset level; validation subjects are always excluded from training.
    """

    mode: CVMode
    train_sets: List[str]
    test_sets: List[str]
    validation_sets: List[str] = field(default_factory=list)
    max_folds: Optional[int] = None

    @classmethod
    def from_json(cls, path: "Path | str") -> "CVConfig":
        with open(path) as f:
            return cls.from_dict(json.load(f))

    @classmethod
    def from_dict(cls, d: dict) -> "CVConfig":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})

    def validate(self) -> None:
        """Raise ValueError on any invariant violation."""
        train = set(self.train_sets)
        test = set(self.test_sets)
        val = set(self.validation_sets)

        if not train:
            raise ValueError("CVConfig: train_sets is empty")
        if not test:
            raise ValueError("CVConfig: test_sets is empty")

        overlap = train & val
        if overlap:
            raise ValueError(
                f"CVConfig: train_sets and validation_sets overlap on "
                f"{sorted(overlap)}; validation must be a held-out dataset"
            )

        if self.mode == "loso":
            extras = test - train
            if extras:
                raise ValueError(
                    f"CVConfig(mode='loso'): requires test_sets ⊆ train_sets; "
                    f"found in test_sets but not train_sets: {sorted(extras)}. "
                    f"If you intended to train on {sorted(train)} and test on "
                    f"disjoint datasets, use mode='transfer'."
                )
        elif self.mode == "transfer":
            overlap = train & test
            if overlap:
                raise ValueError(
                    f"CVConfig(mode='transfer'): requires train_sets ∩ "
                    f"test_sets = ∅; overlap: {sorted(overlap)}. "
                    f"If you intended leave-one-subject-out within that "
                    f"overlap, use mode='loso'."
                )
        else:
            raise ValueError(
                f"CVConfig: mode must be 'loso' or 'transfer', got {self.mode!r}"
            )
