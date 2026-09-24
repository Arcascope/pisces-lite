"""MetricsConfig: parsed ``metrics.json``, plus spec construction."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Optional

from pisces_lite.metrics.specs import (
    MetricSpec,
    make_class_count_metric,
    make_cm_cell,
    make_oura_gap,
    make_stage_mape,
    metric_spec_from_name,
)


Format = Literal["long", "wide"]


DEFAULT_CLASS_NAMES: dict[int, List[str]] = {
    2: ["wake", "sleep"],
    3: ["wake", "nrem", "rem"],
    4: ["wake", "light", "deep", "rem"],
    5: ["wake", "n1", "n2", "n3", "rem"],
}


def default_class_names(num_classes: int) -> List[str]:
    if num_classes in DEFAULT_CLASS_NAMES:
        return list(DEFAULT_CLASS_NAMES[num_classes])
    return [f"class{i}" for i in range(num_classes)]


@dataclass
class MetricsConfig:
    """Parsed ``metrics.json``.

    Fields:
      - ``num_classes``, ``psg_dt_minutes``, ``min_stage_minutes``: scoring setup
      - ``metrics``: names of registered metric specs to evaluate
      - ``per_class_counts`` / ``per_class_mape`` / ``confusion_matrix`` /
        ``oura_gap``: per-class and derived metric flags
      - ``format``: ``"long"`` (default) or ``"wide"``
      - ``class_names``: optional, defaults from ``num_classes``
      - ``metric_column`` / ``value_column``: column names used in long format
      - ``wake_class``: index treated as wake for binary scoring
    """

    num_classes: int
    psg_dt_minutes: float = 0.5
    min_stage_minutes: float = 5.0
    metrics: List[str] = field(default_factory=list)
    per_class_counts: bool = False
    per_class_mape: bool = False
    confusion_matrix: bool = False
    oura_gap: bool = False
    format: Format = "long"
    class_names: Optional[List[str]] = None
    metric_column: str = "metric"
    value_column: str = "value"
    wake_class: int = 0

    @classmethod
    def from_json(cls, path: "Path | str") -> "MetricsConfig":
        with open(path) as f:
            return cls.from_dict(json.load(f))

    @classmethod
    def from_dict(cls, d: dict) -> "MetricsConfig":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})

    def resolved_class_names(self) -> List[str]:
        names = self.class_names or default_class_names(self.num_classes)
        return [c.lower() for c in names]

    def build_specs(self) -> List[MetricSpec]:
        class_names = self.resolved_class_names()
        specs: List[MetricSpec] = []

        for name in self.metrics:
            specs.append(metric_spec_from_name(name))

        if self.per_class_counts:
            for ci, cn in enumerate(class_names):
                specs.append(MetricSpec(
                    f"true_{cn}_count", make_class_count_metric(ci, true=True)
                ))
                specs.append(MetricSpec(
                    f"pred_{cn}_count", make_class_count_metric(ci, true=False)
                ))

        if self.per_class_mape:
            for ci in range(1, self.num_classes):
                specs.append(MetricSpec(
                    f"{class_names[ci]}_mape", make_stage_mape(ci)
                ))

        if self.confusion_matrix:
            for ti in range(self.num_classes):
                for pi in range(self.num_classes):
                    specs.append(MetricSpec(
                        f"cm_{class_names[ti]}_{class_names[pi]}",
                        make_cm_cell(ti, pi),
                    ))

        if self.oura_gap:
            specs.append(MetricSpec("oura_gap", make_oura_gap(class_names)))

        return specs
