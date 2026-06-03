"""Load dataset-local adapters without baking their formats into pisces-lite."""
from __future__ import annotations

import importlib.util
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from types import ModuleType
from typing import Protocol

from pisces_lite.datasets.data_set_object import DataSetObject


DEFAULT_ADAPTER_FILENAME = "pisces_lite_adapter.py"


class DataSetAdapter(Protocol):
    """Protocol for adapter objects exposed by custom dataset modules."""

    def load_data_sets(self, root: Path, **kwargs) -> object:
        ...


def import_adapter_module(path: "str | Path") -> ModuleType:
    """Import a dataset adapter module from a file or containing directory."""
    adapter_path = Path(path)
    if adapter_path.is_dir():
        adapter_path = adapter_path / DEFAULT_ADAPTER_FILENAME
    if not adapter_path.exists():
        raise FileNotFoundError(f"Dataset adapter not found: {adapter_path}")
    if adapter_path.suffix != ".py":
        raise ValueError(f"Dataset adapter must be a Python file: {adapter_path}")

    module_name = f"pisces_lite_external_adapter_{abs(hash(adapter_path.resolve()))}"
    spec = importlib.util.spec_from_file_location(module_name, adapter_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import dataset adapter: {adapter_path}")

    module = importlib.util.module_from_spec(spec)

    # Allow adapter modules to import helper files living alongside the adapter.
    adapter_dir = str(adapter_path.parent.resolve())
    added_to_path = False
    if adapter_dir not in sys.path:
        sys.path.insert(0, adapter_dir)
        added_to_path = True
    try:
        spec.loader.exec_module(module)
    finally:
        if added_to_path:
            try:
                sys.path.remove(adapter_dir)
            except ValueError:
                pass

    return module


def normalize_data_sets(data_sets: object) -> dict[str, DataSetObject]:
    """Normalize adapter output into ``{dataset_name: DataSetObject}``."""
    if isinstance(data_sets, DataSetObject):
        return {data_sets.name: data_sets}
    if isinstance(data_sets, Mapping):
        normalized = dict(data_sets)
    elif isinstance(data_sets, Iterable) and not isinstance(data_sets, (str, bytes)):
        normalized = {ds.name: ds for ds in data_sets}
    else:
        raise TypeError(
            "Dataset adapters must return a DataSetObject, an iterable of "
            "DataSetObject, or a mapping of name to DataSetObject."
        )

    for name, data_set in normalized.items():
        if not isinstance(data_set, DataSetObject):
            raise TypeError(
                f"Adapter returned {type(data_set).__name__} for {name!r}; "
                "expected DataSetObject."
            )
    return normalized


def load_data_sets_from_adapter(
    path: "str | Path",
    root: "str | Path | None" = None,
    **kwargs,
) -> dict[str, DataSetObject]:
    """Load custom datasets from a dataset-local ``pisces_lite_adapter.py``.

    The adapter may expose either ``load_data_sets(root, **kwargs)`` or
    ``load_data_set(root, **kwargs)``. ``path`` can be the adapter file itself
    or a directory containing the default adapter filename.

    Warning:
        Loading an adapter imports and executes arbitrary Python code from disk.
        Only load adapters you trust.
    """
    adapter_path = Path(path)
    module = import_adapter_module(adapter_path)
    if root is None:
        root = adapter_path if adapter_path.is_dir() else adapter_path.parent
    root = Path(root)

    load_data_sets = getattr(module, "load_data_sets", None)
    if callable(load_data_sets):
        return normalize_data_sets(load_data_sets(root, **kwargs))
    if load_data_sets is not None:
        raise TypeError(
            "Dataset adapter attribute 'load_data_sets' exists but is not callable."
        )

    load_data_set = getattr(module, "load_data_set", None)
    if callable(load_data_set):
        return normalize_data_sets(load_data_set(root, **kwargs))
    if load_data_set is not None:
        raise TypeError(
            "Dataset adapter attribute 'load_data_set' exists but is not callable."
        )

    adapter_cls = getattr(module, "Adapter", None)
    if adapter_cls is not None:
        if not callable(adapter_cls):
            raise TypeError(
                "Dataset adapter attribute 'Adapter' exists but is not instantiable."
            )
        adapter = adapter_cls()
        adapter_load_data_sets = getattr(adapter, "load_data_sets", None)
        if callable(adapter_load_data_sets):
            return normalize_data_sets(adapter_load_data_sets(root, **kwargs))
        if adapter_load_data_sets is not None:
            raise TypeError(
                "Dataset adapter Adapter.load_data_sets exists but is not callable."
            )

    raise AttributeError(
        "Dataset adapter must expose load_data_sets(root, **kwargs), "
        "load_data_set(root, **kwargs), or Adapter.load_data_sets(root, **kwargs)."
    )
