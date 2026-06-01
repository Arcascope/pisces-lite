"""DataSetObject: discover and load multi-feature sleep-study datasets.

Ported from ``pisces2.data_sets.data_set_object`` with a few cruft removals
(broken polars-syntax ``save_*`` methods and the unused ``load_feature_data``
stub are gone). A new ``config`` attribute carries a :class:`DataSetConfig`
when ``data_set.json`` is present next to the data; callers read dataset-
specific knobs from ``data_set.config`` instead of sniffing ``data_set.name``.
"""
from __future__ import annotations

import logging
import os
import re
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Callable, DefaultDict, Dict, Iterable, List, Optional

import pandas as pd

from pisces_lite.datasets.constants import (
    PSG_COL,
    TIMESTAMP_COL,
    X_COL,
    Y_COL,
    Z_COL,
)
from pisces_lite.datasets.id_extraction import IdExtractor
from pisces_lite.datasets.loading import determine_header_rows_and_delimiter


_log = logging.getLogger(__name__)


FeatureLoader = Callable[[str], Optional[pd.DataFrame]]


class DataSetObject:
    """Discover/load multi-feature datasets laid out as ``cleaned_<feature>/``.

    Directory structure::

        dataset_name/
            cleaned_accelerometer/
                subject_001.csv
                subject_002.csv
            cleaned_psg/
                subject_001.csv
                subject_002.csv
            data_set.json            # optional; parsed into .config

    Attributes:
        name: Dataset name (directory name).
        path: Root directory.
        ids: Subject ids discovered across features.
        config: Parsed ``DataSetConfig`` if a ``data_set.json`` exists, else None.
    """

    FEATURE_PREFIX = "cleaned_"

    def __init__(
        self,
        name: str,
        path: Path,
        config: "Optional[object]" = None,
    ):
        self.name = name
        self.path = Path(path)
        self.ids: List[str] = []
        self.config = config

        self._feature_map: DefaultDict[str, Dict[str, str]] = defaultdict(dict)
        self._feature_cache: DefaultDict[str, Dict[str, pd.DataFrame]] = defaultdict(dict)
        self._feature_loaders: Dict[str, FeatureLoader] = {}

    def __str__(self) -> str:
        return f"{self.name}: {self.path}"

    @property
    def features(self) -> List[str]:
        """Union of file-map features and in-memory cache features."""
        return list(
            set(
                list(self._feature_cache.keys())
                + list(self._feature_map.keys())
                + list(self._feature_loaders.keys())
            )
        )

    @property
    def feature_prefix(self) -> str:
        if self.config is not None:
            return getattr(self.config, "feature_prefix", None) or self.FEATURE_PREFIX
        return self.FEATURE_PREFIX

    def get_feature_path(self, feature: str) -> Path:
        return self.path / (self.feature_prefix + feature)

    def drop_feature_data(self, feature: str, id: str) -> None:
        if feature not in self.features:
            warnings.warn(f"Feature {feature!r} not found in {self.name}.")
        self._feature_cache[feature].pop(id, None)

    def set_feature_data(self, feature: str, id: str, data: pd.DataFrame) -> None:
        self._feature_cache[feature][id] = data

    def set_feature_loader(
        self,
        feature: str,
        loader: FeatureLoader,
        ids: "Iterable[str] | None" = None,
    ) -> None:
        """Register a lazy feature loader for custom dataset layouts.

        The loader receives a subject id and returns the corresponding feature
        frame. This keeps non-standard parsing in dataset-local adapters while
        preserving the normal ``DataSetObject.get_feature_data`` surface.
        """
        self._feature_loaders[feature] = loader
        if ids is not None:
            self.ids = sorted(set(self.ids).union(str(id_) for id_ in ids))

    def get_feature_files(self, feature: str) -> Dict[str, str]:
        return dict(self._feature_map[feature])

    def get_id_files(self, id: str) -> Dict[str, str]:
        return {k: v[id] for k, v in self._feature_map.items() if id in v}

    def get_filename(self, feature: str, id: str) -> Optional[Path]:
        feature_ids = self._feature_map.get(feature)
        if feature_ids is None:
            _log.warning("Feature %r not found in %s", feature, self.name)
            return None
        file = feature_ids.get(id)
        if file is None:
            _log.warning("No %s file found for %s in %s", feature, id, self.name)
            return None
        return self.get_feature_path(feature) / file

    def get_feature_data(
        self,
        feature: str,
        id: str,
        keep_in_memory: bool = True,
    ) -> Optional[pd.DataFrame]:
        if feature not in self.features:
            warnings.warn(f"Feature {feature!r} not found in {self.name}. Returning None.")
            return None
        if id not in self.ids:
            warnings.warn(f"ID {id!r} not found in {self.name}")
            return None
        if (df := self._feature_cache[feature].get(id)) is not None:
            return df

        if (loader := self._feature_loaders.get(feature)) is not None:
            try:
                df = loader(id)
                if df is None:
                    return None
                if len(df) and len(df.columns):
                    df = df.sort_values(by=df.columns[0])
            except Exception as exc:
                warnings.warn(f"Error loading {feature} for {id} in {self.name}:\n{exc}")
                return None
            if keep_in_memory:
                self._feature_cache[feature][id] = df
            return df

        file = self.get_filename(feature, id)
        if file is None:
            return None
        _log.debug("Loading %s", file)

        sep_override = self._configured_delimiter(feature)
        try:
            n_rows, auto_delim = determine_header_rows_and_delimiter(file)
            sep = sep_override if sep_override is not None else auto_delim
            df = pd.read_csv(
                file,
                header=0 if n_rows and n_rows > 0 else None,
                skiprows=max((n_rows or 1) - 1, 0),
                sep=sep,
            )
            df = df.dropna()
        except Exception as exc:
            warnings.warn(f"Error reading {file}:\n{exc}")
            return None

        df = df.sort_values(by=df.columns[0])
        if keep_in_memory:
            self._feature_cache[feature][id] = df
        return df

    def _configured_delimiter(self, feature: str) -> Optional[str]:
        if self.config is None:
            return None
        csv = getattr(self.config, "csv", None)
        if csv is None:
            return None
        per_feature = getattr(csv, "delimiter_by_feature", None) or {}
        if feature in per_feature:
            return per_feature[feature]
        return getattr(csv, "delimiter", None)

    def add_feature_files(
        self,
        feature: str,
        files: Iterable[str],
        id_template: Optional[str],
        id_symbol: str,
    ) -> None:
        if feature not in self.features:
            _log.debug("Adding feature %s to %s", feature, self.name)
            self._feature_map[feature] = {}
        deduped_ids = set(self.ids)
        files_sorted = sorted(list(files))
        extracted_ids = sorted(
            IdExtractor().extract_ids(files_sorted, id_template, id_symbol)
        )
        for id_, file in zip(extracted_ids, files_sorted):
            self._feature_map[feature][id_] = file
            deduped_ids.add(id_)
        self.ids = sorted(deduped_ids)

    def parse_data(
        self,
        ignore_startswith: List[str] = (".",),
        ignore_endswith: List[str] = (".tmp",),
        id_templates: "Dict[str, str] | str | None" = None,
        id_symbol: str = "<<ID>>",
    ) -> None:
        """Scan each feature directory and populate ``ids`` + ``_feature_map``.

        If ``self.config`` provides an ``id_pattern`` (template containing
        ``id_symbol``), it is used when ``id_templates`` is not supplied.
        """
        cfg_template = None
        cfg_templates_by_feature = {}
        if self.config is not None:
            cfg_template = getattr(self.config, "id_pattern", None)
            cfg_templates_by_feature = getattr(self.config, "id_pattern_by_feature", None) or {}

        for feature in self.features:
            feature_path = self.get_feature_path(feature)
            if not feature_path.exists():
                warnings.warn(f"Feature path {feature_path} not found.")
                continue
            files = [f.name for f in feature_path.iterdir() if f.is_file()]
            relevant = [
                f
                for f in files
                if not any(f.startswith(p) for p in ignore_startswith)
                and not any(f.endswith(s) for s in ignore_endswith)
            ]
            if isinstance(id_templates, dict):
                id_template = id_templates.get(
                    feature, cfg_templates_by_feature.get(feature, cfg_template)
                )
            else:
                id_template = id_templates or cfg_templates_by_feature.get(feature, cfg_template)
            if relevant:
                self.add_feature_files(feature, relevant, id_template, id_symbol)

    @classmethod
    def find_data_sets(
        cls,
        root: "str | Path",
        try_parse: bool = True,
        load_configs: bool = True,
    ) -> Dict[str, "DataSetObject"]:
        """Walk ``root`` for ``cleaned_*`` feature directories and build DataSetObjects.

        When ``load_configs`` is True, each discovered dataset directory is
        checked for ``data_set.json`` and, if present, parsed into
        ``DataSetConfig`` (attached as ``.config``). Loading uses a late import
        to avoid a circular dependency.
        """
        root = str(root).replace("\\", "/")
        feature_dir_regex = rf".*/(.+)/{cls.FEATURE_PREFIX}(.+)/?"

        data_sets: Dict[str, DataSetObject] = {}
        for root_dir, _dirs, _files in os.walk(root, followlinks=True):
            normalized_root_dir = root_dir.replace("\\", "/")
            match = re.match(feature_dir_regex, normalized_root_dir)
            if not match:
                continue
            data_set_name = match.group(1)
            feature_name = match.group(2)
            if (ds := data_sets.get(data_set_name)) is None:
                ds = DataSetObject(data_set_name, Path(root_dir).parent)
                data_sets[ds.name] = ds
            ds._feature_map[feature_name] = {}

        if load_configs:
            from pisces_lite.datasets.config import DataSetConfig  # late import

            for ds in data_sets.values():
                json_path = ds.path / "data_set.json"
                if json_path.exists():
                    ds.config = DataSetConfig.from_json(json_path)

        if try_parse:
            for ds in data_sets.values():
                ds.parse_data()
        return data_sets


def get_subject_data(
    data: DataSetObject,
    subject_id: str,
) -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Return ``(accel_df, psg_df)`` for a subject, with standardised columns."""
    accel = data.get_feature_data("accelerometer", subject_id)
    if accel is not None:
        accel.columns = [TIMESTAMP_COL, X_COL, Y_COL, Z_COL]
    psg = data.get_feature_data("psg", subject_id)
    if psg is not None:
        psg.columns = [TIMESTAMP_COL, PSG_COL]
    return accel, psg
