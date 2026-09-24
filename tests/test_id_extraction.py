"""Tests for pisces_lite.datasets.id_extraction."""
from __future__ import annotations

import pytest

from pisces_lite.datasets.id_extraction import IdExtractor


def test_template_pairs_each_file_with_its_own_id():
    files = ["sen10_accel.csv", "sen1_accel.csv", "sen2_accel.csv"]

    pairs = IdExtractor().map_files_to_ids(files, "<<ID>>_accel.csv", "<<ID>>")

    assert pairs == [
        ("sen10", "sen10_accel.csv"),
        ("sen1", "sen1_accel.csv"),
        ("sen2", "sen2_accel.csv"),
    ]
    assert IdExtractor().extract_ids(files, "<<ID>>_accel.csv", "<<ID>>") == [
        "sen1",
        "sen10",
        "sen2",
    ]


def test_prefix_tree_pairs_files_without_relying_on_index_order():
    files = ["sen10_accel.csv", "sen1_accel.csv", "sen2_accel.csv"]

    pairs = IdExtractor().map_files_to_ids(files, None, "<<ID>>")

    assert dict(pairs) == {
        "sen10": "sen10_accel.csv",
        "sen1": "sen1_accel.csv",
        "sen2": "sen2_accel.csv",
    }


def test_single_file_without_template_raises():
    with pytest.raises(ValueError, match="ID template"):
        IdExtractor().map_files_to_ids(["only.csv"], None, "<<ID>>")


def test_empty_files_raises():
    with pytest.raises(ValueError, match="at least one file"):
        IdExtractor().map_files_to_ids([], None, "<<ID>>")


def test_template_strips_only_leading_prefix_and_trailing_suffix():
    # A global str.replace would also remove the "s" inside "csv".
    pairs = IdExtractor().map_files_to_ids(
        ["s1.csv", "ss1.csv"], "s<<ID>>.csv", "<<ID>>"
    )
    assert pairs == [("1", "s1.csv"), ("s1", "ss1.csv")]


def test_prefix_tree_collision_raises():
    # "c.csv" and "cac.csv" both reduce to leaf id "c"; silently overwriting
    # one would drop a subject, so this must be reported instead.
    with pytest.raises(ValueError, match="cannot be told apart"):
        IdExtractor().map_files_to_ids(["c.csv", "cac.csv", "cb.csv"], None, "<<ID>>")
