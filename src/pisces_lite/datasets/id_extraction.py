"""Prefix-tree-based subject-ID extraction from file names.

Used as a fallback when ``data_set.json`` does not specify an ``id_pattern`` —
the tree algorithm reverses file names, walks a prefix tree, and pulls the
varying-suffix portion as the subject id.
"""
from __future__ import annotations

import warnings
from copy import deepcopy
from typing import Dict, List


class SimplifiablePrefixTree:
    def __init__(self, delimiter: str = "", key: str = ""):
        self.key = key
        self.children: Dict[str, SimplifiablePrefixTree] = {}
        self.is_end_of_word = False
        self.delimiter = delimiter

    def _chars_from(self, word: str):
        return word.split(self.delimiter) if self.delimiter else word

    def insert(self, word: str):
        node = self
        for char in self._chars_from(word):
            if char not in node.children:
                node.children[char] = SimplifiablePrefixTree(self.delimiter, key=char)
            node = node.children[char]
        node.is_end_of_word = True

    def simplified(self) -> "SimplifiablePrefixTree":
        return deepcopy(self).simplify()

    def simplify(self):
        if len(self.children) == 1 and not self.is_end_of_word:
            child_key = next(iter(self.children))
            self.key += child_key
            self.children = self.children[child_key].children
            self.simplify()
        else:
            current_keys = list(self.children.keys())
            for key in current_keys:
                child = self.children.pop(key)
                child.simplify()
                self.children[child.key] = child
        return self

    def reversed(self) -> "SimplifiablePrefixTree":
        rev = SimplifiablePrefixTree(self.delimiter, key=self.key[::-1])
        rev.children = {k[::-1]: v.reversed() for k, v in self.children.items()}
        return rev

    def flattened(self, max_depth: int = 1) -> "SimplifiablePrefixTree":
        flat = SimplifiablePrefixTree(self.delimiter, key=self.key)
        if max_depth == 0:
            if not self.is_end_of_word:
                warnings.warn(f"max_depth is 0, but {self.key!r} is not a leaf.", stacklevel=2)
            return flat
        if max_depth == 1:
            for k, v in self.children.items():
                if v.is_end_of_word:
                    flat.children[k] = SimplifiablePrefixTree(self.delimiter, key=k)
                else:
                    for c in v._pushdown():
                        flat.children[c.key] = c
        else:
            for k, v in self.children.items():
                flat.children[k] = v.flattened(max_depth - 1)
        return flat

    def _pushdown(self) -> List["SimplifiablePrefixTree"]:
        pushed = [c for v in self.children.values() for c in v._pushdown()]
        for p in pushed:
            p.key = self.key + self.delimiter + p.key
        if not pushed:
            return [SimplifiablePrefixTree(self.delimiter, key=self.key)]
        return pushed


class IdExtractor(SimplifiablePrefixTree):
    """Extract subject ids from a list of file names.

    If ``id_template`` is given (e.g. ``"subject_<<ID>>.csv"``), ids are
    produced by simple string replace. Otherwise, the prefix-tree algorithm
    reverses file names, simplifies + flattens the tree, and uses the leaves
    (re-reversed) as the ids.

    ``map_files_to_ids`` returns ``(id, file)`` pairs so each file is paired
    with *its own* id. The ids are not necessarily in lexicographic filename
    order -- e.g. ``sen1`` sorts before ``sen10`` while ``sen10_accel.csv``
    sorts before ``sen1_accel.csv`` -- so pairing by index would silently
    assign subjects the wrong files.
    """

    def extract_ids(
        self,
        files: List[str],
        id_template: str | None,
        id_symbol: str,
    ) -> List[str]:
        """Return the sorted, de-duplicated ids found across ``files``."""
        return sorted(
            {id_ for id_, _ in self.map_files_to_ids(files, id_template, id_symbol)}
        )

    def map_files_to_ids(
        self,
        files: "List[str] | tuple[str, ...]",
        id_template: str | None,
        id_symbol: str,
    ) -> List[tuple[str, str]]:
        """Return ``(id, file)`` pairs, one per input file.

        With ``id_template`` each file is converted independently. Without it,
        every file is matched to the longest prefix-tree leaf that heads it, so
        ids whose lexicographic order differs from the filenames still line up.

        Raises ``ValueError`` if two distinct files resolve to the same id --
        that would silently drop one of them when the map is built, so it is
        reported instead. Supply an ``'id_pattern'`` to disambiguate.
        """
        files = list(files)
        if not files:
            raise ValueError("Please provide at least one file name to extract IDs")

        if id_template is not None:
            prefix, suffix = id_template.split(id_symbol)
            pairs = [
                (self._id_from_template(f, prefix, suffix), f) for f in files
            ]
            self._check_for_id_collisions(pairs)
            return pairs

        if len(files) == 1:
            raise ValueError(
                "Please provide an ID template if you only have one file name."
            )

        for file in files:
            self.insert(file[::-1])
        ids = sorted(c.key for c in self._prefix_flattened().children.values())

        pairs = []
        for file in files:
            matches = [i for i in ids if file.startswith(i)]
            if not matches:
                raise ValueError(
                    f"Could not extract a subject ID from {file!r} with the "
                    f"prefix-tree algorithm (candidate IDs: {ids}). Provide an "
                    f"'id_pattern' for this dataset."
                )
            longest = max(len(m) for m in matches)
            pairs.append((next(m for m in matches if len(m) == longest), file))
        self._check_for_id_collisions(pairs)
        return pairs

    @staticmethod
    def _id_from_template(filename: str, prefix: str, suffix: str) -> str:
        """Strip only a *leading* prefix and *trailing* suffix from ``filename``.

        Anchored rather than ``str.replace``: with template ``"s<<ID>>.csv"``,
        replacing every ``"s"`` would also eat the ``s`` in ``csv``.
        """
        if prefix and filename.startswith(prefix):
            filename = filename[len(prefix):]
        if suffix and filename.endswith(suffix):
            filename = filename[: -len(suffix)]
        return filename

    @staticmethod
    def _check_for_id_collisions(pairs: List[tuple[str, str]]) -> None:
        files_by_id: Dict[str, str] = {}
        for id_, file in pairs:
            previous = files_by_id.get(id_)
            if previous is not None and previous != file:
                raise ValueError(
                    f"Subject ID {id_!r} was extracted from two different files "
                    f"({previous!r} and {file!r}); the files cannot be told "
                    f"apart. Provide an 'id_pattern' that separates them."
                )
            files_by_id[id_] = file

    def _prefix_flattened(self) -> "IdExtractor":
        return self.simplified().flattened(1).reversed()
