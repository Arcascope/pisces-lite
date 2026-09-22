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
                warnings.warn(f"max_depth is 0, but {self.key!r} is not a leaf.")
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
    reverses file names, simplifies + flattens the tree, and returns the
    leaves (re-reversed) as the ids.
    """

    def extract_ids(
        self,
        files: List[str],
        id_template: str | None,
        id_symbol: str,
    ) -> List[str]:
        if not files:
            raise ValueError("Please provide at least one file name to extract IDs")

        if len(files) == 1:
            if not id_template:
                raise ValueError(
                    "Please provide an ID template if you only have one file name."
                )
            prefix, suffix = id_template.split(id_symbol)
            return [files[0].replace(prefix, "").replace(suffix, "")]

        if id_template is None:
            for file in files:
                self.insert(file[::-1])
            return sorted(c.key for c in self._prefix_flattened().children.values())

        prefix, suffix = id_template.split(id_symbol)
        return sorted(f.replace(prefix, "").replace(suffix, "") for f in files)

    def _prefix_flattened(self) -> "IdExtractor":
        return self.simplified().flattened(1).reversed()
