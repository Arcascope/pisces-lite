"""CSV sniffing for dataset files with heterogeneous layout."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple


def determine_header_rows_and_delimiter(
    filename: "Path | str",
) -> Tuple[Optional[int], Optional[str]]:
    """Return ``(n_header_rows, delimiter)`` inferred by peeking at the file.

    Finds the first line starting with a digit (or negative number) and treats
    preceding lines as header. Tries space / comma / ", " delimiters against
    that line. Returns ``(None, None)`` on files with no numeric rows.
    """
    MAX_ROWS = 100
    header_row_count = 0
    with open(filename) as f:
        header_found = False
        line = ""
        while not header_found:
            line = f.readline()
            if line == "":
                return None, None
            line = line.strip()
            try:
                int(line[0]) if line[0] != "-" else int(line[:2])
                header_found = True
            except (ValueError, IndexError):
                header_row_count += 1
            if header_row_count >= MAX_ROWS:
                return None, None

        for guess in [" ", ",", ", "]:
            try:
                comps = line.split(guess)
                float(comps[0])
                return header_row_count, guess
            except ValueError:
                continue
        return header_row_count, None
