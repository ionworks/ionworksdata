"""Utility functions for reading cycler data files."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import sys

import polars as pl


class _FilterStderr:
    """Filter stderr to suppress pandas dtype warnings while allowing other output."""

    def __init__(self):
        self.stderr = sys.stderr

    def write(self, text):
        # Filter out dtype warnings
        if (
            "Could not determine dtype for column" in text
            and "falling back to string" in text
        ):
            return
        # Pass through other output
        self.stderr.write(text)

    def flush(self):
        self.stderr.flush()


@contextmanager
def suppress_excel_dtype_warnings():
    """
    Context manager to suppress pandas dtype warnings when reading Excel files.

    Suppresses warnings of the form "Could not determine dtype for column X,
    falling back to string" while allowing other stderr output to pass through.
    """
    stderr_filter = _FilterStderr()
    original_stderr = sys.stderr
    try:
        sys.stderr = stderr_filter
        yield
    finally:
        sys.stderr = original_stderr


def is_maccor_text_extension(ext: str) -> bool:
    """Check if extension is .txt or .+3-4 digits (e.g., .123, .0011)"""
    return ext == ".txt" or (len(ext) in (4, 5) and ext[1:].isdigit())


def read_excel_and_get_column_names(
    filename: Path, header_row: int = 0, sheet_name: str | None = None
) -> tuple[pl.DataFrame, list[str]]:
    """
    Read Excel file and get column names.

    Parameters
    ----------
    filename : Path
        Path to Excel file
    header_row : int
        Row number to use as header (0-indexed)
    sheet_name : str | None
        Sheet name to read, None for first sheet

    Returns
    -------
    tuple[pl.DataFrame, list[str]]
        DataFrame and lowercase column names
    """
    with suppress_excel_dtype_warnings():
        df = pl.read_excel(
            filename, read_options={"header_row": header_row}, sheet_name=sheet_name
        )
    if "date of test:" in [str(col).lower() for col in df.columns]:
        df = pl.read_excel(
            filename, read_options={"header_row": header_row + 1}, sheet_name=sheet_name
        )
    return df, [] if df is None else [str(col).lower() for col in df.columns]
