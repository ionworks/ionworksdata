"""Automatic reader detection based on file content."""

from __future__ import annotations

from pathlib import Path

import fastexcel
import polars as pl

from ._utils import (
    suppress_excel_dtype_warnings,
    is_maccor_text_extension,
    read_excel_and_get_column_names,
)

# Maccor signature columns
MACCOR_COLUMNS = [
    "step",
    "test time",
    "test (sec)",
    "test time (sec)",
    "prog time",
    "current (a)",
    "current",
    "voltage (v)",
    "voltage",
    "cycle",
    "cyc#",
    "cycle id",
    "logtemp001",
    "temperature (°c)",
    "status",
    "state",
    "md",
]

# Neware signature columns
NEWARE_TIMESTAMP_COLS = ["DateTime", "Absolute Time", "Date(h:min:s.ms)"]
NEWARE_CURRENT_COLS = ["Current (mA)", "Cur(mA)", "Current(A)", "Current (A)"]
NEWARE_VOLTAGE_COLS = ["Voltage (V)", "Voltage(V)"]


def _is_biologic_text_extension(ext: str) -> bool:
    """Check if extension is .txt, .mpt, or .mpr"""
    return ext == ".txt" or ext == ".mpt" or ext == ".mpr"


def _has_biologic_time_col(text: str) -> bool:
    """Check if text contains Biologic time column names (case-insensitive)."""
    text_lower = text.lower()
    return any(t in text_lower for t in ["time/s"])


def _is_maccor_text_extension(ext: str) -> bool:
    """Check if extension is .txt or .+3-4 digits (e.g., .123, .0011)"""
    return ext == ".txt" or (len(ext) in (4, 5) and ext[1:].isdigit())


def _has_maccor_time_col(text: str) -> bool:
    """Check if text contains Maccor time column names (case-insensitive)."""
    text_lower = text.lower()
    return any(t in text_lower for t in ["test time", "test (sec)", "prog time"])


def _is_maccor_excel(filename: Path) -> bool:
    """
    Check if an Excel file is a Maccor file by examining column headers.

    Assumes the first row is always the header row.

    Parameters
    ----------
    filename : Path
        Path to the Excel file to check.

    Returns
    -------
    bool
        True if the file appears to be a Maccor file, False otherwise.
    """
    try:
        # Import here to avoid circular dependency
        df, column_names = read_excel_and_get_column_names(filename)
        # Check column headers for Maccor signature
        has_step = any("step" in col for col in column_names)
        has_time = any(_has_maccor_time_col(col) for col in column_names)
        if has_step and has_time:
            return True

        maccor_col_count = sum(
            1 for col in column_names if any(mc in col for mc in MACCOR_COLUMNS)
        )
        if maccor_col_count >= 3:
            return True

        return False
    except Exception:
        return False


def _read_first_lines(filename: Path, num_lines: int = 10) -> list[str]:
    """Read first lines from file, trying multiple encodings."""
    for encoding in ["utf-8", "latin1", "ISO-8859-1"]:
        try:
            with open(filename, encoding=encoding) as f:
                return [f.readline() for _ in range(num_lines)]
        except UnicodeDecodeError:
            continue
    return []


def detect_reader(filename: str | Path) -> str:
    """
    Automatically detect the reader type based on file content.

    Parameters
    ----------
    filename : str | Path
        Path to the file to detect the reader for.

    Returns
    -------
    str
        The detected reader name (e.g., "novonix", "maccor", "neware",
        "repower").

    Raises
    ------
    ValueError
        If the reader type cannot be determined from the file.
    """
    filename = Path(filename)
    ext = filename.suffix.lower()

    # Check for Excel files (Neware or Maccor)
    if ext in [".xls", ".xlsx"]:
        if _is_neware_excel(filename):
            return "neware"
        if _is_maccor_excel(filename):
            return "maccor"

    # Read first few lines to check file signatures
    first_lines = _read_first_lines(filename)
    first_line = first_lines[0] if first_lines else ""
    first_10_lines = "".join(first_lines)

    # Check for Novonix: starts with [Summary] and contains "Novonix"
    if "[Summary]" in first_line and "Novonix" in first_10_lines:
        return "novonix"

    # Check for Maccor: contains "Date of Test:" in first line
    if "Date of Test:" in first_line:
        for line in first_lines:
            if "Step" in line and _has_maccor_time_col(line):
                return "maccor"

    # CSV file checks
    if ext == ".csv":
        # Repower signature
        if all(s in first_line for s in ["Cycle ID", "Step ID", "Record ID"]):
            return "repower"
        # Neware CSV signature
        if any(ts in first_line for ts in NEWARE_TIMESTAMP_COLS):
            return "neware"
        # Maccor CSV signature
        if "Step" in first_line and _has_maccor_time_col(first_line):
            return "maccor"

    # Maccor .txt files or .+3digits: tab-separated with "Step" in header
    if is_maccor_text_extension(ext) and "\t" in first_line:
        for line in first_lines:
            if "Step" in line and _has_maccor_time_col(line):
                return "maccor"

    # Biologic .txt, .mpt, or .mpr files
    if _is_biologic_text_extension(ext):
        for line in first_lines:
            if _has_biologic_time_col(line):
                return "biologic" if ext == ".txt" else "biologic mpt"

    raise ValueError(
        f"Could not automatically detect reader type for file: {filename}. "
        f"Please specify the reader type explicitly."
    )


def _is_neware_excel(filename: Path) -> bool:
    """
    Check if an Excel file is a Neware file by examining column headers.

    Parameters
    ----------
    filename : Path
        Path to the Excel file to check.

    Returns
    -------
    bool
        True if the file appears to be a Neware file, False otherwise.
    """
    try:
        xl_reader = fastexcel.read_excel(filename)
        # Check each sheet for Neware column signatures
        for sheet_name in xl_reader.sheet_names:
            try:
                with suppress_excel_dtype_warnings():
                    df = pl.read_excel(filename, sheet_name=sheet_name)
                if df is None or df.height == 0:
                    continue

                columns = df.columns
                has_timestamp = any(col in columns for col in NEWARE_TIMESTAMP_COLS)
                has_current = any(col in columns for col in NEWARE_CURRENT_COLS)
                has_voltage = any(col in columns for col in NEWARE_VOLTAGE_COLS)

                # Neware file should have timestamp and current/voltage
                if has_timestamp and (has_current or has_voltage):
                    return True
            except Exception:
                continue

        return False
    except Exception:
        return False
