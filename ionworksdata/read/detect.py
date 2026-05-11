"""Automatic reader detection based on file content."""

from __future__ import annotations

import gzip
from pathlib import Path

import fastexcel
import polars as pl

from ._utils import (
    is_maccor_text_extension,
    read_excel_and_get_column_names,
    suppress_excel_dtype_warnings,
)

# Maccor signature columns
MACCOR_COLUMNS = [
    "step",
    "test time",
    "test (sec)",
    "test time (sec)",
    "testtime",
    "prog time",
    "current (a)",
    "current",
    "amps",
    "voltage (v)",
    "voltage",
    "volts",
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

# Arbin signature columns. "Cycle Index" + "Step Index" + a Test Time column
# distinguishes Arbin from Maccor (which uses "Cycle"/"Step" without "Index").
ARBIN_REQUIRED = ("Cycle Index", "Step Index")
ARBIN_TIME_PREFIX = "Test Time"


def _has_arbin_signature(header: str) -> bool:
    return all(s in header for s in ARBIN_REQUIRED) and ARBIN_TIME_PREFIX in header


# BDF (Battery Data Format) signature columns
BDF_REQUIRED_MACHINE = ("test_time_second", "voltage_volt", "current_ampere")
BDF_REQUIRED_LABELS = ("Test Time", "Voltage", "Current")
BDF_EXTENSIONS = (".bdf", ".bdf.gz", ".bdf.parquet")


def _has_bdf_extension(filename: Path) -> bool:
    name = filename.name.lower()
    return any(name.endswith(ext) for ext in BDF_EXTENSIONS)


def _has_bdf_machine_header(first_line: str) -> bool:
    cols = {c.strip() for c in first_line.split(",")}
    return all(name in cols for name in BDF_REQUIRED_MACHINE)


def _has_bdf_label_header(first_line: str) -> bool:
    cols = {c.strip() for c in first_line.split(",")}
    return all(label in cols for label in BDF_REQUIRED_LABELS)


def _first_line_gzipped(filename: Path) -> str:
    try:
        with gzip.open(filename, "rt", errors="replace") as handle:
            return handle.readline()
    except OSError:
        return ""


def _is_biologic_text_extension(ext: str) -> bool:
    """Check if extension is .txt, .mpt, or .mpr"""
    return ext == ".txt" or ext == ".mpt" or ext == ".mpr"


def _has_biologic_time_col(text: str) -> bool:
    """Check if text contains Biologic time column names (case-insensitive)."""
    text_lower = text.lower()
    return any(t in text_lower for t in ["time/s"])


def _has_maccor_time_col(text: str) -> bool:
    """Check if text contains Maccor time column names (case-insensitive)."""
    text_lower = text.lower()
    return any(
        t in text_lower for t in ["test time", "test (sec)", "prog time", "testtime"]
    )


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

    # Arbin .res files are native binary Access/MDB databases.
    if ext == ".res":
        return "arbin"

    # Check for Gamry DTA files
    if ext == ".dta":
        first_lines = _read_first_lines(filename, 50)
        first_50_lines = "".join(first_lines)
        if "ZCURVE" in first_50_lines:
            return "gamry"

    # Check for Excel files (Neware, Arbin, or Maccor)
    if ext in [".xls", ".xlsx"]:
        if _is_neware_excel(filename):
            return "neware"
        if _is_arbin_excel(filename):
            return "arbin"
        if _is_maccor_excel(filename):
            return "maccor"

    # Check for BDF parquet: read schema rather than scanning binary content.
    if _has_bdf_extension(filename) and ext == ".parquet":
        try:
            schema_cols = pl.scan_parquet(filename).collect_schema().names()
        except Exception:
            schema_cols = []
        header = ",".join(schema_cols)
        if _has_bdf_machine_header(header) or _has_bdf_label_header(header):
            return "bdf"

    # Check for BDF gzipped CSV: decompress first line only.
    if filename.name.lower().endswith(".bdf.gz"):
        gz_first = _first_line_gzipped(filename)
        if _has_bdf_machine_header(gz_first) or _has_bdf_label_header(gz_first):
            return "bdf"

    # Read first 50 lines for text-based detection (reuse if already read for .dta)
    if ext != ".dta":
        first_lines = _read_first_lines(filename, 50)
    first_line = first_lines[0] if first_lines else ""
    first_10_lines = "".join(first_lines[:10])
    first_50_lines = "".join(first_lines)

    # Check for Gamry: GALVEIS tag or ZCURVE table in first 50 lines
    if "GALVEIS" in first_10_lines or "ZCURVE" in first_50_lines:
        return "gamry"

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
        # Arbin CSV signature: must come before Maccor (Arbin's "Test Time (s)"
        # would otherwise match the Maccor time-column check).
        if _has_arbin_signature(first_line):
            return "arbin"
        # BioLogic plain CSV: Ecell_V or Ewe_V voltage column
        from .biologic import BiologicCSV

        if BiologicCSV.sniff(first_line):
            return "biologic csv"
        # BaSyTec CSV signature: run_time, c_vol, c_cur columns
        if all(s in first_line for s in ["run_time", "c_vol", "c_cur"]):
            return "basytec"
        # Repower signature
        if all(s in first_line for s in ["Cycle ID", "Step ID", "Record ID"]):
            return "repower"
        # Neware CSV signature
        if any(ts in first_line for ts in NEWARE_TIMESTAMP_COLS):
            return "neware"
        # Maccor CSV signature
        if "Step" in first_line and _has_maccor_time_col(first_line):
            return "maccor"

    # Maccor .txt files or .+3digits: tab- or comma-separated with "Step" in header
    if is_maccor_text_extension(ext):
        for line in first_lines:
            if "Step" in line and _has_maccor_time_col(line):
                return "maccor"

    # Biologic .txt, .mpt, or .mpr files
    if _is_biologic_text_extension(ext):
        for line in first_lines:
            if _has_biologic_time_col(line):
                return "biologic" if ext == ".txt" else "biologic mpt"

    # BDF text-based fallback — runs after all vendor-specific CSV checks so
    # that a Biologic/Maccor/Neware/etc. CSV with matching columns still wins.
    # Primary signal: all three BDF machine-readable names in the header
    # (distinctive enough that a plain .csv is safe to auto-detect).
    if _has_bdf_machine_header(first_line):
        return "bdf"
    # Secondary signal: all three preferred labels AND a BDF-associated
    # extension — plain .csv with only preferred labels is too ambiguous.
    if _has_bdf_extension(filename) and _has_bdf_label_header(first_line):
        return "bdf"

    raise ValueError(
        f"Could not automatically detect reader type for file: {filename}. "
        f"Please specify the reader type explicitly."
    )


def _is_arbin_excel(filename: Path) -> bool:
    """Check if an Excel file is an Arbin export by examining the header."""
    try:
        _df, column_names = read_excel_and_get_column_names(filename)
        # column_names are lowercased by the helper.
        has_required = all(s.lower() in column_names for s in ARBIN_REQUIRED)
        has_time = any(c.startswith("test time") for c in column_names)
        return has_required and has_time
    except Exception:
        return False


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
