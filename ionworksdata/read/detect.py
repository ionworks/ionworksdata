"""Automatic reader detection based on file content."""

from __future__ import annotations

import gzip
from pathlib import Path
import re

import polars as pl

from ._utils import (
    is_maccor_text_extension,
)

NEWARE_TIMESTAMP_COLS = ["DateTime", "Absolute Time", "Date(h:min:s.ms)", "Date"]

# Columns unique to a Neware BTSDA export, used to disambiguate a bare "Date".
NEWARE_SIGNATURE_COLS = [
    "Chg. Cap.(Ah)",
    "DChg. Cap.(Ah)",
    "Spec. Cap.(mAh/g)",
    "Step Type",
]

ARBIN_TIME_PREFIX = "Test Time"


def _has_arbin_signature(header: str) -> bool:
    # Reader imports are deferred throughout this module: readers subclass
    # BaseReader from .read, which imports .detect.
    from .arbin import ARBIN_SIGNATURE_COLUMNS

    return (
        all(s in header for s in ARBIN_SIGNATURE_COLUMNS)
        and ARBIN_TIME_PREFIX in header
    )


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


def _is_maccor_excel(filename: Path) -> bool:
    """
    Check if an Excel file is a Maccor file by examining column headers.

    Parameters
    ----------
    filename : Path
        Path to the Excel file to check.

    Returns
    -------
    bool
        True if the file appears to be a Maccor file, False otherwise.
    """
    from .maccor import Maccor

    return Maccor.sniff_excel(filename)


def _read_first_lines(filename: Path, num_lines: int = 10) -> list[str]:
    """Read first lines from file, trying multiple encodings."""
    for encoding in ["utf-8", "latin1", "ISO-8859-1"]:
        try:
            with open(filename, encoding=encoding) as f:
                return [f.readline() for _ in range(num_lines)]
        except UnicodeDecodeError:
            continue
    return []


def _read_line(filename: Path, index: int) -> str:
    """Return the zero-indexed line *index* of a text file, or "" if absent.

    Parameters
    ----------
    filename : Path
        Path to the file.
    index : int
        Zero-based line number to return.

    Returns
    -------
    str
        The requested line, or an empty string if it could not be read.
    """
    for encoding in ["utf-8", "latin1", "ISO-8859-1"]:
        try:
            with open(filename, encoding=encoding) as f:
                for i, line in enumerate(f):
                    if i == index:
                        return line
            return ""
        except UnicodeDecodeError:
            continue
    return ""


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

    # Reader imports are deferred throughout this module: readers subclass
    # BaseReader from .read, which imports .detect.
    from .maccor import Maccor

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
        from .arbin import Arbin

        if Arbin.sniff_excel(filename):
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

    # Generic parquet fallback for any .parquet file that didn't match BDF.
    if ext == ".parquet":
        return "parquet"

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

    # BaSyTec .txt result export: a "~"-prefixed preamble naming the system.
    if ext == ".txt" and "Resultfile from Basytec" in first_10_lines:
        return "basytec"

    # Check for Novonix: starts with [Summary] and contains "Novonix"
    if "[Summary]" in first_line and "Novonix" in first_10_lines:
        return "novonix"

    # Check for Maccor: contains "Date of Test:" in first line
    if "Date of Test:" in first_line:
        for line in first_lines:
            if "Step" in line and Maccor.has_time_column(line):
                return "maccor"

    # CSV file checks
    if ext == ".csv":
        # Arbin CSV signature: must come before Maccor (Arbin's "Test Time (s)"
        # would otherwise match the Maccor time-column check).
        if _has_arbin_signature(first_line):
            return "arbin"
        # Before Maccor and Neware: a Digatron header also carries "Step".
        from .digatron import Digatron

        if Digatron.sniff(first_line):
            return "digatron"
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
        # Neware CSV signature. A bare "Date" is too common to stand alone, so
        # it needs a Neware-specific companion column.
        if any(ts in first_line for ts in NEWARE_TIMESTAMP_COLS[:-1]):
            return "neware"
        if "Date" in first_line and any(c in first_line for c in NEWARE_SIGNATURE_COLS):
            return "neware"
        # Maccor CSV signature. Scan the preamble too: exports that lead with
        # a metadata block put the column header well below line 1.
        for line in first_lines:
            if "Step" in line and Maccor.has_time_column(line):
                return "maccor"

    # Maccor .txt files or .+3digits: tab- or comma-separated with "Step" in header
    if is_maccor_text_extension(ext):
        for line in first_lines:
            if "Step" in line and Maccor.has_time_column(line):
                return "maccor"

    # Biologic .txt, .mpt, or .mpr files
    if _is_biologic_text_extension(ext):
        for line in first_lines:
            if _has_biologic_time_col(line):
                return "biologic" if ext == ".txt" else "biologic mpt"
        # A BT-Lab preamble can be longer than the sniff window, but it
        # declares its own length, so the header can be located exactly.
        if any("BT-Lab" in line or "EC-Lab" in line for line in first_lines):
            for line in first_lines:
                match = re.search(r"Nb header lines\s*:\s*(\d+)", line)
                if match:
                    header = _read_line(filename, int(match.group(1)) - 1)
                    if _has_biologic_time_col(header):
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
    # Deferred like the other reader imports in this module: readers subclass
    # BaseReader from .read, which imports .detect.
    from .neware import Neware

    return Neware.sniff_excel(filename)
