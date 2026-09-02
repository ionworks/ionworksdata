"""Utility functions for reading cycler data files."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
import re
import sys

import fastexcel
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


# Header with a unit suffix in parentheses, e.g. "Current (A)", "Cur(mA)",
# "Aux_Temperature_1 (C)". Non-greedy base so only the trailing group is a unit.
_UNIT_SUFFIX_RE = re.compile(r"^(?P<base>.+?)\s*\((?P<unit>[^)]+)\)\s*$")


def strip_unit_suffix(col: str) -> str:
    """Normalize a worksheet header to its unit-less, dialect-free base name.

    Cyclers ship the same quantity under several spellings that differ only in
    unit suffix and word separator — ``Test Time (s)``, ``Test_Time(s)`` — so
    both are folded away and a single vocabulary matches every dialect.

    Parameters
    ----------
    col : str
        Raw header string from one worksheet.

    Returns
    -------
    str
        Lowercase base name with any trailing unit suffix removed and runs of
        whitespace or underscores collapsed to single spaces.
    """
    match = _UNIT_SUFFIX_RE.match(col)
    base = match.group("base") if match else col
    return re.sub(r"[\s_]+", " ", base).strip().lower()


def _probe_headers(filename: str | Path, sheet_name: str) -> list[str] | None:
    """Read one worksheet's header row, or None if it cannot be parsed.

    Parameters
    ----------
    filename : str | Path
        Path to the workbook.
    sheet_name : str
        Worksheet to read.

    Returns
    -------
    list[str] | None
        Raw header strings, or None when the sheet fails to parse.
    """
    try:
        with suppress_excel_dtype_warnings():
            head = pl.read_excel(
                filename, sheet_name=sheet_name, read_options={"n_rows": 1}
            )
    except Exception:
        return None
    return None if head is None else [str(c) for c in head.columns]


def list_sheet_names(filename: str | Path) -> list[str]:
    """Return a workbook's sheet names, or an empty list if it cannot be read.

    Parameters
    ----------
    filename : str | Path
        Path to the ``.xls``/``.xlsx`` workbook.

    Returns
    -------
    list[str]
        Sheet names in workbook order. Empty when the file is unreadable.
    """
    try:
        return list(fastexcel.read_excel(str(filename)).sheet_names)
    except Exception:
        return []


def iter_sheet_headers(
    filename: str | Path, skip_prefixes: Sequence[str] = ()
) -> Iterator[tuple[str, list[str]]]:
    """Yield ``(sheet_name, headers)`` for each candidate sheet, header row only.

    Each sheet is read with ``n_rows=1``, which still costs most of a full
    sheet parse: calamine parses the sheet before slicing.

    Parameters
    ----------
    filename : str | Path
        Path to the ``.xls``/``.xlsx`` workbook.
    skip_prefixes : Sequence[str], optional
        Lowercase sheet-name prefixes to skip without reading.

    Yields
    ------
    tuple[str, list[str]]
        Sheet name and its raw header strings. Sheets that fail to parse are
        skipped rather than raising.
    """
    sheet_names = list_sheet_names(filename)
    for name in sheet_names:
        if skip_prefixes and name.strip().lower().startswith(tuple(skip_prefixes)):
            continue
        headers = _probe_headers(filename, name)
        if headers is not None:
            yield name, headers


def find_data_sheet(
    filename: str | Path,
    required_columns: Sequence[str],
    skip_prefixes: Sequence[str] = (),
    prefer_prefix: str | None = None,
    trusted_name: str | None = None,
    aliases: Mapping[str, str] | None = None,
    fallback_to_first: bool = False,
) -> str | None:
    """Return the worksheet holding a workbook's time series.

    Multi-sheet cycler exports lead with metadata, EIS, or vendor summary
    sheets, so the first sheet can yield header text rather than the
    measurement. Candidates are ordered by name, then confirmed by their
    headers — which is what lets an unconventional sheet name still resolve.

    Deliberately not memoized: no portable ``stat`` field distinguishes a
    workbook replaced in place (Windows reports ``st_ino`` as 0 and leaves
    ``st_ctime`` at creation), and a cache measured ~2% end to end.

    Parameters
    ----------
    filename : str | Path
        Path to the ``.xls``/``.xlsx`` workbook.
    required_columns : Sequence[str]
        Column names a sheet must expose to qualify, already in the form
        :func:`strip_unit_suffix` produces (e.g. ``("voltage", "current")``).
    skip_prefixes : Sequence[str], optional
        Lowercase sheet-name prefixes that mark a sheet as non-data, such as
        an ``ACIM_`` EIS sweep or a ``Statistics`` summary.
    prefer_prefix : str | None, optional
        Lowercase sheet-name prefix to try ahead of the other candidates.
    trusted_name : str | None, optional
        Lowercase exact sheet name that identifies the time series by vendor
        convention. A sheet matching it is returned without header
        confirmation, so a header below a banner row does not disqualify it.
    aliases : Mapping[str, str] | None, optional
        Extra header fixups applied after :func:`strip_unit_suffix`, mapping a
        normalized name to the canonical one — for a vendor abbreviation the
        shared normalizer cannot know about, such as Neware's ``cur``.
    fallback_to_first : bool, optional
        When True and no sheet confirms, return the first candidate rather
        than None. For readers whose header row is not always row 0, where a
        one-row probe cannot confirm anything. Defaults to False.

    Returns
    -------
    str | None
        Name of the data sheet, or None if no sheet exposes the required
        columns.
    """
    try:
        sheet_names = fastexcel.read_excel(filename).sheet_names
    except Exception:
        return None

    candidates = [
        n for n in sheet_names if not n.strip().lower().startswith(tuple(skip_prefixes))
    ]
    if trusted_name is not None:
        # Returned unconfirmed: a header below a banner row is invisible to
        # the one-row probe, and would wrongly disqualify the named sheet.
        for name in candidates:
            if name.strip().lower() == trusted_name:
                return name

    if prefer_prefix is not None:
        preferred = [
            n for n in candidates if n.strip().lower().startswith(prefer_prefix)
        ]
        # Preferred names first, so an unconventional name still resolves as a
        # fallback rather than winning over the conventional one.
        candidates = preferred + [n for n in candidates if n not in preferred]

    if len(candidates) == 1:
        # Reading one row still costs most of a full sheet parse (calamine
        # parses the sheet before slicing), so skip confirming the only option.
        return candidates[0]

    alias_map = dict(aliases or {})
    for name in candidates:
        headers = _probe_headers(filename, name)
        if headers is None:
            continue
        bases = set()
        for col in headers:
            base = strip_unit_suffix(col)
            bases.add(alias_map.get(base, base))
        if all(req in bases for req in required_columns):
            return name
    # A header below row 0 is invisible to a one-row probe, so confirmation
    # failing does not mean the sheet is wrong.
    if fallback_to_first and candidates:
        return candidates[0]
    return None
