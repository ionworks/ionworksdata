from __future__ import annotations

import csv as csv_module
from pathlib import Path
import re
from typing import Any
import warnings

import iwutil
from pandas import errors as pd_errors
import polars as pl

from .read import BaseReader

# Name given to columns synthesized to absorb surplus fields in ragged rows.
_PADDING_PREFIX = "_ionworksdata_pad_"

VOLTAGE_COLUMNS = [
    {
        "values": [
            "Voltage[V]",
            "Voltage(V)",
            "Ewe/V",
            "Ewe_V_vs_Li",
            "<V>/V",
            "Ecell/V",
            "Potential[V]",
            "Potential(V)",
            "voltage__V",
            "h_potential__V",
            "voltage_V",
        ],
        "scale": 1,
        "shift": 0,
    },
    {
        "values": ["Voltage[mV]", "Voltage(mV)", "Ewe/mV", "<V>/mV"],
        "scale": 1e-3,
        "shift": 0,
    },
]

CURRENT_COLUMNS = [
    {
        "values": [
            "Current[A]",
            "Current(A)",
            "I/A",
            "<I>/A",
            "current__A",
            "h_current__A",
            "current_A",
        ],
        "scale": 1,
        "shift": 0,
    },
    {
        "values": ["Current[mA]", "Current(mA)", "I/mA", "<I>/mA"],
        "scale": 1e-3,
        "shift": 0,
    },
]
AREAL_CURRENT_COLUMNS = [
    {
        "values": ["Current[mA.cm-2]", "Current[mA/cm2]", "icell_mA_cm2"],
        "scale": 1,
        "shift": 0,
    }
]

TIME_COLUMNS = [
    {
        "values": [
            "Time[s]",
            "Time(s)",
            "t_s",
            "time/s",
            "TestTime(s)",
            "time__s",
            "h_test_time__s",
            "test_time_s",
        ],
        "scale": 1,
        "shift": 0,
    },
    {"values": ["Time[h]", "Time(h)", "t_h", "time/h"], "scale": 3600, "shift": 0},
]

TEMPERATURE_COLUMNS = [
    {
        "values": [
            "Temperature[°C]",
            "Temperature(°C)",
            "Temperature[degC]",
            "Temperature[C]",
            "Temperature(C)",
            "Temp(°C)",
            "Temp(C)",
            "Temp[degC]",
            "T_C",
            "[Neware_xls]T10",
            "[Neware_xls]T490",
            "temperature__C",
            "h_temperature__°C",
            "temperature_1_C",
        ],
        "scale": 1,
        "shift": 0,
    },
    {
        "values": [
            "Temperature[K]",
            "Temperature(K)",
            "Temp(K)",
            "T_K",
        ],
        "scale": 1,
        "shift": -273.15,
    },
]


def _is_padding_row(fields: list[str]) -> bool:
    """Whether the row is a label padded out with empty fields.

    Metadata preamble lines are a label followed by the separator repeated out
    to the table's width (``cell id: ,,,,,,``), so they carry no data of their
    own. Sniffing keys off this rather than a comment marker, because the files
    that need it do not use one.

    A bare title line counts too: nothing after the label to leave empty.
    """
    return all(field.strip() == "" for field in fields[1:])


def sniff_csv_structure(
    filename: str | Path, max_scan_lines: int = 50
) -> tuple[int, int, str]:
    """Locate the header row and the widest row in a delimited text file.

    Returns ``(header_index, max_fields, encoding)``; see ``Returns`` below.
    A file with no preamble yields index 0.

    Preamble lines are padded out to the table width with empty fields, so the
    header is the first non-padding line. A file whose very first line already
    carries data is left at index 0.

    Parameters
    ----------
    filename : str | Path
        Path to the delimited text file to inspect.
    max_scan_lines : int, optional
        Number of leading lines to read. Preambles are short, and scanning the
        whole file would mean reading it twice. Defaults to 50.

    Returns
    -------
    tuple[int, int, str]
        The zero-based header record index, the maximum field count observed,
        and the encoding the file decoded under.
    """
    # Count records the way skiprows does — quoting-aware — so a quoted field
    # spanning a newline cannot shift the header against the index returned.
    records: list[tuple[int, list[str]]] = []
    file_encoding = "utf-8-sig"
    for encoding in ("utf-8-sig", "latin1"):
        try:
            with open(filename, encoding=encoding, newline="") as handle:
                for i, fields in enumerate(csv_module.reader(handle)):
                    if i >= max_scan_lines:
                        break
                    records.append((i, fields))
            file_encoding = encoding
            break
        except (UnicodeDecodeError, csv_module.Error):
            records = []

    # Blank records are not header candidates, but still occupy an index.
    rows = [
        (i, fields) for i, fields in records if any(field.strip() for field in fields)
    ]
    if not rows:
        return 0, 0, file_encoding

    header_index = None
    for line_number, fields in rows:
        if not _is_padding_row(fields):
            header_index = line_number
            break
    if header_index is None:
        # All padding means the header is past the window — unless every row is
        # one field wide, which is a one-column file with no preamble at all.
        one_column = max(len(fields) for _, fields in rows) == 1
        header_index = (
            0 if one_column else _first_non_padding_record(filename, file_encoding)
        )

    max_fields = max(
        (len(fields) for line_number, fields in rows if line_number >= header_index),
        default=0,
    )
    return header_index, max_fields, file_encoding


def read_csv_tolerating_structure(filename: str | Path):
    """Read a CSV whose header may sit under a preamble and whose rows may be
    ragged, returning a pandas DataFrame.

    Two shapes defeat a plain ``read_csv`` and are handled here rather than in
    ``iwutil.read_df``, whose defaults are shared with other packages:

    - a metadata preamble above the header, skipped via ``skiprows``
    - data rows carrying more fields than the header names (typically a
      trailing separator), which otherwise raise ``ParserError``. Explicit
      ``names`` widen the frame so the surplus lands in padding columns that
      are then dropped.

    Parameters
    ----------
    filename : str | Path
        Path to the CSV file to read.

    Returns
    -------
    pandas.DataFrame
        The parsed table, with any unnamed padding columns removed.
    """
    if not isinstance(filename, (str, Path)):
        # read_df also accepts an already-loaded frame, which has no structure
        # to sniff.
        return iwutil.read_df(filename)

    if Path(filename).suffix.lower() != ".csv":
        # Sniffing a binary container yields mojibake instead of failing.
        return iwutil.read_df(filename)

    header_index, max_fields, encoding = sniff_csv_structure(filename)

    if header_index == 0 and max_fields == 0:
        return iwutil.read_df(filename)

    header_fields = _read_nth_record(filename, header_index, encoding)

    if max_fields <= len(header_fields):
        # The sample only proves the scanned records are narrow, so recover on
        # the error rather than treating it as settled.
        try:
            if header_index == 0:
                return iwutil.read_df(filename, encoding=encoding)
            return iwutil.read_df(filename, skiprows=header_index, encoding=encoding)
        except pd_errors.ParserError as error:
            max_fields = max(_parser_error_field_count(error), len(header_fields) + 1)

    try:
        return _read_with_padding(
            filename, header_index, header_fields, max_fields, encoding
        )
    except pd_errors.ParserError:
        # Pandas stops naming a count once the mismatch spans a chunk
        # boundary, so stepping error-by-error stalls there.
        widest = _widest_record(filename, encoding, header_index)
        if widest <= max_fields:
            raise
        return _read_with_padding(
            filename, header_index, header_fields, widest, encoding
        )


def _widest_record(filename: str | Path, encoding: str, start: int) -> int:
    """Field count of the widest record at or after ``start``, or 0.

    Reads the whole file, so it is only worth doing once the bounded scan has
    already proved too narrow.

    Parameters
    ----------
    filename : str | Path
        Path to the CSV file.
    encoding : str
        Encoding the file decoded under.
    start : int
        Zero-based record index to measure from.

    Returns
    -------
    int
        The widest field count seen, or 0 if the file could not be read.
    """
    try:
        with open(filename, encoding=encoding, newline="") as handle:
            return max(
                (
                    len(fields)
                    for i, fields in enumerate(csv_module.reader(handle))
                    if i >= start
                ),
                default=0,
            )
    except (UnicodeDecodeError, csv_module.Error):
        return 0


def _parser_error_field_count(error: Exception) -> int:
    """Field count pandas reported as unexpected, or 0 if it named none.

    The message has the form ``Expected 17 fields in line 8, saw 18``; the
    trailing count is the width the file actually needs.

    Parameters
    ----------
    error : Exception
        The ``ParserError`` raised while reading.

    Returns
    -------
    int
        The observed field count, or 0 when the message cannot be parsed.
    """
    match = re.search(r"saw (\d+)", str(error))
    return int(match.group(1)) if match else 0


def _read_with_padding(
    filename: str | Path,
    header_index: int,
    header_fields: list[str],
    max_fields: int,
    encoding: str = "utf-8-sig",
):
    """Read the table with extra names so surplus fields have somewhere to go.

    Parameters
    ----------
    filename : str | Path
        Path to the CSV file.
    header_index : int
        Zero-based physical line holding the column names.
    header_fields : list[str]
        The column names read from that line.
    max_fields : int
        Widest row in the file; the difference against ``header_fields`` is
        made up with padding columns.
    encoding : str, optional
        Encoding the file decoded under, from ``sniff_csv_structure``.

    Returns
    -------
    pandas.DataFrame
        The parsed table, without the padding columns.
    """
    prefix = _PADDING_PREFIX
    while any(name.startswith(prefix) for name in header_fields):
        # A real column already owns the prefix; pandas rejects duplicate names.
        prefix += "_"
    padding = [f"{prefix}{i}" for i in range(max_fields - len(header_fields))]
    data = iwutil.read_df(
        filename,
        skiprows=header_index + 1,
        names=_mangle_duplicates(header_fields) + padding,
        header=None,
        encoding=encoding,
    )

    present = [c for c in padding if c in data.columns]
    carrying = [c for c in present if data[c].notna().any()]
    if carrying:
        raise ValueError(
            f"{Path(filename).name} has {max_fields} fields in some rows but "
            f"{len(header_fields)} column names, and the surplus holds values "
            f"rather than the empty trailing field this tolerates. Add the "
            f"missing name(s) to the header row."
        )
    return data.drop(columns=present)


def _mangle_duplicates(names: list[str]) -> list[str]:
    """Suffix repeated names the way pandas does when it reads a header itself.

    Passing explicit ``names`` turns a duplicate into ``Duplicate names are not
    allowed``, so a file pandas would happily read on its own fails once the
    ragged path supplies the names.

    Parameters
    ----------
    names : list[str]
        Column names as they appear in the header record.

    Returns
    -------
    list[str]
        The names with each repeat after the first suffixed ``.1``, ``.2``, …
    """
    seen: dict[str, int] = {}
    out: list[str] = []
    for name in names:
        if name in seen:
            seen[name] += 1
            out.append(f"{name}.{seen[name]}")
        else:
            seen[name] = 0
            out.append(name)
    return out


def _first_non_padding_record(filename: str | Path, encoding: str) -> int:
    """Index of the first record that is not a padded preamble row, or 0.

    Used only when the bounded scan found nothing but padding, so the cost of
    reading further is paid on files that actually need it.

    Parameters
    ----------
    filename : str | Path
        Path to the CSV file.
    encoding : str
        Encoding the file decoded under.

    Returns
    -------
    int
        Zero-based record index of the header, or 0 if there is no such record.
    """
    try:
        with open(filename, encoding=encoding, newline="") as handle:
            for i, fields in enumerate(csv_module.reader(handle)):
                if any(f.strip() for f in fields) and not _is_padding_row(fields):
                    return i
    except (UnicodeDecodeError, csv_module.Error):
        return 0
    return 0


def _read_nth_record(
    filename: str | Path, index: int, encoding: str = "utf-8-sig"
) -> list[str]:
    """Return the fields of the zero-indexed CSV record ``index``, or ``[]``.

    Records are counted under the quoting rules ``skiprows`` uses, so a quoted
    field spanning a newline does not shift the count.

    Parameters
    ----------
    filename : str | Path
        Path to the CSV file.
    index : int
        Zero-based record number to return.
    encoding : str, optional
        Encoding the file decoded under, from ``sniff_csv_structure``.

    Returns
    -------
    list[str]
        The record's fields, or an empty list if there is no such record.
    """
    try:
        with open(filename, encoding=encoding, newline="") as handle:
            for i, fields in enumerate(csv_module.reader(handle)):
                if i == index:
                    return fields
    except (UnicodeDecodeError, csv_module.Error):
        return []
    return []


def find_column(
    data_columns: list[str], options: list[dict]
) -> tuple[str, float, float]:
    """Find the first column in a list of options that is present in a DataFrame."""
    for values_scale_shift in options:
        for column in values_scale_shift["values"]:
            if column in data_columns:
                return column, values_scale_shift["scale"], values_scale_shift["shift"]
    raise ValueError(f"Could not find appropriate column out of {options}")


# Canonical columns every reader detects after applying caller mappings.
STANDARD_CANONICAL_COLUMNS = [
    "Time [s]",
    "Voltage [V]",
    "Current [A]",
    "Current [mA.cm-2]",
    "Temperature [degC]",
]


def apply_canonical_detection(applier, mapped_targets: set[str]) -> None:
    """Walk the standard canonical-column detection plan.

    Calls ``applier(canonical_name, alias_options) -> bool`` for each
    canonical column not already supplied by the caller. ``applier`` should
    add the column to the underlying frame and return whether a match was
    found. ``mapped_targets`` is the set of canonical names the caller has
    already supplied (so detection skips them).

    Raises ``ValueError`` if Voltage or Time can be found from neither an
    alias nor ``mapped_targets`` — both are required by every downstream
    consumer, so silent omission would surface later as a much less
    actionable ``KeyError``. Current accepts an areal-current fallback;
    Temperature is optional.
    """

    def needs(target: str) -> bool:
        return target not in mapped_targets

    if needs("Voltage [V]") and not applier("Voltage [V]", VOLTAGE_COLUMNS):
        raise ValueError(f"Could not find a Voltage column out of {VOLTAGE_COLUMNS}")

    current_found = not needs("Current [A]") or applier("Current [A]", CURRENT_COLUMNS)
    if not current_found:
        if not needs("Current [mA.cm-2]"):
            current_found = True
        else:
            current_found = applier("Current [mA.cm-2]", AREAL_CURRENT_COLUMNS)
    if not current_found:
        raise ValueError(
            "Could not find a Current column out of "
            f"{CURRENT_COLUMNS + AREAL_CURRENT_COLUMNS}"
        )

    if needs("Time [s]") and not applier("Time [s]", TIME_COLUMNS):
        raise ValueError(f"Could not find a Time column out of {TIME_COLUMNS}")

    if needs("Temperature [degC]"):
        applier("Temperature [degC]", TEMPERATURE_COLUMNS)


def build_columns_keep(
    present_columns: list[str], extra_column_mappings: dict[str, str]
) -> list[str]:
    """Build the ordered list of columns to keep after detection.

    Standard canonical columns come first, then any caller-supplied mapped
    columns that aren't already in the standard set. Duplicates are removed.
    """
    user_cols = list(extra_column_mappings.values())
    columns_keep: list[str] = []
    seen: set[str] = set()
    for col in STANDARD_CANONICAL_COLUMNS + user_cols:
        if col in present_columns and col not in seen:
            columns_keep.append(col)
            seen.add(col)
    return columns_keep


def detect_canonical(data: pl.DataFrame, mapped_targets: set[str]) -> pl.DataFrame:
    """Add canonical Voltage/Current/Time/Temperature columns from aliases.

    Alias keys (in ``VOLTAGE_COLUMNS`` etc.) are space-stripped, so column
    detection matches against a stripped lookup; the original column name on
    the frame is unchanged. Targets the caller has already supplied via
    ``mapped_targets`` are skipped.
    """
    stripped_map: dict[str, str] = {}
    for c in data.columns:
        key = c.replace(" ", "")
        if key in stripped_map:
            warnings.warn(
                f"Columns {stripped_map[key]!r} and {c!r} collapse to the same"
                f" stripped key {key!r}; {c!r} will shadow the earlier column"
                " during canonical detection.",
                stacklevel=2,
            )
        stripped_map[key] = c
    stripped_cols = list(stripped_map.keys())
    new_exprs: list[pl.Expr] = []

    def applier(target: str, alias_options: list[dict]) -> bool:
        try:
            src_stripped, scale, shift = find_column(stripped_cols, alias_options)
        except ValueError:
            return False
        src = stripped_map[src_stripped]
        _warn_on_rival_columns(target, src, data.columns)
        expr = (
            pl.col(src) * scale + shift if (scale != 1 or shift != 0) else pl.col(src)
        )
        new_exprs.append(expr.alias(target))
        return True

    apply_canonical_detection(applier, mapped_targets)
    return data.with_columns(new_exprs) if new_exprs else data


def _warn_on_rival_columns(target: str, chosen: str, columns: list[str]) -> None:
    """Warn when more than one column could have supplied ``target``.

    A repeated header name (``Voltage[V]`` twice) reaches this point as
    ``Voltage[V]`` and ``Voltage[V].1``. Only one can become the canonical
    column, and the other is dropped — silently, until this says so. The first
    occurrence wins, which is the one a reader scanning the header left to
    right would take.

    Parameters
    ----------
    target : str
        Canonical column being filled, e.g. ``"Voltage [V]"``.
    chosen : str
        Source column detection selected.
    columns : list[str]
        All column names on the frame, in file order.
    """
    # Only the numeric mangle suffix counts: "Voltage[V].aux" is a column in
    # its own right, "Voltage[V].1" is the same header name twice.
    pattern = re.compile(rf"^{re.escape(chosen)}\.\d+$")
    rivals = [c for c in columns if c != chosen and pattern.match(c)]
    if rivals:
        warnings.warn(
            f"{target} could come from {len(rivals) + 1} columns named"
            f" {chosen!r}; using the first and ignoring {rivals}. Rename them in"
            " the header if a later one is the column you want.",
            stacklevel=2,
        )


def synthesize_current_a_from_ma(data: pl.DataFrame) -> pl.DataFrame:
    """If the frame has ``Current [mA]`` but no ``Current [A]``, derive the
    canonical ampere column. Otherwise return the frame unchanged.
    """
    if "Current [A]" not in data.columns and "Current [mA]" in data.columns:
        return data.with_columns((pl.col("Current [mA]") / 1000.0).alias("Current [A]"))
    return data


class CSV(BaseReader):
    name: str = "CSV"
    default_options: dict[str, Any] = {
        "cell_metadata": {},
    }

    def run(
        self,
        filename: str | Path,
        extra_column_mappings: dict[str, str] | None = None,
        options: dict[str, str] | None = None,
    ) -> pl.DataFrame:
        """
        Read a CSV file and return a Polars DataFrame with appropriate column names.

        Parameters
        ----------
        filename : str | Path
            Path to the CSV file to be read.
        extra_column_mappings : dict[str, str] | None, optional
            Dictionary of additional column mappings to use when reading the CSV file.
            The keys are the original column names and the values are the new column
            names. Default is None.
        options : dict[str, str] | None, optional
            Dictionary of options to use when reading the CSV file.

            Options are:

                - cell_metadata: dict, optional
                    Additional metadata about the cell. Default is empty dict.

        Returns
        -------
        pl.DataFrame
            Processed data from the CSV file with standardized column names and units. By
            default, only returns the columns "Time [s]", "Voltage [V]",
            "Current [A]",
            "Current [mA.cm-2]", and "Temperature [degC]".
        """
        options = iwutil.check_and_combine_options(self.default_options, options)
        extra_column_mappings = extra_column_mappings or {}

        data_pd = read_csv_tolerating_structure(filename).rename(
            columns=extra_column_mappings
        )
        data = pl.from_pandas(data_pd)
        data = detect_canonical(data, set(extra_column_mappings.values()))
        data = synthesize_current_a_from_ma(data)
        columns_keep = build_columns_keep(data.columns, extra_column_mappings)
        return self.standard_data_processing(data, columns_keep=columns_keep)

    def read_start_time(
        self,
        filename: str | Path,
        extra_column_mappings: dict[str, str] | None = None,
        options: dict[str, str] | None = None,
    ) -> None:
        warnings.warn(
            "CSV reader does not support reading start time from file",
            stacklevel=2,
        )
        return None


def csv(
    filename: str | Path,
    extra_column_mappings: dict[str, str] | None = None,
    options: dict[str, str] | None = None,
) -> pl.DataFrame:
    return CSV().run(
        filename, extra_column_mappings=extra_column_mappings, options=options
    )
