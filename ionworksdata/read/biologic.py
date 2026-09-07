from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re
from typing import Any

import iwutil
import polars as pl

import ionworksdata as iwdata

from ._metadata import (
    localize,
    nest_metadata,
    parse_datetime,
    take_known_start_time,
)
from .read import BaseReader

#: Sniffed rather than assumed: a European-locale BT-Lab export writes
#: ``3,69E+000`` under the same extension as a dot-decimal one.
_DECIMAL_COMMA_RE = re.compile(r"-?\d+,\d+([eE][+-]?\d+)?")


#: Preamble labels this reader records, mapped to canonical metadata keys.
#: Only top-level (unindented) labels are matched -- see
#: :func:`_parse_biologic_preamble` for why the indented ones cannot be.
_BIOLOGIC_LABELS: dict[str, str] = {
    # The ``.mps`` settings file is the schedule that produced the run.
    "loaded setting file": "procedure",
    "run on channel": "channel_number",
    "user": "operator",
    "device": "instrument",
    "acquisition started on": "start_time",
    "comments": "notes",
}

#: Keyed by parent: ``Saved on :`` nests bare names (``File``, ``Directory``,
#: ``Host``) that a parent-blind parse reads as fields of the run.
_BIOLOGIC_NESTED_LABELS: dict[str, dict[str, str]] = {
    "saved on": {"directory": "source_file_path"},
}

#: ``A1 (SN 0916)`` / ``BCS-815 (SN 0839)`` -- the channel and the device each
#: carry their serial in the same parenthesised form.
_BIOLOGIC_SERIAL_RE = re.compile(r"^(?P<name>.*?)\s*\(SN\s*(?P<serial>[^)]+)\)\s*$")

#: ``BT-Lab for windows v1.73 (software)`` -- an unlabelled line, so it is
#: matched by shape rather than by a label.
_BIOLOGIC_SOFTWARE_RE = re.compile(
    r"^(?P<name>.+?\sv[\d.]+)\s*\(software\)\s*$", re.IGNORECASE
)

#: Month-first, per the one fixture whose cell batch dates it.
_BIOLOGIC_START_FORMATS = (
    "%m/%d/%Y %H:%M:%S.%f",
    "%m/%d/%Y %H:%M:%S",
    "%m/%d/%Y %H:%M",
    "%m/%d/%Y",
)

#: A European export writes a day-first date no format above can represent.
#: Tried after those, so an ambiguous date stays month-first.
_BIOLOGIC_START_FORMATS_DAY_FIRST = (
    "%d/%m/%Y %H:%M:%S.%f",
    "%d/%m/%Y %H:%M:%S",
    "%d/%m/%Y %H:%M",
    "%d/%m/%Y",
)


def _split_serial(value: str) -> tuple[str, str | None]:
    """
    Split a ``name (SN serial)`` preamble value.

    Parameters
    ----------
    value : str
        Raw value, e.g. ``"A1 (SN 0916)"``.

    Returns
    -------
    tuple[str, str | None]
        The name with the serial removed, and the serial when one is present.
    """
    match = _BIOLOGIC_SERIAL_RE.match(value)
    if match is None:
        return value, None
    return match.group("name").strip(), match.group("serial").strip()


def _parse_biologic_preamble(lines: list[str]) -> dict[str, str]:
    """
    Extract metadata from a BioLogic ``.mpt``/``.txt`` preamble.

    Parameters
    ----------
    lines : list[str]
        Preamble lines, in file order, excluding the column header.

    Returns
    -------
    dict[str, str]
        Canonical key to value, for the labels the preamble carries.

    Notes
    -----
    Indentation is load-bearing. BioLogic nests a block of fields under
    ``Saved on :`` using bare names that collide with top-level ones --
    ``File``, ``Directory``, ``Host`` -- so the parse tracks which top-level
    label it is inside and looks a nested name up under that parent only. A
    parser that ignored indentation would record the export's directory as if
    it were a field of the run itself.
    """
    found: dict[str, str] = {}
    parent: str | None = None
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        indented = line[:1] in (" ", "\t")
        label, sep, value = stripped.partition(":")
        if not sep:
            # Unlabelled lines: the only one worth keeping names the software.
            software = _BIOLOGIC_SOFTWARE_RE.match(stripped)
            if software and not indented:
                found.setdefault("software_version", software.group("name").strip())
            continue
        label = label.strip().lower()
        value = value.strip()
        if indented:
            key = _BIOLOGIC_NESTED_LABELS.get(parent or "", {}).get(label)
        else:
            # A value-less top-level label opens a nested block.
            parent = label
            key = _BIOLOGIC_LABELS.get(label)
        if key is None or not value:
            continue
        found.setdefault(key, value)
    return found


def _uses_decimal_comma(lines: list[str], header_index: int, sep: str) -> bool:
    """Return True if the data rows use a comma as the decimal separator.

    Parameters
    ----------
    lines : list[str]
        All lines of the file.
    header_index : int
        Index of the column-header line.
    sep : str
        Field separator.

    Returns
    -------
    bool
        True when a sampled data field looks like ``3,69E+000``.
    """
    for line in lines[header_index + 1 : header_index + 21]:
        fields = [f.strip() for f in line.split(sep) if f.strip()]
        if not fields:
            continue
        if any(_DECIMAL_COMMA_RE.fullmatch(f) for f in fields):
            return True
    return False


class Biologic(BaseReader):
    """
    Reader for Biologic files (.mpt and .txt formats).

    The file format (separator, skiprows) is auto-detected based on extension:
    - .mpt files use tab separator
    - .txt files auto-detect separator (tab or comma)
    """

    name: str = "Biologic"
    default_options: dict[str, Any] = {
        "file_encoding": "ISO-8859-1",
        "timezone": "UTC",
        "cell_metadata": {},
    }

    @staticmethod
    def _get_file_args(
        filename: str | Path, options: dict[str, str] | None = None
    ) -> tuple[int, str, bool]:
        """
        Get file arguments for reading a Biologic file.

        Parameters
        ----------
        filename : str | Path
            Path to the Biologic file.
        options : dict[str, str] | None
            Options dict with file_encoding key.

        Returns
        -------
        tuple[int, str, bool]
            Tuple of (skiprows, sep, decimal_comma).
        """
        encoding = options["file_encoding"]
        ext = Path(filename).suffix.lower()

        with open(filename, encoding=encoding) as f:
            lines = f.readlines()

        # Determine separator based on file type
        if ext == ".mpt":
            sep = "\t"
        else:
            # Auto-detect for other file types
            sep = "\t" if any("\t" in line for line in lines[:20]) else ","

        # Determine skiprows
        # Try to find line with "Nb header lines : int" and extract the int
        header_line = next((line for line in lines if "Nb header lines" in line), None)
        if header_line is not None:
            match = re.search(r"Nb header lines\s*:\s*(\d+)", header_line)
            if match:
                idx = int(match.group(1)) - 1
                return idx, sep, _uses_decimal_comma(lines, idx, sep)
            # If header_line found but regex doesn't match, fall through
        # Fallback to looking for the line that contains "mode"
        for i, row in enumerate(lines):
            if "mode" in row or "freq/Hz" in row:
                return i, sep, _uses_decimal_comma(lines, i, sep)
        # If neither method worked, raise an error
        raise ValueError("Could not find header row in Biologic file")

    @staticmethod
    def _get_column_renamings() -> dict[str, str]:
        """
        Get standard column renaming mappings for Biologic files.

        Returns
        -------
        dict[str, str]
            Dictionary mapping original column names to standardized names.
        """
        return {
            "Ecell/V": "Voltage [V]",
            "Ewe/V": "Voltage [V]",
            "<Ewe>/V": "Voltage [V]",
            "I/mA": "Current [mA]",
            "<I>/mA": "Current [mA]",
            "time/s": "Time [s]",
            "Temperature/°C": "Temperature [degC]",
            "Temperature/Â°C": "Temperature [degC]",
            "Temperature/degC": "Temperature [degC]",
            "Ns": "Step from cycler",
            "Cycle number": "Cycle from cycler",
            "cycle number": "Cycle from cycler",
            "freq/Hz": "Frequency [Hz]",
            "Re(Z)/Ohm": "Z_Re [Ohm]",
            "-Im(Z)/Ohm": "-Z_Im [Ohm]",
            "|Z|/Ohm": "Z_Mod [Ohm]",
            "Phase(Z)/deg": "Z_Phase [deg]",
        }

    def run(
        self,
        filename: str | Path,
        extra_column_mappings: dict[str, str] | None = None,
        options: dict[str, str] | None = None,
    ) -> pl.DataFrame:
        """
        Read and process data from a BioLogic file.

        The following column mappings are applied by default:

            - "Ecell/V" -> "Voltage [V]"
            - "Ewe/V" -> "Voltage [V]"
            - "I/mA" -> "Current [mA]"
            - "<I>/mA" -> "Current [mA]"
            - "time/s" -> "Time [s]"
            - "Temperature/°C" -> "Temperature [degC]"
            - "Ns" -> "Step from cycler"
            - "Cycle number" -> "Cycle from cycler"
            - "freq/Hz" -> "Frequency [Hz]"
            - "Re(Z)/Ohm" -> "Z_Re [Ohm]"
            - "-Im(Z)/Ohm" -> "Z_Im [Ohm]" (raw imaginary part; BioLogic reports
              the negated value, which is flipped back on read)
            - "|Z|/Ohm" -> "Z_Mod [Ohm]"
            - "Phase(Z)/deg" -> "Z_Phase [deg]"

        Additional column mappings can be provided via extra_column_mappings.

        Parameters
        ----------
        filename : str
            Path to the BioLogic file to be read (.mpt or .txt).
        extra_column_mappings : dict of str to str, optional
            Dictionary of additional column mappings. Keys are original column
            names, values are the new column names.
        options : dict of str to str, optional
            Dictionary of options for reading the BioLogic file. Options are:

            - file_encoding: str, optional
                Encoding format for the file. Default is "ISO-8859-1".
            - timezone: str, optional
                Timezone for timestamps. Default is "UTC".

        Returns
        -------
        pl.DataFrame
            Processed data with standardized column names and units.
        """
        options = iwutil.check_and_combine_options(self.default_options, options)
        skiprows, sep, decimal_comma = self._get_file_args(filename, options)

        # Read headers first to determine which columns exist
        # This is necessary because polars 1.x mishandles schema_overrides
        # for non-existent columns
        header_df = pl.read_csv(
            filename,
            encoding=options["file_encoding"],
            separator=sep,
            skip_rows=skiprows,
            n_rows=0,
            infer_schema_length=0,
            truncate_ragged_lines=True,
        )

        # BioLogic data is entirely numeric, so force all columns to Float64 to
        # avoid type inference issues where initial integer-like values (e.g., "0")
        # cause a column to be inferred as Int64, then fail on later float values.
        schema_overrides = {col: pl.Float64 for col in header_df.columns}
        schema_overrides.pop("Date", None)

        data = pl.read_csv(
            filename,
            encoding=options["file_encoding"],
            separator=sep,
            skip_rows=skiprows,
            schema_overrides=schema_overrides,
            truncate_ragged_lines=True,
            decimal_comma=decimal_comma,
        )

        column_renamings = self._get_column_renamings()
        column_renamings.update(extra_column_mappings or {})

        # Map-ordered, so ``Ecell/V`` outranks ``Ewe/V`` for ``Voltage [V]``.
        existing_renames = iwdata.util.resolve_renamings(
            column_renamings, data, priority=extra_column_mappings
        )
        if existing_renames:
            data = data.rename(existing_renames)

        # BioLogic reports the negated imaginary part, so flip it into the
        # canonical Z_Im [Ohm] (raw Im(Z), negative for capacitive behaviour).
        if "-Z_Im [Ohm]" in data.columns:
            data = data.with_columns((-pl.col("-Z_Im [Ohm]")).alias("Z_Im [Ohm]"))
            data = data.drop("-Z_Im [Ohm]")

        # Convert current to amps
        if "Current [mA]" in data.columns:
            data = data.with_columns(
                (pl.col("Current [mA]") / 1000.0).alias("Current [A]")
            )
            data = data.drop("Current [mA]")

        # Keep the derived canonical columns, not the intermediate rename targets.
        columns_keep = list(
            set(column_renamings.values()) - {"Current [mA]", "-Z_Im [Ohm]"}
            | {"Current [A]", "Z_Im [Ohm]"}
        )
        return self.standard_data_processing(data, columns_keep=columns_keep)

    def read_start_time(
        self,
        filename: str | Path,
        extra_column_mappings: dict[str, str] | None = None,
        options: dict[str, str] | None = None,
    ) -> datetime | None:
        """
        Read the start time from a BioLogic file.

        Parameters
        ----------
        filename : str
            Path to the BioLogic file to be read (.mpt or .txt).
        options : dict of str to str, optional
            Dictionary of options for reading the BioLogic file.

        Returns
        -------
        datetime | None
            The start time of the BioLogic file, or None if not found.
        """
        options = iwutil.check_and_combine_options(self.default_options, options)
        skiprows, sep, _decimal_comma = self._get_file_args(filename, options)

        # Only ``Date`` is needed, so everything stays text — numeric casting
        # here would fail on a decimal-comma export for no benefit.
        data = pl.read_csv(
            filename,
            encoding=options["file_encoding"],
            separator=sep,
            skip_rows=skiprows,
            n_rows=1,
            infer_schema_length=0,
            truncate_ragged_lines=True,
        )
        try:
            start_datetime = datetime.strptime(data["Date"][0], "%Y-%m-%d %H:%M:%S")
            # `localize`, not `replace(tzinfo=)`, which would take the
            # zone's LMT offset rather than the one in force.
            return localize(start_datetime, options)
        except (KeyError, pl.exceptions.ColumnNotFoundError):
            return None

    def read_metadata(
        self,
        filename: str | Path,
        options: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """
        Read the descriptive metadata from a BioLogic file's preamble.

        Parameters
        ----------
        filename : str | Path
            Path to the BioLogic file to be read (``.mpt`` or ``.txt``).
        options : dict[str, str] | None, optional
            Dictionary of options for reading the BioLogic file. ``timezone``
            (default ``"UTC"``) is applied to ``Acquisition started on``, which
            the preamble writes without an offset.

        Returns
        -------
        dict[str, Any]
            Preamble fields grouped the way the measurement API defines them:

            - ``protocol["name"]``: the ``Loaded Setting File``, i.e. the
              ``.mps`` settings file that produced the run. Kept verbatim,
              path and all: BioLogic writes an absolute path here, and the
              lab's directory layout is part of how the schedule is
              identified, so trimming to a basename would merge two settings
              files that differ only by folder.
            - ``test_setup["channel_number"]``: ``Run on channel``, kept as
              written (``"A1"``) -- BioLogic channels are not bare numbers.
            - ``test_setup["cycler_serial_number"]``: the ``(SN ...)`` on
              ``Device`` when it carries one, else the one on ``Run on
              channel``. The device's own serial identifies the instrument;
              the channel's is a fallback for an export that names no device.
            - ``test_setup["instrument"]``: ``Device``, e.g. ``"BCS-815"``,
              with its serial split off into the field above.
            - ``test_setup["operator"]``: ``User``.
            - ``test_setup["source_file_path"]``: the ``Directory`` nested
              under ``Saved on``.
            - ``test_setup["software_version"]``: the ``BT-Lab for windows
              v1.73 (software)`` line, which carries no label.
            - ``test_setup["notes"]``: ``Comments``.
            - ``test_setup["cycler"]``: this reader's name.
            - ``start_time``: timezone-aware, from the data's ``Date`` column
              where the export has one, else from ``Acquisition started on``.
              Top-level, because the API models it as a column rather than
              inside a group. The column wins because it states the instant
              outright, where the preamble's locale-dependent ``06/01/2022``
              could be read either way round -- and the wrong way puts this
              five months from :meth:`read_start_time` for the same file.

            Keys the preamble does not carry are omitted, and BioLogic writes
            many of these labels with an empty value (``User :`` alone on its
            line), which counts as absent rather than as an empty string.

            A plain-CSV export has no preamble at all, so
            :class:`BiologicCSV` returns only the reader's own name; see
            :meth:`BiologicCSV.read_metadata`.
        """
        known, options = take_known_start_time(options)
        options = iwutil.check_and_combine_options(self.default_options, options)
        skiprows, _sep, _decimal_comma = self._get_file_args(filename, options)
        with open(filename, encoding=options["file_encoding"]) as handle:
            # Bounded by the header index the format states, so the data rows
            # below are never scanned for labels.
            preamble = [next(handle, "") for _ in range(skiprows)]

        parsed: dict[str, Any] = dict(_parse_biologic_preamble(preamble))

        # Both the channel and the device carry a serial in the same
        # parenthesised form; the device's wins, being the instrument's own.
        channel_serial: str | None = None
        if "channel_number" in parsed:
            parsed["channel_number"], channel_serial = _split_serial(
                parsed["channel_number"]
            )
        if "instrument" in parsed:
            parsed["instrument"], device_serial = _split_serial(parsed["instrument"])
            if device_serial is not None:
                parsed["cycler_serial_number"] = device_serial
        parsed.setdefault("cycler_serial_number", channel_serial)

        # Prefer the ``Date`` column: it states the instant outright, where the
        # preamble's locale-dependent ``06/01/2022`` could be either order.
        started = parsed.pop("start_time", None)
        from_data = known or self.read_start_time(filename, options=options)
        if from_data is not None:
            parsed["start_time"] = from_data
        elif isinstance(started, str):
            naive = parse_datetime(
                started, _BIOLOGIC_START_FORMATS + _BIOLOGIC_START_FORMATS_DAY_FIRST
            )
            if naive is not None:
                parsed["start_time"] = localize(naive, options)

        nested = nest_metadata(parsed)
        nested.setdefault("test_setup", {})["cycler"] = self.name
        return nested


class BiologicMPT(Biologic):
    """
    Reader for Biologic MPT files.

    This is an alias for the Biologic reader - both handle .mpt and .txt formats
    automatically based on file extension.
    """

    name: str = "Biologic MPT"


class BiologicCSV(BaseReader):
    """Reader for BioLogic plain CSV exports.

    BioLogic users often export cycling data to plain CSV instead of the native
    .mpt/.mpr format. These CSVs have different column names (e.g. ``Ecell_V``,
    ``I_mA``, ``cycleNumber``) that are not recognised by the generic CSV reader
    or the standard BioLogic reader.
    """

    name: str = "Biologic CSV"
    default_options: dict[str, Any] = {
        "cell_metadata": {},
    }

    #: Map raw BioLogic CSV columns to ionworks standard names.  Values that
    #: require unit conversion are handled separately in ``run()``.
    COLUMN_MAP: dict[str, str] = {
        "time_s": "Time [s]",
        "Ecell_V": "Voltage [V]",
        "Ewe_V": "Voltage [V]",
        "I_mA": "Current [mA]",
        "EnergyCharge_W_h": "Charge energy [W.h]",
        "QCharge_mA_h": "Charge capacity [mA.h]",
        "EnergyDischarge_W_h": "Discharge energy [W.h]",
        "QDischarge_mA_h": "Discharge capacity [mA.h]",
        "Temperature__C": "Temperature [degC]",
        "cycleNumber": "Cycle from cycler",
        "Ns": "Step from cycler",
    }

    #: Columns whose values are in milli-units and need ÷ 1000.
    MILLI_COLUMNS: dict[str, str] = {
        "Current [mA]": "Current [A]",
        "Charge capacity [mA.h]": "Charge capacity [A.h]",
        "Discharge capacity [mA.h]": "Discharge capacity [A.h]",
    }

    @classmethod
    def sniff(cls, first_line: str) -> bool:
        """Return True if *first_line* looks like a BioLogic plain CSV header."""
        return "Ecell_V" in first_line or "Ewe_V" in first_line

    def run(
        self,
        filename: str | Path,
        extra_column_mappings: dict[str, str] | None = None,
        options: dict[str, str] | None = None,
    ) -> pl.DataFrame:
        """Read a BioLogic plain CSV export and return standardised data.

        Parameters
        ----------
        filename : str | Path
            Path to the CSV file.
        extra_column_mappings : dict[str, str] | None, optional
            Additional column mappings on top of the built-in map.
        options : dict[str, str] | None, optional
            Reader options (currently only ``cell_metadata``).

        Returns
        -------
        pl.DataFrame
            Processed data with standard ionworks column names and SI units.
        """
        options = iwutil.check_and_combine_options(self.default_options, options)
        data = pl.read_csv(filename, infer_schema_length=10000)

        # Build the full renaming dict
        column_renamings = dict(self.COLUMN_MAP)
        column_renamings.update(extra_column_mappings or {})

        # Map-ordered, so ``Ecell_V`` outranks ``Ewe_V`` for ``Voltage [V]``.
        existing_renames = iwdata.util.resolve_renamings(
            column_renamings, data, priority=extra_column_mappings
        )
        if existing_renames:
            data = data.rename(existing_renames)

        # Batch-convert milli-unit columns to SI in a single pass
        milli_exprs = [
            (pl.col(milli_col) / 1000.0).alias(si_col)
            for milli_col, si_col in self.MILLI_COLUMNS.items()
            if milli_col in data.columns
        ]
        milli_to_drop = [mc for mc in self.MILLI_COLUMNS if mc in data.columns]
        if milli_exprs:
            data = data.with_columns(milli_exprs).drop(milli_to_drop)

        # Only include SI columns that were actually produced
        converted_si = {self.MILLI_COLUMNS[mc] for mc in milli_to_drop}
        columns_keep = list(
            (set(column_renamings.values()) - set(self.MILLI_COLUMNS.keys()))
            | converted_si
        )
        return self.standard_data_processing(data, columns_keep=columns_keep)

    def read_start_time(
        self,
        filename: str | Path,
        extra_column_mappings: dict[str, str] | None = None,
        options: dict[str, str] | None = None,
    ) -> None:
        """BioLogic plain CSV exports do not contain a start-time header."""
        return None

    def read_metadata(
        self,
        filename: str | Path,
        options: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """
        Report only the reader's name: a plain CSV export has no preamble.

        Overriding :meth:`Biologic.read_metadata` rather than inheriting it is
        the point. The parent locates the preamble via ``Nb header lines`` and
        falls back to the first line mentioning ``mode``/``freq/Hz`` when that
        is absent -- which, for a headerless CSV, lands on a data row and would
        parse column values as labelled metadata fields.

        Parameters
        ----------
        filename : str | Path
            Path to the BioLogic CSV file. Unused; the format carries no
            header to read.
        options : dict[str, str] | None, optional
            Unused, present for API compatibility with the other readers.

        Returns
        -------
        dict[str, Any]
            ``{"test_setup": {"cycler": "Biologic CSV"}}``.
        """
        return {"test_setup": {"cycler": self.name}}


def biologic_csv(
    filename: str | Path,
    extra_column_mappings: dict[str, str] | None = None,
    options: dict[str, str] | None = None,
) -> pl.DataFrame:
    """Convenience function for :class:`BiologicCSV`."""
    return BiologicCSV().run(
        filename, extra_column_mappings=extra_column_mappings, options=options
    )


def biologic(
    filename: str | Path,
    extra_column_mappings: dict[str, str] | None = None,
    options: dict[str, str] | None = None,
) -> pl.DataFrame:
    return Biologic().run(
        filename, extra_column_mappings=extra_column_mappings, options=options
    )


def biologic_mpt(
    filename: str | Path,
    extra_column_mappings: dict[str, str] | None = None,
    options: dict[str, str] | None = None,
) -> pl.DataFrame:
    return BiologicMPT().run(
        filename, extra_column_mappings=extra_column_mappings, options=options
    )
