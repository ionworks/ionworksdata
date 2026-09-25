from __future__ import annotations

from datetime import datetime
from io import BytesIO, StringIO
from pathlib import Path
import re
from typing import Any

import fastexcel
import iwutil
import polars as pl

import ionworksdata as iwdata

from ._metadata import (
    localize,
    nest_metadata,
    parse_datetime,
    take_known_start_time,
)
from ._utils import find_data_sheet, iter_sheet_headers, list_sheet_names
from .read import BaseReader

# A Neware time-series sheet exposes a voltage and a current column; the
# metadata sheets that lead a BTSDA workbook expose neither.
_NEWARE_DATA_SHEET_REQUIRED = ("voltage", "current")

# BTSDA abbreviates current as ``Cur(mA)``; the shared normalizer strips the
# unit but cannot know the vendor's abbreviation.
_NEWARE_HEADER_ALIASES = {"cur": "current"}

# Signature columns for detection. Kept beside the reader so a new BTSDA
# header dialect is added in one place rather than two.
_NEWARE_TIMESTAMP_COLUMNS = (
    "DateTime",
    "Absolute Time",
    "Date(h:min:s.ms)",
    "Date",
)
_NEWARE_CURRENT_COLUMNS = ("Current (mA)", "Cur(mA)", "Current(A)", "Current (A)")
_NEWARE_VOLTAGE_COLUMNS = ("Voltage (V)", "Voltage(V)")


#: The default 100 rows guesses Int64 from an auxiliary column's leading zeros,
#: then fails on a decimal further down.
_INFER_SCHEMA_ROWS = 10000


#: The metadata sheet a Neware BTSDA workbook leads with.
_NEWARE_INFO_SHEET = "info"

#: ``Sorting files`` and ``Class`` are omitted as export bookkeeping, and
#: ``P/N`` as the start time again in a filename-safe spelling.
_NEWARE_INFO_LABELS: dict[str, str] = {
    # Both name the schedule; ``Process name`` is the labelled pair above the
    # block, ``Step file`` the column inside it. They agree in practice.
    "process name": "procedure",
    "step file": "procedure",
    "channel": "channel_number",
    "device": "instrument",
    "starting time": "start_time",
    "end time": "end_time",
    "remarks": "notes",
}

#: Neware writes these loose under the banner with no label of their own, so
#: they can only be found by their offset from it.
_NEWARE_INFO_BANNER = "test information"
_NEWARE_INFO_BANNER_CELLS: dict[str, int] = {
    "test_name": 1,
    "export_date": 2,
    "source_file_path": 3,
}

#: Neware writes plain ISO-like timestamps without an offset.
_NEWARE_INFO_DATETIME_FORMATS = (
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y-%m-%d %H:%M",
)


class Neware(BaseReader):
    name: str = "Neware"
    default_options: dict[str, Any] = {
        "cell_metadata": {},
        "file_encoding": "utf-8",
        "sheets": None,
    }

    # Time-series sheet in a multi-sheet Neware BTSDA workbook (the others hold
    # metadata: unit/test/cycle/step).
    _btsda_record_sheet = "record"

    @classmethod
    def sniff_excel(cls, filename: str | Path) -> bool:
        """Return True if *filename* is a Neware ``.xls``/``.xlsx`` export.

        Every sheet is checked, because a BTSDA workbook leads with a metadata
        sheet carrying none of the signature columns. Headers are read one row
        at a time rather than resolving the data sheet, so detection does not
        pay for the sheet-selection probe of a file it is about to reject.

        The signature is deliberately looser than the sheet-selection
        criteria: a timestamp plus *either* current or voltage, so a workbook
        missing one of the two still routes here instead of failing detection.

        Parameters
        ----------
        filename : str | Path
            Path to the workbook to inspect.

        Returns
        -------
        bool
            True when some sheet carries the Neware signature.
        """
        for _name, headers in iter_sheet_headers(filename):
            has_timestamp = any(c in headers for c in _NEWARE_TIMESTAMP_COLUMNS)
            has_current = any(c in headers for c in _NEWARE_CURRENT_COLUMNS)
            has_voltage = any(c in headers for c in _NEWARE_VOLTAGE_COLUMNS)
            if has_timestamp and (has_current or has_voltage):
                return True
        return False

    # Raw column names that should be read as Float64 to avoid type inference issues
    # where initial integer-like values (e.g., "0") cause truncation of decimal values
    _raw_numeric_columns = [
        "Current (mA)",
        "Cur(mA)",
        "Current (A)",
        "Current(A)",
        "Voltage (V)",
        "Voltage(V)",
        "Temperature 1 (degC)",
    ]

    @staticmethod
    def _resolve_csv_encoding(
        filename: Path, preferred_encoding: str | None = None
    ) -> tuple[str, list[str], str]:
        """Resolve a usable CSV encoding by reading the entire file.

        Each candidate encoding is tested by decoding the **whole** file, not just the
        header line.  This avoids false positives where a header is pure ASCII but
        Latin-1 bytes appear deeper in the body.  The decoded content is returned so
        that callers can pass it directly to Polars without re-reading the file.

        Parameters
        ----------
        filename : Path
            Path to the CSV file.
        preferred_encoding : str | None, optional
            User-specified encoding to try first.  ``"utf8-lossy"`` is normalised to
            ``"utf-8"`` for Python's ``open()``; callers should use the original
            ``file_encoding`` value when choosing the Polars encoding.

        Returns
        -------
        tuple[str, list[str], str]
            ``(resolved_encoding, header_row, file_content)``.
        """
        import csv

        encodings_to_try: list[str] = []
        for encoding in [preferred_encoding, "utf-8", "latin-1"]:
            if not encoding:
                continue
            # Normalise utf8-lossy to utf-8 so Python's open() accepts it.
            normalised = "utf-8" if encoding.lower() == "utf8-lossy" else encoding
            if normalised not in encodings_to_try:
                encodings_to_try.append(normalised)

        last_decode_error: UnicodeDecodeError | None = None
        for encoding in encodings_to_try:
            try:
                with open(filename, encoding=encoding, newline="") as f:
                    content = f.read()
                header = next(csv.reader(StringIO(content)))
                return encoding, header, content
            except UnicodeDecodeError as exc:
                last_decode_error = exc
                continue

        if last_decode_error is not None:
            raise last_decode_error
        raise ValueError(f"Could not determine a readable encoding for {filename}")

    def _read_file_data(
        self,
        filename: str | Path,
        sheets: dict | None = None,
        file_encoding: str = "utf-8",
    ) -> pl.DataFrame:
        """Read data from CSV or Excel with Polars, optional sheet filtering."""
        filename = Path(filename)

        if filename.suffix.lower() in [".xls", ".xlsx"]:
            # Read Excel file with Polars
            if sheets is None:
                # A Neware BTSDA export splits into several sheets with the
                # time series on the ``record`` sheet, while the first sheet is
                # metadata and lacks the data columns. Auto-select ``record``
                # when present; otherwise read the first sheet as before.
                record_sheet = find_data_sheet(
                    filename,
                    required_columns=_NEWARE_DATA_SHEET_REQUIRED,
                    trusted_name=self._btsda_record_sheet,
                    aliases=_NEWARE_HEADER_ALIASES,
                )
                df_pl = pl.read_excel(filename, sheet_name=record_sheet)
                # Cast raw numeric columns to Float64 to handle type inference issues
                df_pl = self._coerce_numeric_columns(
                    df_pl, columns=self._raw_numeric_columns
                )
                return df_pl

            # Get all sheet names in the Excel file (using fastexcel for sheet discovery)
            xl_reader = fastexcel.read_excel(filename)
            available_sheets = xl_reader.sheet_names

            # Determine which sheets to read based on specification
            sheets_to_read = self._get_sheets_to_read(sheets, available_sheets)

            # Read and combine data from selected sheets using Polars
            dataframes_pl = []
            for sheet in sheets_to_read:
                df_pl = pl.read_excel(filename, sheet_name=sheet)
                # Cast raw numeric columns to Float64 to handle type inference issues
                df_pl = self._coerce_numeric_columns(
                    df_pl, columns=self._raw_numeric_columns
                )
                # Add sheet name as a column if reading multiple sheets
                if len(sheets_to_read) > 1:
                    df_pl = df_pl.with_columns(pl.lit(sheet).alias("Sheet"))
                dataframes_pl.append(df_pl)

            # Concatenate all polars dataframes
            if len(dataframes_pl) == 1:
                combined_df_pl = dataframes_pl[0]
            else:
                combined_df_pl = pl.concat(dataframes_pl, how="vertical_relaxed")

            return combined_df_pl

        else:
            # Read CSV file with Polars
            if sheets is not None:
                raise ValueError(
                    "Sheet selection is only supported for Excel files (.xls, .xlsx)"
                )
            # Resolve encoding by reading the whole file once, and reuse
            # the decoded content to avoid a redundant second read.
            resolved_encoding, header, content = self._resolve_csv_encoding(
                filename, file_encoding
            )
            schema_overrides = {
                col: pl.Float64 for col in self._raw_numeric_columns if col in header
            }

            if resolved_encoding.lower() in {"utf-8", "utf8"}:
                # Let Polars read the file directly for its optimised C path.
                # Honour utf8-lossy when the user explicitly requested it.
                polars_encoding = (
                    "utf8-lossy" if file_encoding.lower() == "utf8-lossy" else "utf8"
                )
                df_pl = pl.read_csv(
                    filename,
                    encoding=polars_encoding,
                    schema_overrides=schema_overrides,
                    infer_schema_length=_INFER_SCHEMA_ROWS,
                )
            else:
                # Non-UTF-8: content is already decoded to Unicode by
                # _resolve_csv_encoding, re-encode to UTF-8 bytes for Polars.
                df_pl = pl.read_csv(
                    BytesIO(content.encode("utf-8")),
                    encoding="utf8",
                    schema_overrides=schema_overrides,
                    infer_schema_length=_INFER_SCHEMA_ROWS,
                )
            return df_pl

    def _get_sheets_to_read(
        self, sheets: dict, available_sheets: list[str]
    ) -> list[str]:
        """Parse sheet specification and return list of sheet names to read."""
        if not isinstance(sheets, dict):
            raise ValueError(
                "'sheets' must be a dictionary with 'type' and 'value' keys"
            )

        if "type" not in sheets:
            raise ValueError("'sheets' dict must contain 'type' key")

        sheet_type = sheets["type"]
        sheet_value = sheets.get("value")

        if sheet_type == "name":
            if sheet_value is None:
                raise ValueError(
                    "For 'name' type, 'value' must be a sheet name or list of sheet names"
                )

            # Convert single string to list for uniform processing
            if isinstance(sheet_value, str):
                sheet_names = [sheet_value]
            elif isinstance(sheet_value, list):
                sheet_names = sheet_value
            else:
                raise ValueError(
                    "For 'name' type, 'value' must be a sheet name or list of sheet names"
                )

            sheets_to_read = []
            for sheet in sheet_names:
                if sheet in available_sheets:
                    sheets_to_read.append(sheet)
                else:
                    raise ValueError(
                        f"Sheet '{sheet}' not found in Excel file. Available sheets: {available_sheets}"
                    )

            return sheets_to_read

        elif sheet_type == "pattern":
            if sheet_value is None or not isinstance(sheet_value, str):
                raise ValueError("For 'pattern' type, 'value' must be a regex string")

            try:
                pattern = re.compile(sheet_value)
                matched_sheets = [
                    sheet for sheet in available_sheets if pattern.search(sheet)
                ]

                if not matched_sheets:
                    raise ValueError(
                        f"No sheets found matching pattern '{sheet_value}'. Available sheets: {available_sheets}"
                    )

                return matched_sheets
            except re.error as e:
                raise ValueError(f"Invalid regex pattern '{sheet_value}': {e}") from e

        elif sheet_type == "all":
            return available_sheets

        else:
            raise ValueError(
                f"Unsupported sheet type '{sheet_type}'. Supported types: 'name', 'pattern', 'all'"
            )

    def _apply_column_renamings(
        self, data: pl.DataFrame, extra_column_mappings: dict[str, str] | None = None
    ) -> tuple[pl.DataFrame, dict[str, str]]:
        """Apply column renamings to Neware files data."""
        column_renamings = {
            "Current (mA)": "Current [mA]",
            "Cur(mA)": "Current [mA]",
            "Current (A)": "Current [A]",
            "Current(A)": "Current [A]",
            "Voltage (V)": "Voltage [V]",
            "Voltage(V)": "Voltage [V]",
            "Temperature 1 (degC)": "Temperature [degC]",
            "Step ID": "Step from cycler",
            "Step": "Step from cycler",
            "Cycle ID": "Cycle from cycler",
            "Cycle": "Cycle from cycler",
            "Status": "Status",
            "DateTime": "Timestamp",
            "Absolute Time": "Timestamp",
            "Date(h:min:s.ms)": "Timestamp",
            "Date": "Timestamp",
        }
        column_renamings.update(extra_column_mappings or {})
        present_map = iwdata.util.resolve_renamings(
            column_renamings, data, priority=extra_column_mappings
        )
        if present_map:
            data = data.rename(present_map)
        return data, column_renamings

    @staticmethod
    def _ensure_timestamp_datetime(data: pl.DataFrame) -> pl.DataFrame:
        """Ensure the Timestamp column is a timezone-aware datetime."""
        ts_dtype = data.schema.get("Timestamp")
        if ts_dtype == pl.String:
            return data.with_columns(
                pl.col("Timestamp")
                .str.strptime(pl.Datetime, strict=False)
                .dt.replace_time_zone("UTC")
                .alias("Timestamp")
            )
        # Already a datetime (e.g. from Excel), just ensure UTC timezone
        if isinstance(ts_dtype, pl.Datetime):
            if ts_dtype.time_zone is None:
                return data.with_columns(
                    pl.col("Timestamp").dt.replace_time_zone("UTC").alias("Timestamp")
                )
            return data
        # Other types (e.g. Date, Int) — cast to Datetime then add timezone
        return data.with_columns(
            pl.col("Timestamp")
            .cast(pl.Datetime)
            .dt.replace_time_zone("UTC")
            .alias("Timestamp")
        )

    def _filter_1970_timestamps(self, data: pl.DataFrame) -> pl.DataFrame:
        """Filter out January 1970 timestamps if first valid timestamp is after 1970.

        These are often data artifacts from uninitialized timestamps.
        """
        # Use epoch seconds for comparison to avoid timezone issues
        jan_1970_epoch = 0  # 1970-01-01 00:00:00 UTC
        feb_1970_epoch = 2678400  # 1970-02-01 00:00:00 UTC (31 days * 86400)

        ts_epoch = pl.col("Timestamp").dt.epoch("s")
        is_jan_1970 = (ts_epoch >= jan_1970_epoch) & (ts_epoch < feb_1970_epoch)
        non_1970 = data.filter(~is_jan_1970)
        if non_1970.height > 0:
            first_valid_epoch = non_1970.select(ts_epoch.min()).item()
            if first_valid_epoch > feb_1970_epoch:
                data = non_1970
        return data

    def run(
        self,
        filename: str | Path,
        extra_column_mappings: dict[str, str] | None = None,
        options: dict[str, str] | None = None,
    ) -> pl.DataFrame:
        """
        Read and process data from a Neware file (CSV or Excel). The following column mappings are applied by default:

            - "Current (mA)", "Cur(mA)" -> "Current [mA]"
            - "Current (A)", "Current(A)" -> "Current [A]"
            - "Voltage (V)", "Voltage(V)" -> "Voltage [V]"
            - "Temperature 1 (degC)" -> "Temperature [degC]"
            - "Step ID", "Step" -> "Step from cycler"
            - "Cycle ID", "Cycle" -> "Cycle from cycler"
            - "Status" -> "Status"
            - "DateTime", "Absolute Time", "Date(h:min:s.ms)" -> "Timestamp"

        Additional column mappings can be provided via the extra_column_mappings parameter.

        Parameters
        ----------
        filename : str | Path
            Path to the Neware file to be read (supports .csv, .xls, .xlsx).
        extra_column_mappings : dict[str, str] | None, optional
            Dictionary of additional column mappings to use when reading the Neware file.
            The keys are the original column names and the values are the new column
            names. Default is None.
        options : dict[str, str] | None, optional
            Dictionary of options to use when reading the Neware file. Supported options:

            - 'cell_metadata': dictionary of metadata about the cell
            - 'file_encoding': text encoding for CSV files. Default is `'utf-8'`.
              If UTF-8 decoding fails, the reader falls back to Latin-1-compatible encodings.
            - 'sheets': dict specifying sheet selection for Excel files (.xls/.xlsx only).
              If not specified, the reader auto-selects the ``record`` sheet of a
              multi-sheet Neware BTSDA workbook when present, otherwise reads the
              first sheet (index 0). Format:

              * {'type': 'name', 'value': 'Sheet1'} for single sheet
              * {'type': 'name', 'value': ['Sheet1', 'Sheet2']} for multiple sheets
              * {'type': 'pattern', 'value': 'regex_pattern'} for pattern matching
              * {'type': 'all'} to read all sheets

        Returns
        -------
        pandas.DataFrame
            Processed data from the Neware file with standardized column names and units.
            If multiple sheets are read, a 'Sheet' column is added to identify the source sheet.

        Notes
        -----
        This function reads a Neware file (CSV or Excel), processes the data, and returns a DataFrame
        with standardized column names and units. It also handles data cleaning tasks such
        as removing NaNs and converting the datetime to seconds from start.
        For Excel files, you can specify which sheets to read using 'sheets' in options.
        """
        opts: dict[str, Any] = iwutil.check_and_combine_options(
            self.default_options, options
        )

        # Extract sheet selection options
        sheets = opts.get("sheets", None)
        file_encoding = opts["file_encoding"]

        # Load data and rename columns
        data = self._read_file_data(filename, sheets, file_encoding=file_encoding)
        data, column_renamings = self._apply_column_renamings(
            data, extra_column_mappings
        )

        # Convert datetime to seconds from start (parse and add UTC timezone)
        data = self._ensure_timestamp_datetime(data)

        # Filter out January 1970 timestamps if they appear to be data artifacts
        data = self._filter_1970_timestamps(data)

        # Sort by Timestamp to ensure monotonic time (important for multi-sheet data)
        data = data.sort("Timestamp")

        # Compute Time [s] from earliest Timestamp
        start_epoch = data.select(pl.col("Timestamp").dt.epoch("s").min()).item()
        data = data.with_columns(
            (pl.col("Timestamp").dt.epoch("s") - start_epoch).alias("Time [s]")
        )

        # Convert current to amps
        if "Current [mA]" in data.columns:
            data = data.with_columns(
                (pl.col("Current [mA]") / 1000.0).alias("Current [A]")
            )
            data = data.drop("Current [mA]")

        # Keep only the columns we care about
        columns_keep = list(
            set(column_renamings.values()) - {"Current [mA]", "Status", "Timestamp"}
            | {"Current [A]", "Time [s]"}
        )
        data = self.standard_data_processing(data, columns_keep=columns_keep)

        return data

    def read_start_time(
        self,
        filename: str | Path,
        extra_column_mappings: dict[str, str] | None = None,
        options: dict[str, str] | None = None,
    ) -> datetime:
        """
        Read the start time from a Neware file (CSV or Excel).

        Parameters
        ----------
        filename : str | Path
            Path to the Neware file to be read (supports .csv, .xls, .xlsx).
        extra_column_mappings : dict[str, str] | None, optional
            Dictionary of additional column mappings to use when reading the Neware file.
        options : dict[str, str] | None, optional
            Options for reading the file. See :func:`ionworksdata.read.Neware.run`.
            Can include 'sheets' specification for Excel files.

        Returns
        -------
        datetime
            The start time of the Neware file.
        """
        opts: dict[str, Any] = iwutil.check_and_combine_options(
            self.default_options, options
        )

        # Extract sheet selection options
        sheets = opts.get("sheets", None)
        file_encoding = opts["file_encoding"]

        data = self._read_file_data(filename, sheets, file_encoding=file_encoding)
        data, _ = self._apply_column_renamings(data, extra_column_mappings)

        # Convert timestamp to datetime with UTC timezone
        data = self._ensure_timestamp_datetime(data)

        # Filter out January 1970 timestamps if they appear to be data artifacts
        data = self._filter_1970_timestamps(data)

        start_timestamp = data.select(pl.col("Timestamp").min()).item()
        start_datetime = iwdata.util.check_and_convert_datetime(start_timestamp)
        return start_datetime

    @staticmethod
    def _find_info_sheet(filename: str | Path) -> str | None:
        """Return the workbook's ``Info`` sheet name, if it has one.

        Via :func:`list_sheet_names`, which yields ``[]`` for a workbook it
        cannot open -- metadata is descriptive, so a protected or truncated
        file degrades to the reader's name rather than raising here.
        """
        return next(
            (
                n
                for n in list_sheet_names(filename)
                if n.strip().lower() == _NEWARE_INFO_SHEET
            ),
            None,
        )

    @classmethod
    def _read_info_sheet(cls, filename: str | Path) -> dict[str, str]:
        """
        Read the metadata a Neware workbook keeps on its ``Info`` sheet.

        Parameters
        ----------
        filename : str | Path
            Path to the ``.xls``/``.xlsx`` workbook.

        Returns
        -------
        dict[str, str]
            Canonical key to value, empty when the workbook has no ``Info``
            sheet (a single-sheet export does not).

        Notes
        -----
        The sheet mixes three layouts, and rows are located by content rather
        than by index because which of them a given export writes varies:

        - a ``Test information`` banner whose next three rows are the export
          filename, the export time and the source path, each written loose
          with no label of its own;
        - ``<label>:`` / value pairs in the first two columns
          (``Process name:``);
        - a labelled block: a row of column names with values on the row below.
        """
        sheet = cls._find_info_sheet(filename)
        if sheet is None:
            return {}
        try:
            grid = pl.read_excel(filename, sheet_name=sheet, has_header=False)
        except Exception:
            # Listing the sheets can succeed on a workbook whose contents then
            # fail to parse; descriptive metadata is not worth raising over.
            return {}

        def cell(row: int, column: int) -> str | None:
            if row < 0 or row >= grid.height or column >= grid.width:
                return None
            value = grid.row(row)[column]
            return None if value is None else str(value).strip() or None

        found: dict[str, str] = {}
        for row_index in range(grid.height):
            first = cell(row_index, 0)
            if first is None:
                continue
            label = first.rstrip(":").strip().lower()

            if label == _NEWARE_INFO_BANNER:
                for key, offset in _NEWARE_INFO_BANNER_CELLS.items():
                    value = cell(row_index + offset, 0)
                    if value is not None:
                        found.setdefault(key, value)
                continue

            # A ``<label>:`` / value pair. Only the colon form is treated this
            # way: a bare first cell is a data value of the block below.
            if first.endswith(":"):
                key = _NEWARE_INFO_LABELS.get(label)
                value = cell(row_index, 1)
                if key is not None and value is not None:
                    found.setdefault(key, value)
                continue

            # A labelled block: recognise it by its first cell being a label
            # we know, then read the row beneath it.
            if _NEWARE_INFO_LABELS.get(label) is None:
                continue
            for column, header in enumerate(grid.row(row_index)):
                if header is None:
                    continue
                key = _NEWARE_INFO_LABELS.get(str(header).rstrip(":").strip().lower())
                value = cell(row_index + 1, column)
                if key is not None and value is not None:
                    found.setdefault(key, value)
        return found

    def read_metadata(
        self,
        filename: str | Path,
        options: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """
        Read the descriptive metadata a Neware export carries.

        Parameters
        ----------
        filename : str | Path
            Path to the Neware file (``.csv``, ``.xls``, or ``.xlsx``).
        options : dict[str, str] | None, optional
            Options for reading the file. See :func:`ionworksdata.read.Neware.run`.
            There is no ``timezone`` option: this reader treats Neware
            timestamps as UTC throughout (see
            :meth:`_ensure_timestamp_datetime`), and the sheet's own
            timestamps are read the same way, so the two agree.

        Returns
        -------
        dict[str, Any]
            Fields grouped the way the measurement API defines them:

            - ``protocol["name"]``: the schedule ``.xml``, from ``Process
              name`` or the block's ``Step file``.
            - ``test_setup["channel_number"]``: ``Channel``.
            - ``test_setup["instrument"]``: ``device``, the cycler unit's id.
            - ``test_setup["test_name"]``: the export filename the ``Test
              information`` banner opens with.
            - ``test_setup["source_file_path"]``: the path it was exported from.
            - ``test_setup["export_date"]``: when it was exported, ISO.
            - ``test_setup["end_time"]``: ``End time``, ISO -- the API models
              only ``start_time`` as a datetime of its own.
            - ``test_setup["notes"]``: ``Remarks``.
            - ``test_setup["cycler"]``: this reader's name.
            - ``start_time``: from ``Starting time``, else from
              :meth:`read_start_time`, which reads the first data row. UTC, as
              everywhere else in this reader.

            Keys the file does not carry are omitted. A ``.csv`` export and a
            single-sheet workbook have no ``Info`` sheet at all, so for those
            the result is the reader's name plus ``start_time``.
        """
        known, options = take_known_start_time(options)
        # Called for its validation, not its result: a caller passing
        # `timezone` is told it is unrecognized rather than silently ignored.
        iwutil.check_and_combine_options(self.default_options, options)
        parsed: dict[str, Any] = {}
        if Path(filename).suffix.lower() in (".xls", ".xlsx"):
            parsed = dict(self._read_info_sheet(filename))

        for key in ("start_time", "end_time", "export_date"):
            raw = parsed.pop(key, None)
            if not isinstance(raw, str):
                continue
            naive = parse_datetime(raw, _NEWARE_INFO_DATETIME_FORMATS)
            if naive is None:
                continue
            # UTC, not an option: `_ensure_timestamp_datetime` stamps the
            # time series UTC, so another zone here would disagree with it.
            aware = localize(naive, {})
            # start_time is a datetime column of its own; the other two ride
            # along inside test_setup, which is JSON, so they go as strings.
            parsed[key] = aware if key == "start_time" else aware.isoformat()

        if "start_time" not in parsed:
            # The first data row states the time, but reading it costs a full
            # parse, so a caller holding the frame passes the timestamp in.
            parsed["start_time"] = known or self.read_start_time(
                filename, options=options
            )

        nested = nest_metadata(parsed)
        nested.setdefault("test_setup", {})["cycler"] = self.name
        return nested


def neware(
    filename: str | Path,
    extra_column_mappings: dict[str, str] | None = None,
    options: dict[str, str] | None = None,
) -> pl.DataFrame:
    return Neware().run(
        filename, extra_column_mappings=extra_column_mappings, options=options
    )
