from __future__ import annotations

import csv as csv_py
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Any

import fastexcel
import iwutil
import polars as pl
import pytz

import ionworksdata as iwdata

from ._utils import (
    is_maccor_text_extension,
    read_excel_and_get_column_names,
    suppress_excel_dtype_warnings,
)
from .read import BaseReader


class Maccor(BaseReader):
    name: str = "Maccor"
    default_options: dict[str, Any] = {
        "file_encoding": "ISO-8859-1",
        "timezone": "UTC",
        "cell_metadata": {},
        "time_offset_fix": -1,  # Minimum time difference to enforce. -1 means raise error on non-increasing time
    }

    @staticmethod
    def _get_file_args(
        filename: str | Path, options: dict[str, str] | None = None
    ) -> tuple[str, list[int], str, str | None, str | None, bool]:
        # Find how many header rows to skip and set the read kwargs based on the file extension
        encoding = options["file_encoding"]
        thousands = None
        is_excel = False
        ext = Path(filename).suffix.lower()

        if ext in [".xls", ".xlsx"]:
            # Excel files - return special flag
            is_excel = True
            # For Excel, we'll handle header detection separately
            return encoding, [], ",", None, None, is_excel

        with open(filename, encoding=encoding) as f:
            if ext == ".csv":
                reader = csv_py.reader(f)
                sep = ","
                units_row = True
                comment = "#"
            elif is_maccor_text_extension(ext):
                reader = f.readlines()
                sep = "\t"
                thousands = ","
                units_row = False
                comment = None
            else:
                raise ValueError(f"Unsupported file extension: {ext}")
            for i, row in enumerate(reader):
                if "Step" in row:
                    skiprows = i
                    break
            else:
                raise ValueError("Could not find header row in Maccor file")
        if units_row:
            # skip all the header rows, plus the row after the header (which contains units)
            skiprows = list(range(skiprows)) + [skiprows + 1]
        else:
            skiprows = list(range(skiprows))
        return encoding, skiprows, sep, comment, thousands, is_excel

    def run(
        self,
        filename: str | Path,
        extra_column_mappings: dict[str, str] | None = None,
        options: dict[str, str] | None = None,
    ) -> pl.DataFrame:
        """
        Read and process data from a Maccor file. The following column mappings are applied by default:

            - "Voltage", "Volts", "Voltage (V)" -> "Voltage [V]"
            - "Current", "Amps", "Current (A)" -> "Current [A]"
            - "Prog Time", "Test (Sec)", "Test Time (sec)" -> "Time [s]"
            - "Test Time (Hr)" -> "Time [h]"
            - "Cycle", "Cyc#", "Cycle ID", "Cycle P" -> "Cycle from cycler"
            - "Step", "Step ID" -> "Step from cycler"
            - "LogTemp001", "Temperature (°C)" -> "Temperature [degC]"
            - "Status", "State", "MD" -> "Status"
            - "Capacity (Ah)", "Capacity (AHr)", "Cap. (Ah)" -> "Capacity [A.h]"
            - "Energy (Wh)", "Energy (WHr)" -> "Energy [W.h]"
            - "Chg Capacity (Ah)", "Chg Capacity (AHr)" -> "Charge capacity [A.h]"
            - "DChg Capacity (Ah)", "DChg Capacity (AHr)" -> "Discharge capacity [A.h]"
            - "Chg Energy (Wh)", "Chg Energy (WHr)" -> "Charge energy [W.h]"
            - "DChg Energy (Wh)", "DChg Energy (WHr)" -> "Discharge energy [W.h]"
            - "DPT" -> "Timestamp" (parsed as datetime, used to compute "Time [s]")
            - "Test Time" -> "Timestamp" (if datetime string, otherwise treated as numeric)

        Additional column mappings can be provided via the extra_column_mappings parameter.

        Note: Timestamp columns are parsed and used to compute "Time [s]" if not already present,
        then removed from the final output.

        Parameters
        ----------
        filename : str | Path
            Path to the Maccor file to be read. Supports:
            - .txt files (tab-separated)
            - .csv files (comma-separated with units row)
            - .xls/.xlsx files (Excel format)
            - Files with .+3digits extension (e.g., .123, .456)
        extra_column_mappings : dict of str to str, optional
            Dictionary of additional column mappings to use when reading the Maccor file.
            The keys are the original column names and the values are the new column
            names. Default is None.
        options : dict of str to str, optional
            Dictionary of options to use when reading the Maccor file.  Options are:

            - file_encoding: str, optional
                Encoding format for the Maccor file. Default is "ISO-8859-1".
                Note: encoding is not used for Excel files.
            - timezone: str, optional
                Timezone to use for the Maccor file. Default is "UTC".
            - time_offset_fix: float, optional
                Minimum time difference to enforce between consecutive points.
                If -1 (default), raises ValueError when time decreases or duplicates.
                If >= 0, ensures all time differences are at least this value using
                vectorized operations: fixed_diff = max(diff(time), time_offset_fix),
                then reconstructs time via cumsum.

        Returns
        -------
        pl.DataFrame
            Processed data from the Maccor file with standardized column names and units.

        Notes
        -----
        This function reads a Maccor file, processes the data, and returns a DataFrame
        with standardized column names and units. It also handles data cleaning and
        formatting tasks such as removing NaN values and adjusting the time to start at zero.
        Supports multiple file formats including text files (.txt, .+3digits), CSV files,
        and Excel files (.xls, .xlsx).
        """
        options = iwutil.check_and_combine_options(self.default_options, options)

        # Load data and rename columns
        encoding, skiprows, sep, comment, thousands, is_excel = self._get_file_args(
            filename, options
        )

        if is_excel:
            # Handle Excel files
            data = self._read_excel_file(filename, encoding)
        else:
            # Derive header row index and whether a units row exists from the skiprows list
            # skiprows contains all pre-header rows and optionally the units row after the header
            skip_set = set(skiprows)
            # header row is the smallest non-skipped row index
            header_index = 0
            while header_index in skip_set:
                header_index += 1
            units_row_present = (header_index + 1) in skip_set

            read_kwargs = {
                "separator": sep,
                "skip_rows": header_index,
                "truncate_ragged_lines": True,
                # dtypes will be set below after extracting header columns
            }
            if units_row_present:
                read_kwargs["skip_rows_after_header"] = 1
            # Avoid passing comment handling for broad Polars version compatibility
            # We'll rely on skip logic and header detection above.

            # Polars only supports 'utf8' and 'utf8-lossy'. If a different encoding is
            # requested, decode manually and pass a StringIO buffer to polars.
            encoding_lower = (encoding or "utf8").lower()
            # Read header line to build a dtypes mapping that forces all columns to Utf8
            with open(filename, encoding=encoding) as f:
                # advance to header row
                for _ in range(header_index):
                    f.readline()
                header_line = f.readline()
            header_reader = csv_py.reader([header_line], delimiter=sep)
            header_cols = next(header_reader)
            dtypes_map = dict.fromkeys(header_cols, pl.Utf8)
            # Use schema_overrides (newer Polars) but stay compatible with older versions
            read_kwargs["schema_overrides"] = dtypes_map
            if "dtypes" in read_kwargs:
                read_kwargs.pop("dtypes", None)

            if encoding_lower in {"utf8", "utf-8", "utf8-lossy"}:
                # Map to polars-supported encodings
                read_kwargs["encoding"] = (
                    "utf8" if encoding_lower in {"utf8", "utf-8"} else "utf8-lossy"
                )
                data = pl.read_csv(filename, **read_kwargs)
            else:
                with open(filename, encoding=encoding) as f:
                    content = f.read()
                data = pl.read_csv(StringIO(content), encoding="utf8", **read_kwargs)

        # Get standard column renamings and process Test Time column
        column_renamings = self._get_column_renamings()
        data, column_renamings = self._process_test_time_column(data, column_renamings)

        column_renamings.update(extra_column_mappings or {})
        iwdata.util.check_for_duplicates(column_renamings, data)
        # Only rename columns that exist to avoid KeyErrors in polars
        existing_renames = {
            k: v for k, v in column_renamings.items() if k in data.columns
        }
        if existing_renames:
            data = data.rename(existing_renames)

        # If STATUS column is present, drop any rows where STATUS is MSG
        # and convert to single letter format (e.g. "D" not "DCH" for discharge)
        # Note: This is done after column renaming so it catches files with
        # status columns named "State" or "MD" (which are renamed to "Status")
        if "Status" in data.columns:
            data = data.filter(pl.col("Status") != "MSG")
            data = data.with_columns(
                pl.col("Status").cast(pl.Utf8).str.slice(0, 1).alias("Status")
            )

        # If numbers were read as strings (e.g., due to thousands separators), coerce to numeric
        data = self._coerce_numeric_columns(data)

        # Parse Timestamp column and compute Time [s] if needed
        data = self._parse_timestamp_column(data)

        # Convert time to seconds
        if "Time [h]" in data.columns:
            # Ensure numeric then convert
            data = self._coerce_numeric(data, "Time [h]")
            data = data.with_columns((pl.col("Time [h]") * 3600.0).alias("Time [s]"))
            data = data.drop("Time [h]")

        # Fix unsigned current if needed
        data = self._fix_unsigned_current(data)

        # Validate and optionally fix time to be strictly increasing
        # Do this BEFORE standard_data_processing to avoid losing duplicate timestamps
        time_offset_fix = options.get("time_offset_fix", -1)
        data = self._validate_and_fix_time(data, time_offset_fix)

        # Keep only the columns we care about
        # Get all renamed columns, remove "Time [h]" since we converted it to seconds
        # Also drop status column and Timestamp as they are no longer needed
        columns_keep = list(
            set(column_renamings.values()) - {"Time [h]", "Status", "Timestamp"}
        )
        data = self.standard_data_processing(data, columns_keep=columns_keep)

        return data

    @staticmethod
    def _parse_timestamp_column(data: pl.DataFrame) -> pl.DataFrame:
        """
        Parse Timestamp column and compute Time [s] if needed.

        Parameters
        ----------
        data : pl.DataFrame
            Input dataframe with potential "Timestamp" column.

        Returns
        -------
        pl.DataFrame
            Dataframe with parsed timestamps and computed Time [s] if applicable.
        """
        if "Timestamp" not in data.columns:
            return data

        # Parse datetime with multiple format attempts
        data = data.with_columns(
            pl.coalesce(
                # Try MM/DD/YYYY HH:MM:SS format (common for Maccor DPT)
                pl.col("Timestamp").str.strptime(
                    pl.Datetime, format="%m/%d/%Y %H:%M:%S", strict=False
                ),
                # Try YYYY-MM-DD HH:MM:SS format
                pl.col("Timestamp").str.strptime(
                    pl.Datetime, format="%Y-%m-%d %H:%M:%S", strict=False
                ),
                # Try MM/DD/YYYY HH:MM:SS AM/PM format
                pl.col("Timestamp").str.strptime(
                    pl.Datetime, format="%m/%d/%Y %I:%M:%S %p", strict=False
                ),
            )
            .dt.replace_time_zone("UTC")
            .alias("Timestamp")
        )

        # If we don't have a numeric "Time [s]" column, compute it from Timestamp
        if "Time [s]" not in data.columns:
            # Compute Time [s] from earliest Timestamp
            start_epoch = data.select(pl.col("Timestamp").dt.epoch("s").min()).item()
            data = data.with_columns(
                (pl.col("Timestamp").dt.epoch("s") - start_epoch).alias("Time [s]")
            )

        return data

    def _validate_and_fix_time(
        self, data: pl.DataFrame, time_offset_fix: float
    ) -> pl.DataFrame:
        """
        Validate that time is strictly increasing and optionally fix it.

        Parameters
        ----------
        data : pl.DataFrame
            Input dataframe with "Time [s]" column.
        time_offset_fix : float
            Minimum time difference to enforce when fixing.
            If -1, raises ValueError. If >= 0, ensures all time differences are at least this value.

        Returns
        -------
        pl.DataFrame
            Dataframe with validated or fixed time.

        Raises
        ------
        ValueError
            If time is not strictly increasing and time_offset_fix is -1.
        """
        if "Time [s]" not in data.columns:
            return data

        # Vectorized check: compute differences between consecutive times
        time_col = data["Time [s]"]
        time_diff = time_col.diff()  # time[i] - time[i-1]

        # Check if any difference is < 0 (decreasing, excluding first row which is null)
        # Note: duplicates (diff == 0) are allowed and will be handled by standard_data_processing
        has_decreasing = (time_diff.tail(-1) < 0).any()

        if not has_decreasing:
            return data

        if time_offset_fix == -1:
            # Find first problematic index for error message (only for decreasing times)
            bad_mask = time_diff < 0
            bad_indices = data.with_row_index().filter(bad_mask)["index"].to_list()
            i = bad_indices[0]
            time_values = time_col.to_list()

            raise ValueError(
                f"Time [s] must be strictly increasing. "
                f"Found Time[{i - 1}] = {time_values[i - 1]:.6f}s > Time[{i}] = {time_values[i]:.6f}s. "
                f"Set options['time_offset_fix'] to a positive offset (in seconds) to automatically fix."
            )

        # Apply offset fix: ensure all negative differences are at least time_offset_fix
        # Only fix decreasing times (negative diff), leave duplicates (zero diff) alone
        # Efficient vectorized approach using numpy
        import numpy as np

        time_values = time_col.to_numpy()

        # Compute differences between consecutive time points
        time_diff_np = np.diff(time_values)

        # Only fix negative differences (decreasing times), leave duplicates and positive diffs alone
        fixed_diff = np.where(time_diff_np < 0, time_offset_fix, time_diff_np)

        # Reconstruct time series: start at first point, then add cumulative fixed differences
        fixed_time = np.concatenate(
            [[time_values[0]], time_values[0] + np.cumsum(fixed_diff)]
        )

        return data.with_columns(pl.Series("Time [s]", fixed_time))

    def _fix_unsigned_current(self, data: pl.DataFrame) -> pl.DataFrame:
        """
        Fix unsigned current by flipping sign during charge if needed.

        If both "D" (discharge) and "C" (charge) are in the "Status" column
        and the current is always positive, then the current isn't signed,
        so we need to flip it during charge.

        Parameters
        ----------
        data : pl.DataFrame
            Input dataframe with potential "Status" and "Current [A]" columns.

        Returns
        -------
        pl.DataFrame
            Dataframe with current sign corrected if needed.
        """
        if "Status" not in data.columns or "Current [A]" not in data.columns:
            return data

        statuses = set(data.select(pl.col("Status").unique()).to_series().to_list())
        if "D" not in statuses or "C" not in statuses:
            return data

        # Ensure numeric current
        data = self._coerce_numeric(data, "Current [A]")

        c_min = (
            data.filter(pl.col("Status") == "C")
            .select(pl.col("Current [A]").min())
            .item()
        )
        d_min = (
            data.filter(pl.col("Status") == "D")
            .select(pl.col("Current [A]").min())
            .item()
        )

        # If both charge and discharge currents are positive, current is unsigned
        if c_min is not None and d_min is not None and c_min >= 0 and d_min >= 0:
            data = data.with_columns(
                pl.when(pl.col("Status") == "C")
                .then(-pl.col("Current [A]"))
                .otherwise(pl.col("Current [A]"))
                .alias("Current [A]")
            )

        return data

    @staticmethod
    def _get_column_renamings() -> dict[str, str]:
        """
        Get standard column renaming mappings for Maccor files.

        Returns
        -------
        dict[str, str]
            Dictionary mapping original column names to standardized names.
        """
        return {
            "Voltage": "Voltage [V]",
            "Volts": "Voltage [V]",
            "Voltage (V)": "Voltage [V]",
            "Current": "Current [A]",
            "Amps": "Current [A]",
            "Current (A)": "Current [A]",
            "Prog Time": "Time [s]",
            "Test (Sec)": "Time [s]",
            "Test Time (sec)": "Time [s]",
            "Test Time (Hr)": "Time [h]",
            "Cycle": "Cycle from cycler",
            "Cyc#": "Cycle from cycler",
            "Cycle ID": "Cycle from cycler",
            "Cycle P": "Cycle from cycler",
            "Step": "Step from cycler",
            "Step ID": "Step from cycler",
            "LogTemp001": "Temperature [degC]",
            "Temperature (°C)": "Temperature [degC]",
            "Status": "Status",
            "State": "Status",
            "MD": "Status",
            "Capacity (Ah)": "Capacity [A.h]",
            "Capacity (AHr)": "Capacity [A.h]",
            "Cap. (Ah)": "Capacity [A.h]",
            "Energy (Wh)": "Energy [W.h]",
            "Energy (WHr)": "Energy [W.h]",
            "Chg Capacity (Ah)": "Charge capacity [A.h]",
            "Chg Capacity (AHr)": "Charge capacity [A.h]",
            "DChg Capacity (Ah)": "Discharge capacity [A.h]",
            "DChg Capacity (AHr)": "Discharge capacity [A.h]",
            "Chg Energy (Wh)": "Charge energy [W.h]",
            "Chg Energy (WHr)": "Charge energy [W.h]",
            "DChg Energy (Wh)": "Discharge energy [W.h]",
            "DChg Energy (WHr)": "Discharge energy [W.h]",
            "DPT": "Timestamp",
        }

    @staticmethod
    def _parse_excel_duration(duration_str: str) -> float | None:
        """
        Parse Excel duration format :D:HH:MM:SS to total seconds.

        Parameters
        ----------
        duration_str : str
            Duration string in format ":D:HH:MM:SS"

        Returns
        -------
        float | None
            Total seconds, or None if parsing fails.
        """
        if not duration_str.startswith(":"):
            return None
        parts = duration_str[1:].split(":")
        if len(parts) != 4:
            return None
        try:
            days, hours, minutes, seconds = map(int, parts)
            return float(days * 86400 + hours * 3600 + minutes * 60 + seconds)
        except (ValueError, TypeError):
            return None

    def _process_test_time_column(
        self, data: pl.DataFrame, column_renamings: dict[str, str]
    ) -> tuple[pl.DataFrame, dict[str, str]]:
        """
        Process "Test Time" column and determine its format.

        Handles three formats:
        1. Excel duration (":D:HH:MM:SS") -> converts to seconds
        2. Datetime strings (contains "/" or "-") -> maps to Timestamp
        3. Numeric values -> leaves as-is

        Parameters
        ----------
        data : pl.DataFrame
            Input dataframe with potential "Test Time" column.
        column_renamings : dict[str, str]
            Column renaming dictionary to update.

        Returns
        -------
        tuple[pl.DataFrame, dict[str, str]]
            Updated dataframe and column_renamings dict.
        """
        if "Test Time" not in data.columns:
            return data, column_renamings

        # Sample first non-null value to determine type
        sample = (
            data.select(pl.col("Test Time"))
            .filter(pl.col("Test Time").is_not_null())
            .head(1)
        )

        if sample.height == 0:
            return data, column_renamings

        val = sample.item(0, 0)

        # If it looks like Excel time duration (starts with ":"), parse it
        if isinstance(val, str) and val.startswith(":"):
            data = data.with_columns(
                pl.col("Test Time")
                .map_elements(self._parse_excel_duration, return_dtype=pl.Float64)
                .alias("Test Time")
            )
            column_renamings["Test Time"] = "Time [s]"
        # If it looks like a datetime string (contains "/" or "-"), map to Timestamp
        elif isinstance(val, str) and ("/" in val or "-" in val):
            column_renamings["Test Time"] = "Timestamp"
        # Otherwise treat as numeric time column (might already be in seconds)

        return data, column_renamings

    def _read_excel_file(self, filename: str | Path, encoding: str) -> pl.DataFrame:
        """
        Read Maccor data from an Excel file (.xls or .xlsx).

        Parameters
        ----------
        filename : str | Path
            Path to the Excel file.
        encoding : str
            File encoding (not used for Excel but kept for consistency).

        Returns
        -------
        pl.DataFrame
            Raw data from Excel file with header row identified.
        """
        # Read Excel file - first row is always the header
        xl_reader = fastexcel.read_excel(filename)
        sheet_names = xl_reader.sheet_names

        # Read the first sheet, assuming first row is header
        # Suppress pandas dtype warning when reading Excel (printed to stderr)
        with suppress_excel_dtype_warnings():
            data, _ = read_excel_and_get_column_names(
                filename, sheet_name=sheet_names[0]
            )

        return data

    def read_header(
        self, filename: str | Path, options: dict[str, str] | None = None
    ) -> str:
        """
        Read the header from a Maccor file.
        """
        options = iwutil.check_and_combine_options(self.default_options, options)
        encoding, skiprows, _, _, _, is_excel = self._get_file_args(filename, options)

        if is_excel:
            # For Excel files, first row is always the header
            xl_reader = fastexcel.read_excel(filename)
            sheet_names = xl_reader.sheet_names
            # Suppress pandas dtype warning when reading Excel (printed to stderr)
            with suppress_excel_dtype_warnings():
                df_raw = pl.read_excel(filename, sheet_name=sheet_names[0])

            # Return header row as string (column names are the header)
            return "\t".join(str(col) for col in df_raw.columns)
        else:
            with open(filename, encoding=encoding) as f:
                if len(skiprows) == 1:
                    # Header is single line
                    header_text = f.readline()
                else:
                    # Header is multiple lines
                    header_text = "".join(f.readline() for _ in skiprows)
            return header_text

    def read_start_time(
        self,
        filename: str | Path,
        extra_column_mappings: dict[str, str] | None = None,
        options: dict[str, str] | None = None,
    ) -> datetime | None:
        """
        Read the start time from a Maccor file.

        Parameters
        ----------
        filename : str | Path
            Path to the Maccor file to be read. Supports:
            - .txt files (tab-separated)
            - .csv files (comma-separated with units row)
            - .xls/.xlsx files (Excel format)
            - Files with .+3digits extension (e.g., .123, .456)
        options : dict of str to str, optional
            See :func:`ionworksdata.read.Maccor.run`.

        Returns
        -------
        datetime | None
            The start time of the Maccor file, or None if not found.
        """
        options = iwutil.check_and_combine_options(self.default_options, options)

        # Load the header row
        start_datetime = None
        header_text = self.read_header(filename, options)

        # Flatten header to single line by replacing newlines with spaces
        header_text = " ".join(header_text.split())
        if "Date of Test:" in header_text:
            # Date always comes after "Date of Test:" and before "Filename:"
            date_str = (
                header_text.split("Date of Test:")[1].split("Filename:")[0].strip()
            )
            # Try some common date formats
            for fmt in [
                "%d %B %Y, %I:%M:%S %p",
                "%m/%d/%Y",
            ]:
                try:
                    if fmt == "%m/%d/%Y":
                        # if only date is present, assume time is 00:00:00
                        # maybe unsafe for multiple tests on the same day?
                        date_str = date_str + " 00:00:00"
                        fmt = "%m/%d/%Y %H:%M:%S"
                    start_datetime = datetime.strptime(date_str, fmt)
                    break
                except ValueError:
                    continue

        if start_datetime:
            timezone = options.get("timezone", "UTC")
            if isinstance(timezone, str):
                timezone = pytz.timezone(timezone)
            else:
                raise ValueError(f"Invalid timezone: {timezone}")
            start_datetime = start_datetime.replace(tzinfo=timezone)
            return iwdata.util.check_and_convert_datetime(start_datetime)
        else:
            return None


def maccor(
    filename: str | Path,
    extra_column_mappings: dict[str, str] | None = None,
    options: dict[str, str] | None = None,
) -> pl.DataFrame:
    return Maccor().run(
        filename, extra_column_mappings=extra_column_mappings, options=options
    )
