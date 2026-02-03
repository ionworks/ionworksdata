from __future__ import annotations

from .read import BaseReader
from pathlib import Path
import polars as pl
import ionworksdata as iwdata
import iwutil
from typing import Any
from datetime import datetime
import re
import fastexcel


class Neware(BaseReader):
    name: str = "Neware"
    default_options: dict[str, Any] = {
        "cell_metadata": {},
        "sheets": None,
    }

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

    def _read_file_data(
        self, filename: str | Path, sheets: dict | None = None
    ) -> pl.DataFrame:
        """Read data from CSV or Excel with Polars, optional sheet filtering."""
        filename = Path(filename)

        if filename.suffix.lower() in [".xls", ".xlsx"]:
            # Read Excel file with Polars
            if sheets is None:
                # No sheet specification - read the first sheet
                df_pl = pl.read_excel(filename)
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
            # Build schema_overrides only for columns that exist in the file
            # Read header to determine columns
            import csv

            with open(filename) as f:
                reader = csv.reader(f)
                header = next(reader)
            schema_overrides = {
                col: pl.Float64 for col in self._raw_numeric_columns if col in header
            }
            df_pl = pl.read_csv(filename, schema_overrides=schema_overrides)
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
        }
        column_renamings.update(extra_column_mappings or {})
        # Validate duplicates (check_for_duplicates works with Polars)
        iwdata.util.check_for_duplicates(column_renamings, data)
        # Only rename columns that exist to avoid errors
        present_map = {k: v for k, v in column_renamings.items() if k in data.columns}
        if present_map:
            data = data.rename(present_map)
        return data, column_renamings

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

            - "Current (mA)", "Cur(mA)", "Current(A)" -> "Current [mA]"
            - "Current (A)" -> "Current [A]"
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
            - 'sheets': dict specifying sheet selection for Excel files (.xls/.xlsx only).
              If not specified, reads the first sheet (index 0). Format:

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

        # Load data and rename columns
        data = self._read_file_data(filename, sheets)
        data, column_renamings = self._apply_column_renamings(
            data, extra_column_mappings
        )

        # Convert datetime to seconds from start (parse and add UTC timezone)
        data = data.with_columns(
            pl.col("Timestamp")
            .str.strptime(pl.Datetime, strict=False)
            .dt.replace_time_zone("UTC")
            .alias("Timestamp")
        )

        # Filter out January 1970 timestamps if they appear to be data artifacts
        data = self._filter_1970_timestamps(data)

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

        data = self._read_file_data(filename, sheets)
        data, _ = self._apply_column_renamings(data, extra_column_mappings)
        data = data.with_columns(
            pl.col("Timestamp")
            .str.strptime(pl.Datetime, strict=False)
            .dt.replace_time_zone("UTC")
            .alias("Timestamp")
        )

        # Filter out January 1970 timestamps if they appear to be data artifacts
        data = self._filter_1970_timestamps(data)

        start_timestamp = data.select(pl.col("Timestamp").min()).item()
        start_datetime = iwdata.util.check_and_convert_datetime(start_timestamp)
        return start_datetime


def neware(
    filename: str | Path,
    extra_column_mappings: dict[str, str] | None = None,
    options: dict[str, str] | None = None,
) -> pl.DataFrame:
    return Neware().run(
        filename, extra_column_mappings=extra_column_mappings, options=options
    )
