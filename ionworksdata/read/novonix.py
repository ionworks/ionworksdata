# pyright: reportMissingTypeStubs=false
from .read import BaseReader
from pathlib import Path
import ionworksdata as iwdata
import iwutil  # type: ignore[reportMissingTypeStubs]
import pytz  # type: ignore[reportMissingTypeStubs]
from datetime import datetime
from typing import Any, cast
import polars as pl


class Novonix(BaseReader):
    name: str = "Novonix"
    default_options: dict[str, Any] = {
        "timezone": "UTC",
        "cell_metadata": {},
    }

    @staticmethod
    def _get_header_row(filename: str | Path) -> int:
        """
        Find the header row index for the data table.

        Returns the 0-based line index of the header that starts with
        "Date and Time".
        """
        with open(filename, encoding="utf-8") as f:
            for i, row in enumerate(f):
                if row.strip().startswith("Date and Time"):
                    return i
        raise ValueError("Could not find data header row in Novonix file")

    @staticmethod
    def _read_summary_started(filename: str | Path) -> datetime | None:
        """
        Read the Started timestamp from the [Summary] section if present.
        """
        with open(filename, encoding="utf-8") as f:
            for row in f:
                row = row.strip()
                if row.startswith("Started:"):
                    # Example: Started: 2023-06-14 5:22:45 PM
                    dt_str = row.split("Started:", 1)[1].strip()
                    # Try common Novonix format
                    for fmt in [
                        "%Y-%m-%d %I:%M:%S %p",
                        "%Y-%m-%d %H:%M:%S",
                    ]:
                        try:
                            return datetime.strptime(dt_str, fmt)
                        except ValueError:
                            continue
                    return None
        return None

    def run(
        self,
        filename: str | Path,
        extra_column_mappings: dict[str, str] | None = None,
        options: dict[str, str] | None = None,
    ) -> pl.DataFrame:
        """
        Read a NOVONIX CSV and return a DataFrame with standardized columns.

        Parameters
        ----------
        filename : str | Path
            Path to the NOVONIX CSV file to be read.
        extra_column_mappings : dict[str, str] | None, optional
            Additional column mappings to apply after initial normalization.
        options : dict[str, str] | None, optional
            Options are:

                - timezone: str, optional
                    Timezone for timestamps if needed. Default is "UTC".
                - cell_metadata: dict, optional
                    Additional metadata about the cell.

        Returns
        -------
        pandas.DataFrame
            Time series with columns mapped to:
            - "Time [s]"
            - "Voltage [V]"
            - "Current [A]"
            - "Temperature [degC]" (if available)
            - "Step from cycler" (if available)
            - "Cycle from cycler" (if available)
        """
        options = iwutil.check_and_combine_options(self.default_options, options)
        header_idx = self._get_header_row(filename)

        # Force numeric columns to be read as Float64 to avoid type inference issues
        # where initial integer-like values (e.g., "0") cause the column to be read
        # as Int64, truncating subsequent decimal values
        schema_overrides = {
            "Potential (V)": pl.Float64,
            "Current (A)": pl.Float64,
            "Run Time (h)": pl.Float64,
            "Temperature (°C)": pl.Float64,
        }

        # Read data table with Polars
        df = pl.read_csv(
            filename,
            skip_rows=header_idx,
            truncate_ragged_lines=True,
            schema_overrides=schema_overrides,
        )

        # Standard renamings
        column_renamings = {
            "Potential (V)": "Voltage [V]",
            "Current (A)": "Current [A]",
            "Run Time (h)": "Time [h]",
            "Temperature (°C)": "Temperature [degC]",
            "Cycle Number": "Cycle from cycler",
            "Step Number": "Step from cycler",
        }
        column_renamings.update(extra_column_mappings or {})
        # Validate duplicate mappings (check_for_duplicates works with Polars)
        iwdata.util.check_for_duplicates(column_renamings, df)
        # Only rename columns that exist to avoid errors
        present_map = {k: v for k, v in column_renamings.items() if k in df.columns}
        if present_map:
            df = df.rename(present_map)

        # Time column
        if "Time [h]" in df.columns:
            df = df.with_columns((pl.col("Time [h]") * 3600.0).alias("Time [s]"))
        elif "Date and Time" in df.columns:
            # Try parsing with multiple formats (Novonix uses various datetime formats)
            # Use coalesce to try 12-hour format first, then 24-hour format
            df = df.with_columns(
                pl.coalesce(
                    pl.col("Date and Time").str.strptime(
                        pl.Datetime, format="%Y-%m-%d %I:%M:%S %p", strict=False
                    ),
                    pl.col("Date and Time").str.strptime(
                        pl.Datetime, format="%Y-%m-%d %H:%M:%S", strict=False
                    ),
                ).alias("__dt__")
            )
            # Compute seconds from start
            df = df.with_columns(
                (
                    pl.col("__dt__").dt.epoch("s")
                    - pl.col("__dt__").dt.epoch("s").min()
                ).alias("Time [s]")
            ).drop("__dt__")
        else:
            raise ValueError(
                "Novonix file must contain 'Run Time (h)' or 'Date and Time'"
            )

        # Keep/compute only the relevant columns
        columns_keep = [
            col
            for col in [
                "Time [s]",
                "Voltage [V]",
                "Current [A]",
                "Temperature [degC]",
                "Step from cycler",
                "Cycle from cycler",
            ]
            if col in df.columns
        ]

        df = self.standard_data_processing(df, columns_keep=columns_keep)
        return df

    def read_start_time(
        self,
        filename: str | Path,
        extra_column_mappings: dict[str, str] | None = None,
        options: dict[str, str] | None = None,
    ):
        """
        Read the test start time from the NOVONIX file summary.

        Parameters
        ----------
        filename : str | Path
            Path to the NOVONIX CSV file to be read.
        options : dict[str, str] | None, optional
            Options containing the timezone string (default "UTC").

        Returns
        -------
        datetime | None
            The timezone-aware start time, or None if not found.
        """
        opts = cast(
            dict[str, Any],
            iwutil.check_and_combine_options(self.default_options, options),
        )
        start_datetime = self._read_summary_started(filename)
        if start_datetime is None:
            return None

        timezone = opts.get("timezone", "UTC")
        if isinstance(timezone, str):
            timezone = pytz.timezone(timezone)
        else:
            raise ValueError(f"Invalid timezone: {timezone}")
        assert start_datetime is not None
        start_datetime = timezone.localize(start_datetime)
        start_datetime = iwdata.util.check_and_convert_datetime(
            cast(datetime, start_datetime)
        )
        return start_datetime


def novonix(
    filename: str | Path,
    extra_column_mappings: dict[str, str] | None = None,
    options: dict[str, str] | None = None,
) -> pl.DataFrame:
    return Novonix().run(
        filename, extra_column_mappings=extra_column_mappings, options=options
    )
