from __future__ import annotations

import csv as csv_py
from datetime import datetime
from pathlib import Path
from typing import Any

import iwutil
import polars as pl
import pytz

import ionworksdata as iwdata

from .read import BaseReader


def _try_int_or_nan(x: str) -> int | None:
    try:
        return int(x)
    except ValueError:
        return None


class Repower(BaseReader):
    name: str = "Repower"
    default_options: dict[str, Any] = {
        "file_encoding": "latin1",
        "cell_metadata": "[required]",
        "timezone": "UTC",
    }

    def run(
        self,
        filename: str | Path,
        extra_column_mappings: dict[str, str] | None = None,
        options: dict[str, str] | None = None,
    ) -> pl.DataFrame:
        """
        Read and process data from a Repower file. The following column mappings are applied by default:

            - "Voltage(V)" -> "Voltage [V]"
            - "Current(A)" -> "Current [A]"
            - "Cycle ID" -> "Cycle from cycler"
            - "Step ID" -> "Step from cycler"
            - "Step State" -> "Status"
            - "System Time" -> "System time"

        Time columns can be called Relative Time(Sec) or Relative Time(Hour). The time unit is automatically detected and the time column is renamed accordingly.

        Additional column mappings can be provided via the extra_column_mappings parameter.

        Parameters
        ----------
        filename : str | Path
            Path to the Repower file to be read.
        extra_column_mappings : dict[str, str] | None, optional
            Dictionary of additional column mappings to use when reading the Repower file.
            The keys are the original column names and the values are the new column
            names. Default is None.
        options : dict[str, str] | None, optional
            Dictionary of options to use when reading the Repower file.

            Options are:

            - file_encoding: str, optional
                Encoding format for the Repower file. Default is "latin1".
            - cell_metadata: dict, required
                Dictionary containing metadata about the cell, including voltage cutoffs.
            - timezone: str, optional
                Timezone for timestamps. Default is "UTC".

        Returns
        -------
        polars.DataFrame
            Processed data from the Repower file with standardized column names and units.

        Notes
        -----
        This function reads a Repower file, processes the data, and returns a DataFrame
        with standardized column names and units. It handles data cleaning, formatting,
        and time adjustments based on the specific structure of Repower files.
        """
        options = iwutil.check_and_combine_options(self.default_options, options)
        cell_metadata = options["cell_metadata"]

        # define renamings
        renamings = {
            "Voltage(V)": "Voltage [V]",
            "Current(A)": "Current [A]",
            "Cycle ID": "Cycle from cycler",
            "Step ID": "Step from cycler",
            "Step State": "Status",
            "System Time": "System time",
            "Capacity(Ah)": "Capacity [A.h]",
            "Energy(Wh)": "Energy [W.h]",
            "Charge Capacity(Ah)": "Charge capacity [A.h]",
            "Discharge Capacity(Ah)": "Discharge capacity [A.h]",
            "Charge Energy(Wh)": "Charge energy [W.h]",
            "Discharge Energy(Wh)": "Discharge energy [W.h]",
        }
        renamings.update(extra_column_mappings or {})

        # Read the columns and find the columns with names that can change
        encoding = options["file_encoding"]
        with open(filename, encoding=encoding) as f:
            csv_reader = csv_py.reader(f)
            columns = next(csv_reader)
        # time columns can be called Relative Time(Sec) or Relative Time(Hour)
        time_col = [col for col in columns if col.startswith("Relative Time")]
        if len(time_col) == 1:
            time_col = time_col[0]
            if time_col == "Relative Time(Sec)":
                renamings[time_col] = "Relative time [s]"
                time_unit = "seconds"
            elif time_col == "Relative Time(Hour)":
                renamings[time_col] = "Relative time [h]"
                time_unit = "hours"
            else:
                raise ValueError("Unrecognized unit of time")
        elif len(time_col) == 0:
            raise ValueError("No time column found")
        else:
            raise ValueError("Multiple time columns found")

        # temperature columns start with MTV
        mtv_col = [col for col in columns if col.startswith("MTV")]
        if len(mtv_col) == 1:
            mtv_col = mtv_col[0]
            renamings[mtv_col] = "Temperature [degC]"

        columns_to_keep = list(renamings.keys())

        # Force numeric columns to be read as Float64 to avoid type inference issues
        # where initial integer-like values (e.g., "0") cause the column to be read as Int64,
        # truncating subsequent decimal values
        schema_overrides = {
            "Voltage(V)": pl.Float64,
            "Current(A)": pl.Float64,
            "Capacity(Ah)": pl.Float64,
            "Energy(Wh)": pl.Float64,
            "Charge Capacity(Ah)": pl.Float64,
            "Discharge Capacity(Ah)": pl.Float64,
            "Charge Energy(Wh)": pl.Float64,
            "Discharge Energy(Wh)": pl.Float64,
        }
        if time_col in columns_to_keep:
            schema_overrides[time_col] = pl.Float64

        # Read CSV with Polars
        data = pl.read_csv(
            filename,
            encoding=encoding,
            columns=columns_to_keep,
            ignore_errors=True,
            truncate_ragged_lines=True,
            schema_overrides=schema_overrides,
        )

        # Convert "Cycle ID" column to int, handling errors
        if "Cycle ID" in data.columns:
            data = data.with_columns(pl.col("Cycle ID").cast(pl.Int64, strict=False))

        iwdata.util.check_for_duplicates(renamings, data)

        # Remove rows where all values are null
        data = data.filter(~pl.all_horizontal(pl.all().is_null()))

        # Remove rows that got messed up as evidenced by NaNs in the last column
        last_col = data.columns[-1]
        first_null_idx = data.select(pl.col(last_col).is_null().arg_max()).item()
        if first_null_idx is not None and data[last_col][first_null_idx] is None:
            data = data.slice(0, first_null_idx)

        # Convert system time to seconds
        t = pl.col("System Time").str.strptime(pl.Datetime, format=" %Y-%m-%d %H:%M:%S")
        data = data.with_columns(t.alias("__parsed_time"))
        first_time = data.select(pl.col("__parsed_time").first()).item()
        data = data.with_columns(
            ((pl.col("__parsed_time") - first_time).dt.total_seconds()).alias(
                "System time [s]"
            )
        ).drop("__parsed_time")

        # Remove any points where time is negative
        data = data.filter(pl.col("System time [s]") >= 0)

        # Remove any points after the point where the time difference is greater than 2 hours
        data = data.with_columns(
            pl.col("System time [s]").diff().fill_null(0).alias("__time_diff")
        )
        first_large_diff_idx = data.select(
            (pl.col("__time_diff") > 7200).arg_max()
        ).item()
        if first_large_diff_idx is not None and first_large_diff_idx > 0:
            # Check if there actually is a large diff
            has_large_diff = data.filter(pl.col("__time_diff") > 7200).height > 0
            if has_large_diff:
                data = data.slice(0, first_large_diff_idx)
        data = data.drop("__time_diff")

        data = data.rename(renamings)

        # Make sure the temperature column exists
        if "Temperature [degC]" not in data.columns:
            data = data.with_columns(
                pl.lit(None).cast(pl.Float64).alias("Temperature [degC]")
            )

        # remove any voltage outliers
        V_min = cell_metadata["Lower voltage cut-off [V]"]
        V_max = cell_metadata["Upper voltage cut-off [V]"]
        data = data.filter(
            (pl.col("Voltage [V]") < V_max * (1 + 1e-3))
            & (pl.col("Voltage [V]") > V_min * (1 - 1e-3))
        )

        if time_unit == "seconds":
            # any negative diffs greater than 1s indicate a reset of the relative time
            # call this a 1 second step
            data = data.with_columns(
                pl.col("Relative time [s]").diff().fill_null(0).alias("__time_diff")
            )
            data = data.with_columns(
                pl.when(pl.col("__time_diff") < -1)
                .then(pl.lit(1.0))
                .when(pl.col("__time_diff") < 0.1)
                .then(pl.lit(0.1))
                .otherwise(pl.col("__time_diff"))
                .alias("__time_diff")
            )
            data = data.with_columns(
                pl.col("__time_diff").cum_sum().alias("Time [s]")
            ).drop("__time_diff")

            # only keep rows where the relative time is an exact multiple of 1 second
            # this captures the exact start of each step but skips the weird offset points
            # within the first second
            data = data.with_columns(
                pl.col("Relative time [s]")
                .floor()
                .alias("Relative time first second [s]")
            )

            # Build aggregation expressions
            columns = data.columns
            agg_exprs = []
            for col in columns:
                if col not in ["Step from cycler", "Relative time first second [s]"]:
                    if col in ["Current [A]", "Voltage [V]"]:
                        # Take the value from the last point in the second because this cycler has a
                        # delay in recording the current and voltage
                        agg_exprs.append(pl.col(col).last().alias(col))
                    else:
                        agg_exprs.append(pl.col(col).first().alias(col))

            data = (
                data.group_by(["Step from cycler", "Relative time first second [s]"])
                .agg(agg_exprs)
                .sort(["Step from cycler", "Relative time first second [s]"])
            )

        elif time_unit == "hours":
            # we don't have enough granularity in the time data to use the relative time
            # make sure time diff is always non-negative
            data = data.with_columns(
                pl.col("System time [s]").diff().fill_null(0).alias("__time_diff")
            )
            data = data.with_columns(
                pl.when(pl.col("__time_diff") < 0)
                .then(pl.lit(0.0))
                .otherwise(pl.col("__time_diff"))
                .alias("__time_diff")
            )
            data = data.with_columns(
                pl.col("__time_diff").cum_sum().alias("Time [s]")
            ).drop("__time_diff")

            # Build aggregation expressions
            columns = data.columns
            agg_exprs = []
            for col in columns:
                if col not in ["Step from cycler", "System time [s]"]:
                    if col in ["Current [A]", "Voltage [V]"]:
                        # Take the value from the last point in the second because this cycler has a
                        # delay in recording the current and voltage
                        agg_exprs.append(pl.col(col).last().alias(col))
                    else:
                        agg_exprs.append(pl.col(col).first().alias(col))

            data = (
                data.group_by(["Step from cycler", "System time [s]"])
                .agg(agg_exprs)
                .sort(["Step from cycler", "System time [s]"])
            )

        # keep only the columns we want
        columns_keep = [
            "Time [s]",
            "Voltage [V]",
            "Current [A]",
            "Temperature [degC]",
            "Status",
            "Step from cycler",
            "Cycle from cycler",
            "System time",
        ]
        if extra_column_mappings:
            columns_keep.extend(extra_column_mappings.values())

        data = self.standard_data_processing(data, columns_keep=columns_keep)

        return data

    def read_start_time(
        self,
        filename: str | Path,
        extra_column_mappings: dict[str, str] | None = None,
        options: dict[str, str] | None = None,
    ) -> datetime:
        """
        Read the start time from a Repower file.

        Parameters
        ----------
        filename : str | Path
            Path to the Repower file to be read.
        options : dict[str, str] | None, optional
            Options for reading the file. See :func:`ionworksdata.read.Repower.run`.

        Returns
        -------
        datetime
            The start time of the Repower file.
        """
        options = iwutil.check_and_combine_options(self.default_options, options)
        data = pl.read_csv(
            filename,
            encoding=options["file_encoding"],
            ignore_errors=True,
            truncate_ragged_lines=True,
        )

        # Convert "Cycle ID" column to int, handling errors
        if "Cycle ID" in data.columns:
            data = data.with_columns(pl.col("Cycle ID").cast(pl.Int64, strict=False))

        # Parse system time and find minimum
        system_time = pl.col("System Time").str.strptime(
            pl.Datetime, format=" %Y-%m-%d %H:%M:%S"
        )
        start_datetime = (
            data.select(system_time.alias("parsed_time"))
            .select(pl.col("parsed_time").min())
            .item()
        )

        timezone = options.get("timezone", "UTC")
        if isinstance(timezone, str):
            timezone = pytz.timezone(timezone)
        else:
            raise ValueError(f"Invalid timezone: {timezone}")
        start_datetime = start_datetime.replace(tzinfo=timezone)
        start_datetime = iwdata.util.check_and_convert_datetime(start_datetime)
        return start_datetime


def repower(
    filename: str | Path,
    extra_column_mappings: dict[str, str] | None = None,
    options: dict[str, str] | None = None,
) -> pl.DataFrame:
    return Repower().run(
        filename, extra_column_mappings=extra_column_mappings, options=options
    )
