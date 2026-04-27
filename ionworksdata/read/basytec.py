"""Reader for BaSyTec battery cycler CSV exports.

BaSyTec cyclers (CTS, X50) are commonly used in European academic battery labs.
The CSV format has no preamble — column names on the first line, data rows below.

File format
-----------
Columns: ``run_time`` (HH:MM:SS.sss, hours may exceed 24), ``c_vol`` (V),
``c_cur`` (A, **negative during discharge**), ``c_surf_temp`` (degC),
``amb_temp`` (degC, often NaN), ``step_type`` (int).

Current sign convention
-----------------------
BaSyTec uses negative current = discharge, opposite to the ionworks convention
(positive = discharge). The reader flips the sign on import.

Companion metadata files
------------------------
Each CSV may have a ``_meta.txt`` sibling (e.g. ``stroebl_CU_meta.txt`` for
``stroebl_CU.csv``) containing key-value metadata above a ``---`` separator.
The reader extracts ``Measurement start date`` (DD.MM.YYYY) for ``start_time``.

Multi-file per cell
-------------------
A single cell may produce multiple CSVs for different test phases (ET = entry
test, CU = checkup, exCU = extended checkup, AT = aging test). This reader
handles individual files; multi-file concatenation is the caller's
responsibility.

Reference dataset: Stroebl et al. 2024 "Multi-Stage Lithium Ion Battery Aging
Study" (https://doi.org/10.1038/s41597-024-03859-z).
"""

# pyright: reportMissingTypeStubs=false
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, cast

import iwutil  # type: ignore[reportMissingTypeStubs]
import polars as pl
import pytz  # type: ignore[reportMissingTypeStubs]

import ionworksdata as iwdata

from .read import BaseReader


class Basytec(BaseReader):
    name: str = "Basytec"
    default_options: dict[str, Any] = {
        "timezone": "UTC",
        "cell_metadata": {},
    }

    @staticmethod
    def _parse_run_time_column(data: pl.DataFrame) -> pl.DataFrame:
        """Convert the ``run_time`` column from ``HH:MM:SS.sss`` to seconds.

        Hours may exceed 24 (e.g. ``205:06:00.397``), so standard datetime
        parsing cannot be used.

        Parameters
        ----------
        data : pl.DataFrame
            DataFrame containing a ``run_time`` string column.

        Returns
        -------
        pl.DataFrame
            DataFrame with ``Time [s]`` replacing the ``run_time`` column.
        """
        parts = data["run_time"].str.strip_chars().str.split(":")
        hours = parts.list.get(0).cast(pl.Float64)
        minutes = parts.list.get(1).cast(pl.Float64)
        seconds = parts.list.get(2).cast(pl.Float64)
        time_s = hours * 3600.0 + minutes * 60.0 + seconds
        return data.with_columns(time_s.alias("Time [s]")).drop("run_time")

    @staticmethod
    def _find_meta_file(filename: str | Path) -> Path | None:
        """Return the companion ``_meta.txt`` path if it exists.

        For a file named ``stroebl_CU.csv`` the metadata file is
        ``stroebl_CU_meta.txt`` in the same directory.

        Parameters
        ----------
        filename : str | Path
            Path to the BaSyTec CSV file.

        Returns
        -------
        Path | None
            Path to the metadata file, or None if not found.
        """
        p = Path(filename)
        meta_file = p.with_name(p.stem + "_meta.txt")
        return meta_file if meta_file.exists() else None

    @staticmethod
    def _read_meta_start_date(meta_path: Path) -> datetime | None:
        """Parse ``Measurement start date`` from a BaSyTec metadata file.

        The date format is ``DD.MM.YYYY``.

        Parameters
        ----------
        meta_path : Path
            Path to the ``_meta.txt`` file.

        Returns
        -------
        datetime | None
            Parsed date as a naive datetime (midnight), or None if not found.
        """
        with open(meta_path, encoding="utf-8") as f:
            for line in f:
                if line.startswith("Measurement start date:"):
                    date_str = line.split(":", 1)[1].strip()
                    try:
                        return datetime.strptime(date_str, "%d.%m.%Y")
                    except ValueError:
                        return None
        return None

    def run(
        self,
        filename: str | Path,
        extra_column_mappings: dict[str, str] | None = None,
        options: dict[str, str] | None = None,
    ) -> pl.DataFrame:
        """Read a BaSyTec CSV and return a DataFrame with standardized columns.

        Parameters
        ----------
        filename : str | Path
            Path to the BaSyTec CSV file.
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
        pl.DataFrame
            Time series with columns mapped to:
            - "Time [s]"
            - "Voltage [V]"
            - "Current [A]"
            - "Temperature [degC]" (if available)
            - "Step from cycler" (if available)
        """
        options = iwutil.check_and_combine_options(self.default_options, options)

        schema_overrides = {
            "c_vol": pl.Float64,
            "c_cur": pl.Float64,
            "c_surf_temp": pl.Float64,
            "amb_temp": pl.Float64,
            "step_type": pl.Float64,
            "run_time": pl.String,
        }

        df = pl.read_csv(
            filename,
            schema_overrides=schema_overrides,
            null_values=["NaN", "nan"],
            truncate_ragged_lines=True,
        )

        # Parse run_time HH:MM:SS.sss → Time [s]
        df = self._parse_run_time_column(df)

        # Column mappings
        column_renamings = {
            "c_vol": "Voltage [V]",
            "c_cur": "Current [A]",
            "c_surf_temp": "Temperature [degC]",
            "step_type": "Step from cycler",
        }
        column_renamings.update(extra_column_mappings or {})
        iwdata.util.check_for_duplicates(column_renamings, df)
        present_map = {k: v for k, v in column_renamings.items() if k in df.columns}
        if present_map:
            df = df.rename(present_map)

        # Flip current sign: BaSyTec uses negative=discharge, ionworks uses
        # positive=discharge
        df = df.with_columns((-pl.col("Current [A]")).alias("Current [A]"))

        columns_keep = [
            col
            for col in [
                "Time [s]",
                "Voltage [V]",
                "Current [A]",
                "Temperature [degC]",
                "Step from cycler",
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
        """Read the test start time from the companion BaSyTec metadata file.

        Parameters
        ----------
        filename : str | Path
            Path to the BaSyTec CSV file.
        extra_column_mappings : dict[str, str] | None, optional
            Unused, present for API compatibility.
        options : dict[str, str] | None, optional
            Options containing the timezone string (default "UTC").

        Returns
        -------
        datetime | None
            The timezone-aware start time, or None if no metadata file or date found.
        """
        opts = cast(
            dict[str, Any],
            iwutil.check_and_combine_options(self.default_options, options),
        )
        meta_path = self._find_meta_file(filename)
        if meta_path is None:
            return None

        start_datetime = self._read_meta_start_date(meta_path)
        if start_datetime is None:
            return None

        timezone = opts.get("timezone", "UTC")
        if isinstance(timezone, str):
            timezone = pytz.timezone(timezone)
        else:
            raise ValueError(f"Invalid timezone: {timezone}")
        start_datetime = timezone.localize(start_datetime)
        start_datetime = iwdata.util.check_and_convert_datetime(
            cast(datetime, start_datetime)
        )
        return start_datetime


def basytec(
    filename: str | Path,
    extra_column_mappings: dict[str, str] | None = None,
    options: dict[str, str] | None = None,
) -> pl.DataFrame:
    return Basytec().run(
        filename, extra_column_mappings=extra_column_mappings, options=options
    )
