# pyright: reportMissingTypeStubs=false
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import iwutil  # type: ignore[reportMissingTypeStubs]
import polars as pl
import pytz  # type: ignore[reportMissingTypeStubs]

import ionworksdata as iwdata

from ._metadata import (
    localize,
    nest_metadata,
    parse_datetime,
    parse_labelled_lines,
    take_known_start_time,
)
from .read import BaseReader

#: Observed as exactly -9999 (no probe connected). The bound is `<=` only
#: because no real temperature is colder, not because a range is documented.
_NO_SENSOR_SENTINEL = -9998.0

#: The ``[Summary]`` block ends here; the ``[Protocol]`` block below it repeats
#: ``Protocol:`` and adds hundreds of step lines, so metadata reads stop here.
_SUMMARY_END = "[End Summary]"

#: Mass/Capacity/Area/DC Offset are omitted: they belong to the cell record,
#: not to one measurement's test setup.
_NOVONIX_SUMMARY_LABELS: dict[str, str] = {
    "protocol": "procedure",
    "channel": "channel_number",
    "cell": "cell_label",
    "cellid": "source_file_id",
    "description": "notes",
    "serial number": "cycler_serial_number",
    "version": "software_version",
    "started": "start_time",
}

#: Novonix writes both spellings depending on export version.
_NOVONIX_START_FORMATS = ("%Y-%m-%d %I:%M:%S %p", "%Y-%m-%d %H:%M:%S")

#: Recording this would put a note on every measurement whose operator left
#: the pre-filled field alone, indistinguishable from one they wrote.
_NOVONIX_DESCRIPTION_PLACEHOLDER = "enter description here..."


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
        with open(filename, encoding="latin-1") as f:
            for i, row in enumerate(f):
                if row.strip().startswith("Date and Time"):
                    return i
        raise ValueError("Could not find data header row in Novonix file")

    @staticmethod
    def _read_summary_lines(filename: str | Path) -> list[str]:
        """
        Read the ``[Summary]`` block's lines, stopping at ``[End Summary]``.

        Parameters
        ----------
        filename : str | Path
            Path to the NOVONIX CSV file.

        Returns
        -------
        list[str]
            Lines of the summary block. Reading stops at the block terminator
            rather than consuming the file, because the ``[Protocol]`` block
            below it repeats ``Protocol:`` and runs to hundreds of step lines.
        """
        lines: list[str] = []
        with open(filename, encoding="latin-1") as f:
            for row in f:
                if row.strip() == _SUMMARY_END:
                    break
                lines.append(row)
        return lines

    @classmethod
    def _read_summary_started(cls, filename: str | Path) -> datetime | None:
        """
        Read the Started timestamp from the [Summary] section if present.
        """
        summary = parse_labelled_lines(
            cls._read_summary_lines(filename), {"started": "start_time"}
        )
        started = summary.get("start_time")
        if started is None:
            return None
        return parse_datetime(started, _NOVONIX_START_FORMATS)

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
            "Circuit Temperature (°C)": pl.Float64,
            "Date and Time": pl.String,
        }

        # Read data table with Polars
        df = pl.read_csv(
            filename,
            skip_rows=header_idx,
            truncate_ragged_lines=True,
            schema_overrides=schema_overrides,
            infer_schema_length=10000,
            encoding="utf8-lossy",
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
        present_map = iwdata.util.resolve_renamings(
            column_renamings, df, priority=extra_column_mappings
        )
        if present_map:
            df = df.rename(present_map)

        # -9999 means "no thermocouple connected", not a reading. Filtered
        # before selection so the fallback cannot reintroduce it.
        present = [
            c
            for c in ("Temperature [degC]", "Circuit Temperature (°C)")
            if c in df.columns
        ]
        df = df.with_columns(
            [
                pl.when(pl.col(c) <= _NO_SENSOR_SENTINEL)
                .then(None)
                .otherwise(pl.col(c))
                .alias(c)
                for c in present
            ]
        )
        usable = next((c for c in present if df[c].null_count() < df.height), None)
        if usable != "Temperature [degC]":
            # Drop first: renaming onto a live name is a Polars DuplicateError.
            df = df.drop("Temperature [degC]", strict=False)
            if usable is not None:
                df = df.rename({usable: "Temperature [degC]"})

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

    def read_metadata(
        self,
        filename: str | Path,
        options: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """
        Read the descriptive metadata from a NOVONIX file's ``[Summary]`` block.

        Parameters
        ----------
        filename : str | Path
            Path to the NOVONIX CSV file to be read.
        options : dict[str, str] | None, optional
            Options containing the timezone string (default ``"UTC"``), applied
            to ``Started``, which the header writes without an offset.

        Returns
        -------
        dict[str, Any]
            Header fields grouped the way the measurement API defines them, so
            the result can be merged into a measurement dict as-is:

            - ``protocol["name"]``: the ``Protocol`` schedule file, e.g.
              ``"200520 FM 42V C20.pro1"``.
            - ``test_setup["channel_number"]``: ``Channel``, kept as written
              (``"09"``) because the leading zero is part of how the lab names
              its channels.
            - ``test_setup["cell_label"]``: ``Cell``, the cycler's own label
              for the cell, which need not match the cell instance the
              measurement is uploaded to.
            - ``test_setup["source_file_id"]``: ``CellID``, an id in the
              cycler's own id space and **not** a cross-system identifier.
            - ``test_setup["notes"]``: ``Description``, unless it still
              holds the placeholder the Novonix UI pre-fills it with.
            - ``test_setup["cycler_serial_number"]``: ``Serial Number``.
            - ``test_setup["software_version"]``: ``Version``.
            - ``test_setup["cycler"]``: this reader's name.
            - ``start_time``: timezone-aware, from ``Started``. Top-level,
              because the API models it as a column rather than inside a group.

            Keys the header does not carry are omitted. Novonix writes several
            of these labels with an empty value (``Serial Number:`` alone on
            its line), and an empty value counts as absent rather than as an
            empty string.

            ``Mass``, ``Capacity``, ``Area`` and ``DC Offset Voltage`` are
            deliberately not returned: they describe the cell and the
            calibration, so they belong to the cell record rather than to one
            measurement's test setup.
        """
        # Consumed by the readers that fall back to the first data row; this
        # one reads its own, and only needs the key not to reach validation.
        _, options = take_known_start_time(options)
        opts = cast(
            dict[str, Any],
            iwutil.check_and_combine_options(self.default_options, options),
        )
        parsed = parse_labelled_lines(
            self._read_summary_lines(filename), _NOVONIX_SUMMARY_LABELS
        )
        if parsed.get("notes", "").strip().lower() == _NOVONIX_DESCRIPTION_PLACEHOLDER:
            del parsed["notes"]
        started = parsed.pop("start_time", None)
        if started is not None:
            naive = parse_datetime(started, _NOVONIX_START_FORMATS)
            if naive is not None:
                parsed["start_time"] = localize(naive, opts)
        nested = nest_metadata(parsed)
        nested.setdefault("test_setup", {})["cycler"] = self.name
        return nested


def novonix(
    filename: str | Path,
    extra_column_mappings: dict[str, str] | None = None,
    options: dict[str, str] | None = None,
) -> pl.DataFrame:
    return Novonix().run(
        filename, extra_column_mappings=extra_column_mappings, options=options
    )
