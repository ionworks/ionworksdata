"""Reader for Gamry EIS data files (.dta and .csv with ZCURVE tables)."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, cast

import iwutil  # type: ignore[reportMissingTypeStubs]
import polars as pl
import pytz  # type: ignore[reportMissingTypeStubs]

import ionworksdata as iwdata

from .read import BaseReader


class Gamry(BaseReader):
    name: str = "Gamry"
    default_options: dict[str, Any] = {
        "timezone": "UTC",
        "cell_metadata": {},
    }

    @staticmethod
    def _find_zcurve_table(
        lines: list[str],
    ) -> tuple[list[str], int]:
        """
        Locate the ZCURVE table in a list of file lines.

        Returns
        -------
        tuple[list[str], int]
            (headers, data_start_line_index)

        Raises
        ------
        ValueError
            If no ZCURVE table is found.
        """
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("ZCURVE"):
                if i + 1 >= len(lines):
                    raise ValueError(
                        "Found ZCURVE header but file is truncated "
                        "(missing column headers)"
                    )
                # Next line is headers, line after is units, data starts after that
                headers = lines[i + 1].strip().split("\t")
                return headers, i + 3
        raise ValueError("Could not find ZCURVE table in file")

    @staticmethod
    def _parse_start_time(lines: list[str]) -> datetime | None:
        """Extract start time from STARTTIME header line."""
        for line in lines:
            if line.strip().startswith("STARTTIME"):
                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    dt_str = parts[2].strip()
                    for fmt in [
                        "%Y-%m-%d %I:%M:%S %p",
                        "%Y-%m-%d %H:%M:%S",
                    ]:
                        try:
                            return datetime.strptime(dt_str, fmt)
                        except ValueError:
                            continue
        return None

    @staticmethod
    def _read_lines(filename: str | Path) -> list[str]:
        """Read all lines from a file, trying multiple encodings."""
        for encoding in ["utf-8", "latin1"]:
            try:
                with open(filename, encoding=encoding) as f:
                    return f.readlines()
            except UnicodeDecodeError:
                continue
        raise ValueError(f"Could not read file: {filename}")

    def run(
        self,
        filename: str | Path,
        extra_column_mappings: dict[str, str] | None = None,
        options: dict[str, str] | None = None,
    ) -> pl.DataFrame:
        """
        Read a Gamry EIS file and return a DataFrame with standardized columns.

        Supports both ``.dta`` (Gamry native) and ``.csv`` files that contain
        a ``ZCURVE`` table with tab-separated impedance data.

        Parameters
        ----------
        filename : str | Path
            Path to the Gamry file.
        extra_column_mappings : dict[str, str] | None, optional
            Additional column mappings.
        options : dict[str, str] | None, optional
            Reader options (``timezone``, ``cell_metadata``).

        Returns
        -------
        pl.DataFrame
            Standardised EIS data with columns:
            ``Time [s]``, ``Voltage [V]``, ``Current [A]``,
            ``Frequency [Hz]``, ``Z_Re [Ohm]``, ``Z_Im [Ohm]``,
            and optionally ``Z_Mod [Ohm]``, ``Z_Phase [deg]``.
        """
        options = iwutil.check_and_combine_options(self.default_options, options)
        lines = self._read_lines(filename)
        headers, data_start = self._find_zcurve_table(lines)

        # Parse data rows
        data_rows: list[list[str]] = []
        for line in lines[data_start:]:
            line = line.strip()
            if (
                not line
                or line.startswith("EXPERIMENTABORTED")
                or line.startswith("STOPABORT")
            ):
                break
            values = line.split("\t")
            if len(values) == len(headers):
                data_rows.append(values)

        df = pl.DataFrame(data_rows, schema=headers, orient="row")

        # Cast all numeric columns to Float64
        numeric_cols = [
            "Pt",
            "Time",
            "Freq",
            "Zreal",
            "Zimag",
            "Zsig",
            "Zmod",
            "Zphz",
            "Idc",
            "Vdc",
            "IERange",
            "Imod",
            "Vmod",
            "Temp",
        ]
        present_numeric = [c for c in numeric_cols if c in df.columns]
        if present_numeric:
            df = df.with_columns(
                [pl.col(c).cast(pl.Float64, strict=False) for c in present_numeric]
            )

        # Standard column renamings
        column_renamings = {
            "Time": "Time [s]",
            "Vdc": "Voltage [V]",
            "Idc": "Current [A]",
            "Freq": "Frequency [Hz]",
            "Zreal": "Z_Re [Ohm]",
            "Zimag": "Z_Im [Ohm]",
            "Zmod": "Z_Mod [Ohm]",
            "Zphz": "Z_Phase [deg]",
            "Temp": "Temperature [degC]",
        }
        column_renamings.update(extra_column_mappings or {})
        iwdata.util.check_for_duplicates(column_renamings, df)
        present_map = {k: v for k, v in column_renamings.items() if k in df.columns}
        if present_map:
            df = df.rename(present_map)

        columns_keep = [
            col
            for col in [
                "Time [s]",
                "Voltage [V]",
                "Current [A]",
                "Frequency [Hz]",
                "Z_Re [Ohm]",
                "Z_Im [Ohm]",
                "Z_Mod [Ohm]",
                "Z_Phase [deg]",
                "Temperature [degC]",
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
        Read the test start time from the Gamry file header.

        Looks for a ``STARTTIME`` line in the metadata section.

        Returns
        -------
        datetime | None
            Timezone-aware start time, or ``None`` if not found.
        """
        opts = cast(
            dict[str, Any],
            iwutil.check_and_combine_options(self.default_options, options),
        )
        lines = self._read_lines(filename)
        start_datetime = self._parse_start_time(lines)
        if start_datetime is None:
            return None

        timezone = opts.get("timezone", "UTC")
        if isinstance(timezone, str):
            tz = pytz.timezone(timezone)
        else:
            raise ValueError(f"Invalid timezone: {timezone}")
        start_datetime = tz.localize(start_datetime)
        start_datetime = iwdata.util.check_and_convert_datetime(
            cast(datetime, start_datetime)
        )
        return start_datetime


def gamry(
    filename: str | Path,
    extra_column_mappings: dict[str, str] | None = None,
    options: dict[str, str] | None = None,
) -> pl.DataFrame:
    return Gamry().run(
        filename, extra_column_mappings=extra_column_mappings, options=options
    )
