"""Reader for Gamry EIS data files (.dta and .csv with ZCURVE tables)."""

from __future__ import annotations

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
    take_known_start_time,
)
from .read import BaseReader

#: The metadata block that opens a Gamry file, terminated by the data table.
_GAMRY_EXPLAIN = "EXPLAIN"
_GAMRY_TABLE_TAG = "ZCURVE"

#: ``<TAG>\t<TYPE>\t<VALUE>\t<LABEL>`` per line. ``FREQINIT``/``FREQFINAL``/
#: ``IACREQ`` are omitted: sweep parameters belong to the protocol's steps.
_GAMRY_EXPLAIN_TAGS: dict[str, str] = {
    # The technique that produced the file (``GALVEIS``), which is the closest
    # thing a Gamry export has to a named schedule.
    "TAG": "procedure",
    "TITLE": "test_name",
    "PSTAT": "instrument",
    "STARTTIME": "start_time",
}

#: ``TAG`` states its value in the second field; every other row uses the
#: third, the second being the value's type.
_GAMRY_VALUE_FIELD: dict[str, int] = {"TAG": 1}

#: Gamry writes a 12-hour clock with an AM/PM marker, and a 24-hour one
#: depending on the host's locale settings.
_GAMRY_START_FORMATS = ("%Y-%m-%d %I:%M:%S %p", "%Y-%m-%d %H:%M:%S")


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
        present_map = iwdata.util.resolve_renamings(
            column_renamings, df, priority=extra_column_mappings
        )
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

    @classmethod
    def _parse_explain(cls, lines: list[str]) -> dict[str, str]:
        """
        Read the ``EXPLAIN`` metadata block of a Gamry file.

        Parameters
        ----------
        lines : list[str]
            All lines of the file, in order.

        Returns
        -------
        dict[str, str]
            Canonical key to value, for the tags the block carries.

        Notes
        -----
        Parsing stops at ``ZCURVE``, which opens the data table: its rows are
        tab-separated too, so a scan of the whole file would read sweep points
        as tag rows.
        """
        found: dict[str, str] = {}
        for line in lines:
            stripped = line.strip()
            if stripped.startswith(_GAMRY_TABLE_TAG):
                break
            fields = stripped.split("\t")
            key = _GAMRY_EXPLAIN_TAGS.get(fields[0].strip())
            if key is None:
                continue
            # ``TAG`` states its value in field 1; the rest use field 2, with
            # field 1 holding the value's type.
            index = _GAMRY_VALUE_FIELD.get(fields[0].strip(), 2)
            if len(fields) <= index:
                continue
            value = fields[index].strip()
            if value:
                found.setdefault(key, value)
        return found

    def read_metadata(
        self,
        filename: str | Path,
        options: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """
        Read the descriptive metadata from a Gamry file's ``EXPLAIN`` block.

        Parameters
        ----------
        filename : str | Path
            Path to the Gamry ``.dta``/``.csv`` file.
        options : dict[str, str] | None, optional
            Options containing the timezone string (default ``"UTC"``), applied
            to ``STARTTIME``, which carries no offset.

        Returns
        -------
        dict[str, Any]
            Fields grouped the way the measurement API defines them:

            - ``protocol["name"]``: the ``TAG``, i.e. the technique that
              produced the file (``"GALVEIS"``). A Gamry export names no
              schedule file, so the technique is the closest equivalent.
            - ``test_setup["test_name"]``: ``TITLE``, e.g.
              ``"Galvanostatic EIS"``.
            - ``test_setup["instrument"]``: ``PSTAT``, the potentiostat's id.
            - ``test_setup["cycler"]``: this reader's name.
            - ``start_time``: timezone-aware, from ``STARTTIME``.

            Keys the block does not carry are omitted.

            The sweep's own parameters (``FREQINIT``, ``FREQFINAL``,
            ``IACREQ``, ``IDCREQ``) are deliberately not returned: they
            describe the steps the protocol ran, not the physical setup.
        """
        # Consumed by the readers that fall back to the first data row; this
        # one reads its own, and only needs the key not to reach validation.
        _, options = take_known_start_time(options)
        opts = cast(
            dict[str, Any],
            iwutil.check_and_combine_options(self.default_options, options),
        )
        parsed: dict[str, Any] = dict(self._parse_explain(self._read_lines(filename)))
        started = parsed.pop("start_time", None)
        if isinstance(started, str):
            naive = parse_datetime(started, _GAMRY_START_FORMATS)
            if naive is not None:
                parsed["start_time"] = localize(naive, opts)
        nested = nest_metadata(parsed)
        nested.setdefault("test_setup", {})["cycler"] = self.name
        return nested


def gamry(
    filename: str | Path,
    extra_column_mappings: dict[str, str] | None = None,
    options: dict[str, str] | None = None,
) -> pl.DataFrame:
    return Gamry().run(
        filename, extra_column_mappings=extra_column_mappings, options=options
    )
