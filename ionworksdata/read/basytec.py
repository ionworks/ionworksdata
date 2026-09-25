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
import re
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

#: BaSyTec's ``.txt`` result export prefixes every metadata line with ``~``;
#: the last such line is the column header.
_TXT_PREAMBLE_PREFIX = "~"

#: Header names carry their unit in brackets, e.g. ``U[V]``, ``Time[h]``. The
#: degree sign in ``T1[°C]`` is latin1, so the unit is not safe to match on.
_TXT_UNIT_RE = re.compile(r"^(?P<base>[^\[]+)(?:\[(?P<unit>.*)\])?$")

#: Maps a bracket-stripped ``.txt`` header to its ionworks column. Time is
#: handled separately because its unit decides the scale factor.
_TXT_COLUMN_MAP: dict[str, str] = {
    "U": "Voltage [V]",
    "I": "Current [A]",
    "T1": "Temperature [degC]",
    "Line": "Step from cycler",
    "Cyc-Count": "Cycle from cycler",
    "Ah-Charge": "Charge capacity [A.h]",
    "Ah-Discharge": "Discharge capacity [A.h]",
    "Wh-Charge": "Charge energy [W.h]",
    "Wh-Discharge": "Discharge energy [W.h]",
}

#: Channels both BaSyTec dialects log.
_CORE_COLUMNS = (
    "Time [s]",
    "Voltage [V]",
    "Current [A]",
    "Temperature [degC]",
    "Step from cycler",
)

#: The ``.txt`` dialect also carries a cycle counter and vendor accumulators.
_TXT_EXTRA_COLUMNS = (
    "Cycle from cycler",
    "Charge capacity [A.h]",
    "Discharge capacity [A.h]",
    "Charge energy [W.h]",
    "Discharge energy [W.h]",
)

#: Text columns in the ``.txt`` dialect; everything else is numeric.
_TXT_NON_NUMERIC_COLUMNS = frozenset({"Command", "State"})

#: Seconds per unit of the ``Time[...]`` column.
_TXT_TIME_SCALES: dict[str, float] = {"s": 1.0, "min": 60.0, "h": 3600.0}

#: ``Operator (Data converting)`` is deliberately absent: it names whoever ran
#: the export, which says nothing about who ran the test.
_TXT_METADATA_LABELS: dict[str, str] = {
    "testplan": "procedure",
    "name of test": "test_name",
    "battery": "cell_label",
    "testchannel": "channel_number",
    "operator (test)": "operator",
    "start of test": "start_time",
    "end of test": "end_time",
    "date and time of data converting": "export_date",
}

#: The companion file describes the rig rather than the schedule, so it
#: contributes no ``procedure``.
_META_LABELS: dict[str, str] = {
    "measurement start date": "start_time",
    "measurement device used": "instrument",
    "test channel": "channel_number",
    "internal cell serial": "cell_label",
    "laboratory identifier": "operator",
    "climate chamber temperature setpoint": "temperature_setpoint_degc",
}

#: Both dialects write German-style dates; only the ``.txt`` one adds a time.
_TXT_DATETIME_FORMATS = ("%d.%m.%Y %H:%M:%S", "%d.%m.%Y %H:%M", "%d.%m.%Y")

#: ``23°C`` -- the unit is part of the value, and the degree sign is latin1.
_META_SETPOINT_RE = re.compile(r"^\s*(?P<value>-?\d+(?:[.,]\d+)?)")


def _split_txt_header(name: str) -> tuple[str, str | None]:
    """Split a BaSyTec ``.txt`` header into ``(base_name, unit)``.

    Parameters
    ----------
    name : str
        Raw header cell, e.g. ``"U[V]"`` or ``"Cyc-Count"``.

    Returns
    -------
    tuple[str, str | None]
        Base name with the bracketed unit removed, and the unit if present.
    """
    match = _TXT_UNIT_RE.match(name.strip())
    if match is None:
        return name.strip(), None
    unit = match.group("unit")
    return match.group("base").strip(), unit.strip() if unit else None


def _read_txt_header(filename: str | Path, encoding: str) -> tuple[int, list[str], str]:
    """Locate and parse the header of a BaSyTec ``.txt`` export.

    Parameters
    ----------
    filename : str | Path
        Path to the ``.txt`` file.
    encoding : str
        Text encoding to read with.

    Returns
    -------
    tuple[int, list[str], str]
        Number of lines to skip before the data, the header cells, and the
        field separator (tab or whitespace).

    Raises
    ------
    ValueError
        If no ``~``-prefixed header line is found.
    """
    header_line = None
    skiprows = 0
    with open(filename, encoding=encoding) as handle:
        for i, line in enumerate(handle):
            if not line.startswith(_TXT_PREAMBLE_PREFIX):
                break
            header_line = line
            skiprows = i + 1
    if header_line is None:
        raise ValueError(
            f"{filename} does not look like a BaSyTec .txt export: no "
            f"'~'-prefixed header line was found."
        )
    body = header_line.lstrip(_TXT_PREAMBLE_PREFIX).rstrip("\n")
    sep = "\t" if "\t" in body else " "
    cells = [c for c in body.split(sep) if c.strip()] if sep == " " else body.split(sep)
    return skiprows, [c.strip() for c in cells], sep


class Basytec(BaseReader):
    name: str = "Basytec"
    default_options: dict[str, Any] = {
        "timezone": "UTC",
        "cell_metadata": {},
        "file_encoding": "latin1",
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

    def _run_txt(
        self,
        filename: str | Path,
        extra_column_mappings: dict[str, str] | None = None,
        options: dict[str, Any] | None = None,
    ) -> pl.DataFrame:
        """Read a BaSyTec ``.txt`` result export.

        This is a different export dialect from the ``.csv`` one: a
        ``~``-prefixed preamble whose last line is the header, bracketed units
        (``U[V]``, ``Time[h]``), and either tab- or whitespace-separated
        fields.

        Parameters
        ----------
        filename : str | Path
            Path to the BaSyTec ``.txt`` file.
        extra_column_mappings : dict[str, str] | None, optional
            Additional raw → ionworks column mappings.
        options : dict[str, Any] | None, optional
            Reader options; ``file_encoding`` selects the text encoding.

        Returns
        -------
        pl.DataFrame
            Time series with standardized ionworks columns.
        """
        opts = options or {}
        encoding = str(opts.get("file_encoding", "latin1"))
        skiprows, header, sep = _read_txt_header(filename, encoding)

        read_kwargs: dict[str, Any] = {
            "has_header": False,
            "new_columns": header,
            "skip_rows": skiprows,
            "null_values": ["NaN", "nan", ""],
            "encoding": encoding,
            "truncate_ragged_lines": True,
            "infer_schema_length": 10000,
        }
        if sep == "\t":
            df = pl.read_csv(filename, separator="\t", **read_kwargs)
        else:
            # Splitting in the engine, because collapsing space runs in Python
            # held three copies of the file.
            df = (
                pl.scan_csv(
                    filename,
                    separator="\x01",
                    has_header=False,
                    skip_rows=skiprows,
                    new_columns=["_raw"],
                    encoding="utf8-lossy",
                    infer_schema_length=0,
                    truncate_ragged_lines=True,
                )
                .select(
                    [
                        pl.col("_raw")
                        .str.extract_all(r"\S+")
                        .list.get(i, null_on_oob=True)
                        .alias(name)
                        for i, name in enumerate(header)
                    ]
                )
                .collect(engine="streaming")
            )
            # Every field arrives as text; the tab path gets dtypes from Polars.
            df = df.with_columns(
                pl.col(c).cast(pl.Float64, strict=False)
                for c in df.columns
                if c not in _TXT_NON_NUMERIC_COLUMNS
            )

        renamings: dict[str, str] = {}
        time_column: str | None = None
        time_scale = 1.0
        for col in df.columns:
            base, unit = _split_txt_header(col)
            if base == "Time":
                # A .txt export may log time in seconds, minutes or hours.
                time_column = col
                time_scale = _TXT_TIME_SCALES.get((unit or "s").lower(), 1.0)
                continue
            target = _TXT_COLUMN_MAP.get(base)
            if target is None:
                continue
            renamings[col] = target
        renamings.update(extra_column_mappings or {})
        present = iwdata.util.resolve_renamings(
            renamings, df, priority=extra_column_mappings
        )
        if present:
            df = df.rename(present)

        if time_column is None:
            raise ValueError(
                f"{filename} has no 'Time[...]' column; found {header!r}. "
                f"A BaSyTec .txt export must carry a time column."
            )
        df = df.with_columns(
            (pl.col(time_column).cast(pl.Float64, strict=False) * time_scale).alias(
                "Time [s]"
            )
        )

        # BaSyTec logs positive current for charge; ionworks uses positive for
        # discharge.
        if "Current [A]" in df.columns:
            df = df.with_columns((-pl.col("Current [A]")).alias("Current [A]"))

        columns_keep = [
            col for col in _CORE_COLUMNS + _TXT_EXTRA_COLUMNS if col in df.columns
        ]
        return self.standard_data_processing(df, columns_keep=columns_keep)

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

        if Path(filename).suffix.lower() == ".txt":
            return self._run_txt(
                filename,
                extra_column_mappings=extra_column_mappings,
                options=cast(dict[str, Any], options),
            )

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
        present_map = iwdata.util.resolve_renamings(
            column_renamings, df, priority=extra_column_mappings
        )
        if present_map:
            df = df.rename(present_map)

        # Flip current sign: BaSyTec uses negative=discharge, ionworks uses
        # positive=discharge
        df = df.with_columns((-pl.col("Current [A]")).alias("Current [A]"))

        columns_keep = [col for col in _CORE_COLUMNS if col in df.columns]

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

    @staticmethod
    def _read_txt_preamble(filename: str | Path, encoding: str) -> list[str]:
        """
        Read the ``~``-prefixed preamble lines of a ``.txt`` result export.

        Parameters
        ----------
        filename : str | Path
            Path to the ``.txt`` file.
        encoding : str
            Text encoding to read with.

        Returns
        -------
        list[str]
            The preamble lines, minus the last one -- that is the column
            header, whose ``U[V]``/``I[A]`` cells would otherwise parse as
            labelled values.
        """
        lines: list[str] = []
        with open(filename, encoding=encoding) as handle:
            for line in handle:
                if not line.startswith(_TXT_PREAMBLE_PREFIX):
                    break
                lines.append(line)
        return lines[:-1]

    @classmethod
    def _parse_meta_setpoint(cls, value: str) -> float | None:
        """Read the numeric part out of a ``23°C`` chamber setpoint."""
        match = _META_SETPOINT_RE.match(value)
        if match is None:
            return None
        return float(match.group("value").replace(",", "."))

    def read_metadata(
        self,
        filename: str | Path,
        options: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """
        Read the descriptive metadata a BaSyTec export carries.

        The two dialects keep it in different places, and this reads whichever
        the file has: the ``.txt`` result export states it in its ``~``-prefixed
        preamble, while the CSV dialect has no preamble at all and states it in
        a companion ``<stem>_meta.txt``.

        Parameters
        ----------
        filename : str | Path
            Path to the BaSyTec ``.txt`` or ``.csv`` file.
        options : dict[str, str] | None, optional
            Options containing the timezone string (default ``"UTC"``), applied
            to the header's timestamps, which carry no offset.

        Returns
        -------
        dict[str, Any]
            Header fields grouped the way the measurement API defines them:

            - ``protocol["name"]``: the ``Testplan`` (``.txt`` only -- the
              companion file names no schedule).
            - ``test_setup["test_name"]``: ``Name of Test`` (``.txt``).
            - ``test_setup["cell_label"]``: the ``.txt`` dialect's ``Battery``
              type string, or the companion file's ``Internal cell serial``.
            - ``test_setup["channel_number"]``: ``Testchannel`` / ``Test
              channel``, kept as written -- BaSyTec writes a rig-and-channel
              string (``"#10 1563 CH10"``), not a bare number.
            - ``test_setup["operator"]``: ``Operator (Test)``, or the companion
              file's ``Laboratory identifier``. ``Operator (Data converting)``
              is not read: it names whoever ran the export.
            - ``test_setup["export_date"]``: ``Date and Time of Data
              Converting`` (``.txt``), as an ISO string.
            - ``test_setup["end_time"]``: ``End of Test`` (``.txt``), as an ISO
              string, since the API models only ``start_time`` as a datetime.
            - ``test_setup["instrument"]``: ``Measurement device used``
              (companion file).
            - ``test_setup["temperature_setpoint_degc"]``: the numeric part of
              ``Climate chamber temperature setpoint`` (companion file), in
              degrees Celsius.
            - ``test_setup["cycler"]``: this reader's name.
            - ``start_time``: timezone-aware, from ``Start of Test`` (``.txt``,
              to the second) or ``Measurement start date`` (companion file,
              date only, so midnight).

            Keys the file does not carry are omitted, including labels BaSyTec
            writes with an empty value -- ``~Operator (Test):`` alone on its
            line yields no ``operator``.

        Notes
        -----
        This is the only path that reads a ``.txt`` export's ``Start of Test``.
        :meth:`read_start_time` looks only for the companion file, so it
        returns None for a ``.txt`` even though the preamble states the time.
        """
        # Consumed by the readers that fall back to the first data row; this
        # one reads its own, and only needs the key not to reach validation.
        _, options = take_known_start_time(options)
        opts = cast(
            dict[str, Any],
            iwutil.check_and_combine_options(self.default_options, options),
        )
        parsed: dict[str, Any] = {}
        if Path(filename).suffix.lower() == ".txt":
            parsed = parse_labelled_lines(
                self._read_txt_preamble(filename, opts["file_encoding"]),
                _TXT_METADATA_LABELS,
                strip_prefix=_TXT_PREAMBLE_PREFIX,
            )
        else:
            meta_path = self._find_meta_file(filename)
            if meta_path is not None:
                with open(meta_path, encoding="utf-8") as handle:
                    parsed = parse_labelled_lines(handle.readlines(), _META_LABELS)
                setpoint = parsed.pop("temperature_setpoint_degc", None)
                if setpoint is not None:
                    parsed["temperature_setpoint_degc"] = self._parse_meta_setpoint(
                        setpoint
                    )

        # Both dialects write naive local timestamps, so the option decides the
        # zone; an unrecognized spelling drops the field rather than failing.
        for key in ("start_time", "end_time", "export_date"):
            raw = parsed.pop(key, None)
            if not isinstance(raw, str):
                continue
            naive = parse_datetime(raw, _TXT_DATETIME_FORMATS)
            if naive is None:
                continue
            aware = localize(naive, opts)
            # ISO rather than BaSyTec's ``05.09.2025 15:53:31``, so a consumer
            # reading test_setup across cyclers gets one format.
            parsed[key] = aware if key == "start_time" else aware.isoformat()

        nested = nest_metadata(parsed)
        nested.setdefault("test_setup", {})["cycler"] = self.name
        return nested


def basytec(
    filename: str | Path,
    extra_column_mappings: dict[str, str] | None = None,
    options: dict[str, str] | None = None,
) -> pl.DataFrame:
    return Basytec().run(
        filename, extra_column_mappings=extra_column_mappings, options=options
    )
