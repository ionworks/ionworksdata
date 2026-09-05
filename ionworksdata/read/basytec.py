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


def basytec(
    filename: str | Path,
    extra_column_mappings: dict[str, str] | None = None,
    options: dict[str, str] | None = None,
) -> pl.DataFrame:
    return Basytec().run(
        filename, extra_column_mappings=extra_column_mappings, options=options
    )
