"""Reader for Arbin battery cycler exports.

Arbin cyclers (LBT, MITS Pro) are widely used in battery testing labs. This
reader handles three file types:

- **CSV / XLSX**: MITS Pro text exports (the form PyProBE consumes).
- **RES**: Native binary Access/MDB database written by MITS Pro. Requires
  ``mdb-export`` from the ``mdbtools`` package (``brew install mdbtools``).

CSV/XLSX file format
--------------------
Single header row with units in parentheses. Typical columns:

``Data Point, Date Time, Test Time (s), Step Time (s), Cycle Index,
Step Index, Current (A), Voltage (V), Charge Capacity (Ah),
Discharge Capacity (Ah), Charge Energy (Wh), Discharge Energy (Wh),
Aux_Temperature_1 (C)``

Datetime format is ``MM/DD/YYYY HH:MM:SS[.fff]``. Time is cumulative seconds
from test start; ``Step Time`` resets each step.

RES file format
---------------
The ``.res`` file is an MDB/Access database. The main data table is
``Channel_Normal_Table`` with columns (no unit suffixes):

``Test_ID, Data_Point, Test_Time, Step_Time, DateTime, Step_Index,
Cycle_Index, Is_FC_Data, Current, Voltage, Charge_Capacity,
Discharge_Capacity, Charge_Energy, Discharge_Energy, dV/dt,
Internal_Resistance, AC_Impedance, ACI_Phase_Angle``

``DateTime`` is an OLE Automation date (float days since 1899-12-30).
``Test_Time`` is cumulative seconds. Current units are Amps, capacity in A·h,
energy in W·h — same units as the CSV export; no conversion needed.

The start time is read from ``Global_Table.Start_DateTime``.

Current sign convention
-----------------------
Arbin uses positive current for charge and negative for discharge. The
ionworks convention is the opposite (positive = discharge). Sign correction
is handled by ``standard_data_processing`` via
``set_positive_current_for_discharge`` — no manual flip is applied here.

Capacity columns
----------------
Arbin exports separate ``Charge Capacity`` and ``Discharge Capacity`` columns
that monotonically accumulate within each step. They map to the ionworks
``Charge capacity [A.h]`` / ``Discharge capacity [A.h]`` columns directly,
and downstream ``set_capacity`` / ``set_energy`` consume these columns
instead of integrating current — they only apply per-step resets so each
step starts at 0 (the ionworks convention; cross-step accumulation lives in
the steps summary). The raw integration fallback only kicks in when these
columns are absent.
"""

# pyright: reportMissingTypeStubs=false
from __future__ import annotations

import csv
from datetime import datetime, timedelta
import io
from pathlib import Path
import re
import subprocess
from typing import Any, cast

import iwutil  # type: ignore[reportMissingTypeStubs]
import polars as pl
import pytz  # type: ignore[reportMissingTypeStubs]

import ionworksdata as iwdata

from ._utils import (
    read_excel_and_get_column_names,
    suppress_excel_dtype_warnings,
)
from .read import BaseReader

# OLE Automation date epoch used by Arbin .res files.
_OLE_EPOCH = datetime(1899, 12, 30)


def _ole_to_datetime(ole: float) -> datetime:
    """Convert an OLE Automation date float to a Python datetime.

    Parameters
    ----------
    ole : float
        Days since 1899-12-30 (the OLE epoch used by Arbin .res files).

    Returns
    -------
    datetime
        Corresponding naive UTC datetime.
    """
    return _OLE_EPOCH + timedelta(days=ole)


def _mdb_export(filename: str | Path, table: str) -> str:
    """Run mdb-export and return the CSV output as a string.

    Parameters
    ----------
    filename : str | Path
        Path to the ``.res`` (MDB/Access) file.
    table : str
        Name of the table to export.

    Returns
    -------
    str
        Raw CSV output from ``mdb-export``.

    Raises
    ------
    RuntimeError
        If ``mdb-export`` is not found on PATH.
    subprocess.CalledProcessError
        If ``mdb-export`` exits with a non-zero status.
    """
    try:
        result = subprocess.run(
            ["mdb-export", str(filename), table],
            capture_output=True,
            text=True,
            check=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Could not locate the 'mdb-export' executable. "
            "Install mdbtools to read .res files: brew install mdbtools "
            "(macOS) or apt-get install mdbtools (Linux)."
        ) from exc
    return result.stdout


def _read_res_channel_table(filename: str | Path) -> pl.DataFrame:
    """Read the Channel_Normal_Table from an Arbin .res file.

    Arbin ``.res`` files may contain interleaved data from multiple recording
    sessions (e.g. when a test is paused and resumed on a different channel
    slot). The rows are sorted by ``DateTime`` (OLE Automation float) to
    reconstruct chronological order before any downstream processing.

    Parameters
    ----------
    filename : str | Path
        Path to the ``.res`` file.

    Returns
    -------
    pl.DataFrame
        DataFrame sorted by ``DateTime`` with float-typed numeric columns.
    """
    csv_text = _mdb_export(filename, "Channel_Normal_Table")
    reader = csv.DictReader(io.StringIO(csv_text))
    rows = list(reader)
    if not rows:
        return pl.DataFrame()
    numeric_cols = [
        "Test_Time",
        "Step_Time",
        "DateTime",
        "Current",
        "Voltage",
        "Charge_Capacity",
        "Discharge_Capacity",
        "Charge_Energy",
        "Discharge_Energy",
    ]
    df = pl.DataFrame(rows)
    cast_exprs = [
        pl.col(c).cast(pl.Float64, strict=False)
        for c in numeric_cols
        if c in df.columns
    ]
    if cast_exprs:
        df = df.with_columns(cast_exprs)
    if "DateTime" in df.columns:
        df = df.sort("DateTime")
    return df


def _read_res_start_datetime(filename: str | Path) -> float | None:
    """Read Start_DateTime from Global_Table in an Arbin .res file.

    Parameters
    ----------
    filename : str | Path
        Path to the ``.res`` file.

    Returns
    -------
    float | None
        OLE Automation date float, or None if not available.
    """
    csv_text = _mdb_export(filename, "Global_Table")
    reader = csv.DictReader(io.StringIO(csv_text))
    for row in reader:
        raw = row.get("Start_DateTime")
        if raw:
            try:
                return float(raw)
            except ValueError:
                return None
    return None


# Pattern for Arbin column names with unit suffix in parentheses, e.g.
# "Current (A)", "Charge Capacity (Ah)", "Aux_Temperature_1 (C)". The unit
# is captured so we can convert if needed (mA → A, etc.).
_ARBIN_UNIT_RE = re.compile(r"^(?P<base>.+?)\s*\((?P<unit>[^)]+)\)\s*$")


def _strip_unit(col: str) -> tuple[str, str | None]:
    """Split an Arbin header into ``(base_name, unit)``.

    Parameters
    ----------
    col : str
        Raw header string from the CSV/XLSX (e.g. ``"Current (A)"``).

    Returns
    -------
    tuple[str, str | None]
        Lowercase base name (whitespace and underscores normalized) and the
        unit string. If no unit is present, ``unit`` is ``None``.
    """
    match = _ARBIN_UNIT_RE.match(col)
    if match is None:
        return col.strip().lower(), None
    return match.group("base").strip().lower(), match.group("unit").strip()


# Map from Arbin lowercased base name to ionworks column name. Columns with a
# unit suffix lose that suffix during lookup; non-unit columns (Cycle Index,
# Step Index, Date Time) match directly.
_ARBIN_COLUMN_MAP: dict[str, str] = {
    "test time": "Time [s]",
    "current": "Current [A]",
    "voltage": "Voltage [V]",
    "charge capacity": "Charge capacity [A.h]",
    "discharge capacity": "Discharge capacity [A.h]",
    "charge energy": "Charge energy [W.h]",
    "discharge energy": "Discharge energy [W.h]",
    "aux_temperature_1": "Temperature [degC]",
    "cycle index": "Cycle from cycler",
    "step index": "Step from cycler",
    "date time": "Timestamp",
}

# Column map for .res files (MDB schema): names use underscores, no unit suffixes.
# Current, Voltage, Charge_Capacity etc. are already in SI units (A, V, A·h, W·h).
_ARBIN_RES_COLUMN_MAP: dict[str, str] = {
    "Test_Time": "Time [s]",
    "Current": "Current [A]",
    "Voltage": "Voltage [V]",
    "Charge_Capacity": "Charge capacity [A.h]",
    "Discharge_Capacity": "Discharge capacity [A.h]",
    "Charge_Energy": "Charge energy [W.h]",
    "Discharge_Energy": "Discharge energy [W.h]",
    "Cycle_Index": "Cycle from cycler",
    "Step_Index": "Step from cycler",
    "DateTime": "_ole_datetime",
}

# Unit conversion factors keyed by (ionworks column, raw unit lower).
# Anything not listed is assumed to already be in ionworks units.
_UNIT_SCALES: dict[tuple[str, str], float] = {
    ("Current [A]", "ma"): 1e-3,
    ("Time [s]", "min"): 60.0,
    ("Time [s]", "h"): 3600.0,
    ("Time [s]", "hr"): 3600.0,
    ("Charge capacity [A.h]", "mah"): 1e-3,
    ("Discharge capacity [A.h]", "mah"): 1e-3,
    ("Charge energy [W.h]", "mwh"): 1e-3,
    ("Discharge energy [W.h]", "mwh"): 1e-3,
}


def _build_renamings(columns: list[str]) -> tuple[dict[str, str], dict[str, float]]:
    """Map raw Arbin headers to ionworks names and detect unit conversions.

    Parameters
    ----------
    columns : list[str]
        Raw header strings from the file.

    Returns
    -------
    tuple[dict[str, str], dict[str, float]]
        ``renamings`` maps raw header → ionworks name. ``scales`` maps the
        ionworks name → multiplicative factor to apply (only entries that
        differ from 1.0 are included).
    """
    renamings: dict[str, str] = {}
    scales: dict[str, float] = {}
    for col in columns:
        base, unit = _strip_unit(col)
        target = _ARBIN_COLUMN_MAP.get(base)
        if target is None:
            continue
        renamings[col] = target
        if unit is not None:
            scale = _UNIT_SCALES.get((target, unit.lower()))
            if scale is not None and scale != 1.0:
                scales[target] = scale
    return renamings, scales


class Arbin(BaseReader):
    name: str = "Arbin"
    default_options: dict[str, Any] = {
        "timezone": "UTC",
        "cell_metadata": {},
    }

    @staticmethod
    def _read_file(filename: str | Path) -> pl.DataFrame:
        """Read an Arbin CSV, XLSX, or RES export into a Polars DataFrame.

        Parameters
        ----------
        filename : str | Path
            Path to the Arbin export file.

        Returns
        -------
        pl.DataFrame
            Raw DataFrame with original Arbin column names preserved.
            For ``.res`` files, column names match the MDB schema
            (e.g. ``Test_Time``, ``Cycle_Index``) rather than the CSV
            export style (e.g. ``Test Time (s)``, ``Cycle Index``).
        """
        ext = Path(filename).suffix.lower()
        if ext in (".xls", ".xlsx"):
            with suppress_excel_dtype_warnings():
                df = pl.read_excel(filename)
            return df
        if ext == ".res":
            return _read_res_channel_table(filename)
        return pl.read_csv(
            filename,
            null_values=["NaN", "nan", ""],
            try_parse_dates=False,
            infer_schema_length=10000,
        )

    def run(
        self,
        filename: str | Path,
        extra_column_mappings: dict[str, str] | None = None,
        options: dict[str, str] | None = None,
    ) -> pl.DataFrame:
        """Read an Arbin export and return a DataFrame with standardized columns.

        Parameters
        ----------
        filename : str | Path
            Path to the Arbin CSV or XLSX file.
        extra_column_mappings : dict[str, str] | None, optional
            Additional raw → ionworks column mappings applied after the
            built-in ones.
        options : dict[str, str] | None, optional
            Options:

                - timezone : str, optional
                    Timezone to assume for ``Date Time``. Default ``"UTC"``.
                - cell_metadata : dict, optional
                    Reserved for caller-supplied cell metadata.

        Returns
        -------
        pl.DataFrame
            Time series with columns mapped to:
            - ``Time [s]``
            - ``Voltage [V]``
            - ``Current [A]``
            - ``Cycle from cycler`` (if available)
            - ``Step from cycler`` (if available)
            - ``Charge capacity [A.h]`` / ``Discharge capacity [A.h]`` (if available)
            - ``Charge energy [W.h]`` / ``Discharge energy [W.h]`` (if available)
            - ``Temperature [degC]`` (if available)
        """
        options = iwutil.check_and_combine_options(self.default_options, options)
        ext = Path(filename).suffix.lower()
        is_res = ext == ".res"

        df = self._read_file(filename)

        if is_res:
            present = {
                k: v for k, v in _ARBIN_RES_COLUMN_MAP.items() if k in df.columns
            }
            if present:
                df = df.rename(present)
            # Drop the OLE datetime sentinel — start time is read separately.
            if "_ole_datetime" in df.columns:
                df = df.drop("_ole_datetime")
            scales: dict[str, float] = {}
        else:
            renamings, scales = _build_renamings(df.columns)
            renamings.update(extra_column_mappings or {})
            iwdata.util.check_for_duplicates(renamings, df)
            present = {k: v for k, v in renamings.items() if k in df.columns}
            if present:
                df = df.rename(present)

        for target, factor in scales.items():
            if target in df.columns:
                df = df.with_columns(
                    (pl.col(target).cast(pl.Float64, strict=False) * factor).alias(
                        target
                    )
                )

        # Drop the parsed timestamp column — ``Time [s]`` is the time of record
        # and ``read_start_time`` reads ``Date Time`` separately.
        if "Timestamp" in df.columns:
            df = df.drop("Timestamp")

        columns_keep = [
            col
            for col in [
                "Time [s]",
                "Voltage [V]",
                "Current [A]",
                "Cycle from cycler",
                "Step from cycler",
                "Charge capacity [A.h]",
                "Discharge capacity [A.h]",
                "Charge energy [W.h]",
                "Discharge energy [W.h]",
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
        """Read the first ``Date Time`` value from the Arbin file.

        Parameters
        ----------
        filename : str | Path
            Path to the Arbin file.
        extra_column_mappings : dict[str, str] | None, optional
            Unused, present for API compatibility.
        options : dict[str, str] | None, optional
            Options containing the timezone string (default ``"UTC"``).

        Returns
        -------
        datetime | None
            Timezone-aware start time, or None if no ``Date Time`` column was
            found or the first row is empty.
        """
        opts = cast(
            dict[str, Any],
            iwutil.check_and_combine_options(self.default_options, options),
        )
        timezone = opts.get("timezone", "UTC")
        if not isinstance(timezone, str):
            raise ValueError(f"Invalid timezone: {timezone}")
        tz = pytz.timezone(timezone)

        ext = Path(filename).suffix.lower()

        if ext == ".res":
            ole = _read_res_start_datetime(filename)
            if ole is None:
                return None
            naive = _ole_to_datetime(ole)
            localized = tz.localize(naive)
            return iwdata.util.check_and_convert_datetime(cast(datetime, localized))

        if ext in (".xls", ".xlsx"):
            df, _cols = read_excel_and_get_column_names(Path(filename))
            if df is None or "Date Time" not in df.columns or df.height == 0:
                return None
            raw = df["Date Time"][0]
        else:
            head = pl.read_csv(filename, n_rows=1)
            if "Date Time" not in head.columns or head.height == 0:
                return None
            raw = head["Date Time"][0]

        if raw is None:
            return None

        if isinstance(raw, datetime):
            naive = raw.replace(tzinfo=None) if raw.tzinfo is not None else raw
        else:
            naive = None
            for fmt in (
                "%m/%d/%Y %H:%M:%S.%f",
                "%m/%d/%Y %H:%M:%S",
                "%Y-%m-%d %H:%M:%S.%f",
                "%Y-%m-%d %H:%M:%S",
            ):
                try:
                    naive = datetime.strptime(str(raw), fmt)
                    break
                except ValueError:
                    continue
            if naive is None:
                return None

        localized = tz.localize(naive)
        return iwdata.util.check_and_convert_datetime(cast(datetime, localized))


def arbin(
    filename: str | Path,
    extra_column_mappings: dict[str, str] | None = None,
    options: dict[str, str] | None = None,
) -> pl.DataFrame:
    return Arbin().run(
        filename, extra_column_mappings=extra_column_mappings, options=options
    )


class ArbinRes(Arbin):
    """Reader for Arbin native ``.res`` (MDB/Access) files.

    Requires ``mdb-export`` from the ``mdbtools`` package
    (``brew install mdbtools`` on macOS, ``apt-get install mdbtools`` on
    Linux). The underlying data pipeline is identical to :class:`Arbin`; this
    subclass exists so that ``reader="arbin res"`` can be passed explicitly to
    :func:`~ionworksdata.read.time_series` and friends.
    """

    name: str = "Arbin res"

    def run(
        self,
        filename: str | Path,
        extra_column_mappings: dict[str, str] | None = None,
        options: dict[str, str] | None = None,
    ) -> pl.DataFrame:
        """Read an Arbin ``.res`` file and return a DataFrame with standardized columns.

        Parameters
        ----------
        filename : str | Path
            Path to the ``.res`` file.
        extra_column_mappings : dict[str, str] | None, optional
            Unused for ``.res`` files (all columns are mapped automatically);
            present for API compatibility.
        options : dict[str, str] | None, optional
            Options:

                - timezone : str, optional
                    Timezone to assume for ``Start_DateTime``. Default ``"UTC"``.

        Returns
        -------
        pl.DataFrame
            Time series with standardized ionworks columns.

        Raises
        ------
        RuntimeError
            If ``mdb-export`` is not installed.
        """
        if Path(filename).suffix.lower() != ".res":
            raise ValueError(f"ArbinRes reader expects a .res file, got: {filename}")
        return super().run(
            filename, extra_column_mappings=extra_column_mappings, options=options
        )

    def read_start_time(
        self,
        filename: str | Path,
        extra_column_mappings: dict[str, str] | None = None,
        options: dict[str, str] | None = None,
    ):
        """Read the start time from the ``.res`` file's Global_Table.

        Parameters
        ----------
        filename : str | Path
            Path to the ``.res`` file.
        extra_column_mappings : dict[str, str] | None, optional
            Unused, present for API compatibility.
        options : dict[str, str] | None, optional
            Options containing the timezone string (default ``"UTC"``).

        Returns
        -------
        datetime | None
            Timezone-aware start time, or None if unavailable.
        """
        if Path(filename).suffix.lower() != ".res":
            raise ValueError(f"ArbinRes reader expects a .res file, got: {filename}")
        return super().read_start_time(
            filename, extra_column_mappings=extra_column_mappings, options=options
        )


def arbin_res(
    filename: str | Path,
    options: dict[str, str] | None = None,
) -> pl.DataFrame:
    """Read an Arbin ``.res`` file. Shorthand for ``ArbinRes().run(filename)``.

    Parameters
    ----------
    filename : str | Path
        Path to the ``.res`` file.
    options : dict[str, str] | None, optional
        Reader options (e.g. ``{"timezone": "America/New_York"}``).

    Returns
    -------
    pl.DataFrame
        Time series with standardized ionworks columns.
    """
    return ArbinRes().run(filename, options=options)
