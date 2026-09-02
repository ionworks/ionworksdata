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
import subprocess
from typing import Any, cast

import iwutil  # type: ignore[reportMissingTypeStubs]
import polars as pl
import pytz  # type: ignore[reportMissingTypeStubs]

import ionworksdata as iwdata

from ._utils import (
    _UNIT_SUFFIX_RE,
    find_data_sheet,
    list_sheet_names,
    read_excel_and_get_column_names,
    strip_unit_suffix,
    suppress_excel_dtype_warnings,
)
from .read import BaseReader

# OLE Automation date epoch used by Arbin .res files.
_OLE_EPOCH = datetime(1899, 12, 30)

# Carries the pre-rebase test time through ``standard_data_processing``.
_RAW_TIME_COL = "_arbin_raw_time"

# Sort key for placing a sweep among the cycling rows; internal to the reader.
_EIS_TEST_TIME_COL = "_eis_test_time"


# Wall-clock limit for a single ``mdb-export`` invocation. Generous enough for
# large real-world .res files, but bounded so a malformed or locked file can
# never wedge the process (or CI) indefinitely.
_MDB_EXPORT_TIMEOUT_S = 300


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
        If ``mdb-export`` is not found on PATH, or if it fails to finish within
        the timeout (``mdb-export`` can spin indefinitely on a malformed or
        locked MDB file).
    subprocess.CalledProcessError
        If ``mdb-export`` exits with a non-zero status.
    """
    try:
        result = subprocess.run(
            ["mdb-export", str(filename), table],
            capture_output=True,
            text=True,
            check=True,
            timeout=_MDB_EXPORT_TIMEOUT_S,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Could not locate the 'mdb-export' executable. "
            "Install mdbtools to read .res files: brew install mdbtools "
            "(macOS) or apt-get install mdbtools (Linux)."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"'mdb-export' timed out after {_MDB_EXPORT_TIMEOUT_S}s reading "
            f"table {table!r} from {filename}. The file may be corrupt or "
            "locked."
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
    match = _UNIT_SUFFIX_RE.match(col)
    if match is None:
        return strip_unit_suffix(col), None
    return strip_unit_suffix(match.group("base")), match.group("unit").strip()


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
    "surface temp": "Temperature [degC]",
    "cycle index": "Cycle from cycler",
    "step index": "Step from cycler",
    "date time": "Timestamp",
}

# Matched by prefix because the degree sign can arrive in a non-UTF-8 codepage
# (``Aux_Temperature(¡æ)_1``), making the unit useless as a lookup key.
_ARBIN_AUX_TEMPERATURE_PREFIX = "aux temperature"

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


# Signature columns shared by the CSV and Excel sniffers. Distinguishes Arbin
# from Maccor, which names these "Step" and "Cycle" without the "Index" suffix.
ARBIN_SIGNATURE_COLUMNS = ("Cycle Index", "Step Index")

# ``ACIM_*`` holds EIS sweeps and ``Statistics*`` vendor cycle/step summaries;
# neither is a time series.
_ARBIN_EIS_SHEET_PREFIX = "acim"
_ARBIN_NON_DATA_SHEET_PREFIXES = ("global_info", _ARBIN_EIS_SHEET_PREFIX, "statistics")

# A sheet must expose voltage and current to count as the time series.
_ARBIN_DATA_SHEET_REQUIRED = ("voltage", "current")


def find_arbin_data_sheet(filename: str | Path) -> str | None:
    """Return the name of the worksheet holding the Arbin time series.

    Arbin MITS workbooks put test metadata on a leading ``Global_Info`` sheet,
    so reading the first sheet yields four rows of header text rather than the
    measurement.

    Parameters
    ----------
    filename : str | Path
        Path to the Arbin ``.xls``/``.xlsx`` workbook.

    Returns
    -------
    str | None
        Name of the data sheet, or None if no sheet exposes voltage and
        current columns.
    """
    return find_data_sheet(
        filename,
        required_columns=_ARBIN_DATA_SHEET_REQUIRED,
        skip_prefixes=_ARBIN_NON_DATA_SHEET_PREFIXES,
        prefer_prefix="channel",
    )


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
    claimed: set[str] = set()
    for col in columns:
        base, unit = _strip_unit(col)
        target = _ARBIN_COLUMN_MAP.get(base)
        if target is None and base.startswith(_ARBIN_AUX_TEMPERATURE_PREFIX):
            # Mojibaked degree sign in the unit, e.g. ``Aux_Temperature(¡æ)_1``.
            target = "Temperature [degC]"
        if target is None or target in claimed:
            continue
        # Tracked here, not left to ``resolve_renamings``, so that a losing
        # column cannot contribute a unit scale.
        claimed.add(target)
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
        "include_eis": True,
    }

    @classmethod
    def sniff_excel(cls, filename: str | Path) -> bool:
        """Return True if *filename* is an Arbin ``.xls``/``.xlsx`` export.

        The data sheet is resolved first, because an Arbin MITS workbook leads
        with a ``Global_Info`` metadata sheet whose headers carry none of the
        signature columns.

        Parameters
        ----------
        filename : str | Path
            Path to the workbook to inspect.

        Returns
        -------
        bool
            True when the data sheet carries a test-time column alongside a
            cycle or step index.
        """
        try:
            _df, column_names = read_excel_and_get_column_names(
                Path(filename), sheet_name=find_arbin_data_sheet(filename)
            )
        except Exception:
            return False
        # Fold the underscored dialect (Test_Time(s), Cycle_Index) onto the
        # spaced one so a single set of names matches both.
        normalized = {strip_unit_suffix(c) for c in column_names}
        if not any(c.startswith("test time") for c in normalized):
            return False
        if all(s.lower() in normalized for s in ARBIN_SIGNATURE_COLUMNS):
            return True
        # A one-cycle export omits "Cycle Index". Still unambiguous: Maccor
        # names these columns "Step" and "Test Time (sec)".
        return "step index" in normalized

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
            sheet_name = find_arbin_data_sheet(filename)
            with suppress_excel_dtype_warnings():
                df = pl.read_excel(filename, sheet_name=sheet_name)
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
            present = iwdata.util.resolve_renamings(_ARBIN_RES_COLUMN_MAP, df)
            if present:
                df = df.rename(present)
            # Drop the OLE datetime sentinel — start time is read separately.
            if "_ole_datetime" in df.columns:
                df = df.drop("_ole_datetime")
            scales: dict[str, float] = {}
        else:
            renamings, scales = _build_renamings(df.columns)
            scale_source = {target: src for src, target in renamings.items()}
            renamings.update(extra_column_mappings or {})
            present = iwdata.util.resolve_renamings(
                renamings, df, priority=extra_column_mappings
            )
            if present:
                df = df.rename(present)
            # A scale belongs to the header it came from, not to a caller
            # column that outranked it and is already in SI.
            winner = {target: src for src, target in present.items()}
            scales = {
                target: factor
                for target, factor in scales.items()
                if winner.get(target) == scale_source.get(target)
            }

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

        # ``drop_nulls`` can remove the leading row, and ``reset_time`` then
        # rebases from a later one — so read the origin after processing.
        if "Time [s]" in df.columns:
            df = df.with_columns(pl.col("Time [s]").alias(_RAW_TIME_COL))

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
        if _RAW_TIME_COL in df.columns:
            columns_keep.append(_RAW_TIME_COL)

        df = self.standard_data_processing(df, columns_keep=columns_keep)

        raw_start_time = (
            float(df[_RAW_TIME_COL][0])
            if _RAW_TIME_COL in df.columns and df.height > 0
            else 0.0
        )
        df = df.drop(_RAW_TIME_COL, strict=False)

        if options.get("include_eis", True) and ext in (".xls", ".xlsx"):
            df = _interleave_arbin_eis(df, filename, raw_start_time)

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
            df, _cols = read_excel_and_get_column_names(
                Path(filename), sheet_name=find_arbin_data_sheet(filename)
            )
            if df is None or df.height == 0:
                return None
            # Both header dialects: "Date Time" and "Date_Time".
            timestamp_col = next(
                (c for c in df.columns if _strip_unit(str(c))[0] == "date time"),
                None,
            )
            if timestamp_col is None:
                return None
            raw = df[timestamp_col][0]
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


_ARBIN_EIS_COLUMN_MAP: dict[str, str] = {
    "test time": _EIS_TEST_TIME_COL,
    "frequency": "Frequency [Hz]",
    "zreal": "Z_Re [Ohm]",
    "zimg": "Z_Im [Ohm]",
    "zmod": "Z_Mod [Ohm]",
    "zphz": "Z_Phase [deg]",
    "step id": "Step from cycler",
    "cycle id": "Cycle from cycler",
}


def find_arbin_eis_sheet(filename: str | Path) -> str | None:
    """Return the name of the worksheet holding the Arbin EIS sweeps.

    Parameters
    ----------
    filename : str | Path
        Path to the Arbin ``.xls``/``.xlsx`` workbook.

    Returns
    -------
    str | None
        Name of the ``ACIM_*`` sheet, or None if the workbook has none.
    """
    return next(
        (
            n
            for n in list_sheet_names(filename)
            if n.strip().lower().startswith(_ARBIN_EIS_SHEET_PREFIX)
        ),
        None,
    )


class ArbinEIS(Arbin):
    """Reader for the EIS sweeps on an Arbin workbook's ``ACIM_*`` sheet.

    Returns the sweeps on their own. To read them in place within the cycling
    data, use the base :class:`Arbin` reader, which interleaves them by default
    (see ``include_eis``).

    Arbin reports ``Zimg`` as the raw imaginary part, so it becomes
    ``Z_Im [Ohm]`` unchanged — unlike BioLogic, which reports a negated
    ``-Im(Z)/Ohm`` that its reader flips. Mirroring that flip here would invert
    every Nyquist plot, so a test pins the convention against the phase Arbin
    itself reports.
    """

    name: str = "Arbin EIS"

    @staticmethod
    def _read_file(filename: str | Path) -> pl.DataFrame:
        """Read the ``ACIM_*`` sheet of an Arbin workbook.

        Parameters
        ----------
        filename : str | Path
            Path to the Arbin ``.xls``/``.xlsx`` workbook.

        Returns
        -------
        pl.DataFrame
            Raw DataFrame with the original ACIM column names preserved.

        Raises
        ------
        ValueError
            If the workbook has no ``ACIM_*`` sheet.
        """
        sheet_name = find_arbin_eis_sheet(filename)
        if sheet_name is None:
            raise ValueError(
                f"No EIS sheet found in {filename}. Arbin writes impedance data "
                "to a sheet named 'ACIM_*'; this workbook has none."
            )
        with suppress_excel_dtype_warnings():
            return pl.read_excel(filename, sheet_name=sheet_name)

    def run(
        self,
        filename: str | Path,
        extra_column_mappings: dict[str, str] | None = None,
        options: dict[str, str] | None = None,
    ) -> pl.DataFrame:
        """Read Arbin EIS sweeps and return a DataFrame with standardized columns.

        Parameters
        ----------
        filename : str | Path
            Path to the Arbin ``.xls``/``.xlsx`` workbook.
        extra_column_mappings : dict[str, str] | None, optional
            Additional raw → ionworks column mappings applied after the
            built-in ones.
        options : dict[str, str] | None, optional
            Options:

                - timezone : str, optional
                    Timezone to assume for the workbook. Default ``"UTC"``.
                - cell_metadata : dict, optional
                    Reserved for caller-supplied cell metadata.

        Returns
        -------
        pl.DataFrame
            EIS sweeps with ``Frequency [Hz]``, ``Z_Re [Ohm]``, ``Z_Im [Ohm]``,
            ``Z_Mod [Ohm]``, ``Z_Phase [deg]`` and the cycler's step/cycle
            indices. ``Time [s]`` is null: a sweep is measured against
            frequency, not the clock.

        Raises
        ------
        ValueError
            If the file is not an Excel workbook, or has no ``ACIM_*`` sheet.
        """
        if Path(filename).suffix.lower() not in (".xls", ".xlsx"):
            raise ValueError(
                f"ArbinEIS reader expects a .xls or .xlsx workbook, got: {filename}"
            )
        iwutil.check_and_combine_options(self.default_options, options)

        df = self._read_file(filename)
        eis = _standardize_arbin_eis(df, extra_column_mappings)
        # Only the interleave path needs the raw test time, as a sort key.
        return eis.drop(_EIS_TEST_TIME_COL, strict=False)


def _standardize_arbin_eis(
    df: pl.DataFrame,
    extra_column_mappings: dict[str, str] | None = None,
) -> pl.DataFrame:
    """Map raw ``ACIM_*`` columns onto the ionworks EIS vocabulary.

    Parameters
    ----------
    df : pl.DataFrame
        Raw ACIM sheet.
    extra_column_mappings : dict[str, str] | None, optional
        Additional raw → ionworks column mappings.

    Returns
    -------
    pl.DataFrame
        Sweeps with ionworks column names, ordered by the raw test time, with
        ``Time [s]`` null.
    """
    renamings = {
        col: target
        for col in df.columns
        if (target := _ARBIN_EIS_COLUMN_MAP.get(_strip_unit(col)[0])) is not None
    }
    renamings.update(extra_column_mappings or {})
    present = iwdata.util.resolve_renamings(
        renamings, df, priority=extra_column_mappings
    )
    if present:
        df = df.rename(present)

    if _EIS_TEST_TIME_COL in df.columns:
        df = df.sort(_EIS_TEST_TIME_COL)

    numeric = [
        "Frequency [Hz]",
        "Z_Re [Ohm]",
        "Z_Im [Ohm]",
        "Z_Mod [Ohm]",
        "Z_Phase [deg]",
    ]
    df = df.with_columns(
        [pl.col(c).cast(pl.Float64, strict=False) for c in numeric if c in df.columns]
    )
    for col in ("Step from cycler", "Cycle from cycler"):
        if col in df.columns:
            df = df.with_columns(pl.col(col).cast(pl.Int64, strict=False))

    keep = [
        c
        for c in [
            _EIS_TEST_TIME_COL,
            "Frequency [Hz]",
            "Z_Re [Ohm]",
            "Z_Im [Ohm]",
            "Z_Mod [Ohm]",
            "Z_Phase [deg]",
            "Step from cycler",
            "Cycle from cycler",
        ]
        if c in df.columns
    ]
    return df.select(keep).with_columns(
        pl.lit(None, dtype=pl.Float64).alias("Time [s]")
    )


def arbin_eis(
    filename: str | Path,
    extra_column_mappings: dict[str, str] | None = None,
    options: dict[str, str] | None = None,
) -> pl.DataFrame:
    """Read the EIS sweeps from an Arbin workbook's ``ACIM_*`` sheet.

    Parameters
    ----------
    filename : str | Path
        Path to the Arbin ``.xls``/``.xlsx`` workbook.
    extra_column_mappings : dict[str, str] | None, optional
        Additional raw → ionworks column mappings.
    options : dict[str, str] | None, optional
        Reader options (e.g. ``{"timezone": "America/New_York"}``).

    Returns
    -------
    pl.DataFrame
        EIS sweeps with standardized ionworks columns and a null ``Time [s]``.
    """
    return ArbinEIS().run(
        filename, extra_column_mappings=extra_column_mappings, options=options
    )


def _interleave_arbin_eis(
    data: pl.DataFrame,
    filename: str | Path,
    raw_start_time: float,
) -> pl.DataFrame:
    """Splice a workbook's EIS sweeps into its processed time series.

    The sheets are disjoint — the sweeps carry a step index the time series
    never mentions, and their test times fall between its logging ticks — so
    placing them by test time inserts rather than merges.

    Parameters
    ----------
    data : pl.DataFrame
        Processed time series, with ``Time [s]`` already rebased to zero.
    filename : str | Path
        Path to the Arbin workbook.
    raw_start_time : float
        The workbook's first raw test time, used to rebase the sweeps' test
        times onto the same origin before positioning them.

    Returns
    -------
    pl.DataFrame
        The time series with the sweeps inserted in test-time order, or the
        input unchanged when the workbook holds no EIS sheet.
    """
    sheet_name = find_arbin_eis_sheet(filename)
    if sheet_name is None:
        return data

    with suppress_excel_dtype_warnings():
        raw = pl.read_excel(filename, sheet_name=sheet_name)
    eis = _standardize_arbin_eis(raw)
    if eis.height == 0 or _EIS_TEST_TIME_COL not in eis.columns:
        return data
    position = eis.get_column(_EIS_TEST_TIME_COL) - raw_start_time
    eis = eis.drop(_EIS_TEST_TIME_COL)

    # A sort key, so EIS rows keep a null ``Time [s]``. Stable sort plus
    # sweeps-last places a tie after the sample it followed.
    data = data.with_columns(pl.col("Time [s]").alias("_position"))
    eis = eis.with_columns(position.alias("_position"))

    combined = pl.concat([data, eis], how="diagonal")
    combined = combined.sort("_position", nulls_last=True, maintain_order=True)
    combined = combined.drop("_position")

    if "Step from cycler" in combined.columns:
        combined = iwdata.transform.set_step_count(combined)
    if "Cycle from cycler" in combined.columns:
        combined = iwdata.transform.set_cycle_count(combined)

    return combined
