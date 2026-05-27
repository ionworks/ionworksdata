from __future__ import annotations

from pathlib import Path
from typing import Any
import warnings

import iwutil
import polars as pl

from .read import BaseReader

VOLTAGE_COLUMNS = [
    {
        "values": [
            "Voltage[V]",
            "Voltage(V)",
            "Ewe/V",
            "Ewe_V_vs_Li",
            "<V>/V",
            "Ecell/V",
            "Potential[V]",
            "Potential(V)",
            "voltage__V",
            "h_potential__V",
        ],
        "scale": 1,
        "shift": 0,
    },
    {
        "values": ["Voltage[mV]", "Voltage(mV)", "Ewe/mV", "<V>/mV"],
        "scale": 1e-3,
        "shift": 0,
    },
]

CURRENT_COLUMNS = [
    {
        "values": [
            "Current[A]",
            "Current(A)",
            "I/A",
            "<I>/A",
            "current__A",
            "h_current__A",
        ],
        "scale": 1,
        "shift": 0,
    },
    {
        "values": ["Current[mA]", "Current(mA)", "I/mA", "<I>/mA"],
        "scale": 1e-3,
        "shift": 0,
    },
]
AREAL_CURRENT_COLUMNS = [
    {
        "values": ["Current[mA.cm-2]", "Current[mA/cm2]", "icell_mA_cm2"],
        "scale": 1,
        "shift": 0,
    }
]

TIME_COLUMNS = [
    {
        "values": [
            "Time[s]",
            "Time(s)",
            "t_s",
            "time/s",
            "TestTime(s)",
            "time__s",
            "h_test_time__s",
        ],
        "scale": 1,
        "shift": 0,
    },
    {"values": ["Time[h]", "Time(h)", "t_h", "time/h"], "scale": 3600, "shift": 0},
]

TEMPERATURE_COLUMNS = [
    {
        "values": [
            "Temperature[°C]",
            "Temperature(°C)",
            "Temperature[degC]",
            "Temperature[C]",
            "Temperature(C)",
            "Temp(°C)",
            "Temp(C)",
            "Temp[degC]",
            "T_C",
            "[Neware_xls]T10",
            "[Neware_xls]T490",
            "temperature__C",
            "h_temperature__°C",
        ],
        "scale": 1,
        "shift": 0,
    },
    {
        "values": [
            "Temperature[K]",
            "Temperature(K)",
            "Temp(K)",
            "T_K",
        ],
        "scale": 1,
        "shift": -273.15,
    },
]


def find_column(
    data_columns: list[str], options: list[dict]
) -> tuple[str, float, float]:
    """Find the first column in a list of options that is present in a DataFrame."""
    for values_scale_shift in options:
        for column in values_scale_shift["values"]:
            if column in data_columns:
                return column, values_scale_shift["scale"], values_scale_shift["shift"]
    raise ValueError(f"Could not find appropriate column out of {options}")


# Canonical columns every reader detects after applying caller mappings.
STANDARD_CANONICAL_COLUMNS = [
    "Time [s]",
    "Voltage [V]",
    "Current [A]",
    "Current [mA.cm-2]",
    "Temperature [degC]",
]


def apply_canonical_detection(applier, mapped_targets: set[str]) -> None:
    """Walk the standard canonical-column detection plan.

    Calls ``applier(canonical_name, alias_options) -> bool`` for each
    canonical column not already supplied by the caller. ``applier`` should
    add the column to the underlying frame and return whether a match was
    found. ``mapped_targets`` is the set of canonical names the caller has
    already supplied (so detection skips them).

    Raises ``ValueError`` if Voltage or Time can be found from neither an
    alias nor ``mapped_targets`` — both are required by every downstream
    consumer, so silent omission would surface later as a much less
    actionable ``KeyError``. Current accepts an areal-current fallback;
    Temperature is optional.
    """

    def needs(target: str) -> bool:
        return target not in mapped_targets

    if needs("Voltage [V]") and not applier("Voltage [V]", VOLTAGE_COLUMNS):
        raise ValueError(f"Could not find a Voltage column out of {VOLTAGE_COLUMNS}")

    current_found = not needs("Current [A]") or applier("Current [A]", CURRENT_COLUMNS)
    if not current_found:
        if not needs("Current [mA.cm-2]"):
            current_found = True
        else:
            current_found = applier("Current [mA.cm-2]", AREAL_CURRENT_COLUMNS)
    if not current_found:
        raise ValueError(
            "Could not find a Current column out of "
            f"{CURRENT_COLUMNS + AREAL_CURRENT_COLUMNS}"
        )

    if needs("Time [s]") and not applier("Time [s]", TIME_COLUMNS):
        raise ValueError(f"Could not find a Time column out of {TIME_COLUMNS}")

    if needs("Temperature [degC]"):
        applier("Temperature [degC]", TEMPERATURE_COLUMNS)


def build_columns_keep(
    present_columns: list[str], extra_column_mappings: dict[str, str]
) -> list[str]:
    """Build the ordered list of columns to keep after detection.

    Standard canonical columns come first, then any caller-supplied mapped
    columns that aren't already in the standard set. Duplicates are removed.
    """
    user_cols = list(extra_column_mappings.values())
    columns_keep: list[str] = []
    seen: set[str] = set()
    for col in STANDARD_CANONICAL_COLUMNS + user_cols:
        if col in present_columns and col not in seen:
            columns_keep.append(col)
            seen.add(col)
    return columns_keep


def detect_canonical(data: pl.DataFrame, mapped_targets: set[str]) -> pl.DataFrame:
    """Add canonical Voltage/Current/Time/Temperature columns from aliases.

    Alias keys (in ``VOLTAGE_COLUMNS`` etc.) are space-stripped, so column
    detection matches against a stripped lookup; the original column name on
    the frame is unchanged. Targets the caller has already supplied via
    ``mapped_targets`` are skipped.
    """
    stripped_map = {c.replace(" ", ""): c for c in data.columns}
    stripped_cols = list(stripped_map.keys())
    new_exprs: list[pl.Expr] = []

    def applier(target: str, alias_options: list[dict]) -> bool:
        try:
            src_stripped, scale, shift = find_column(stripped_cols, alias_options)
        except ValueError:
            return False
        src = stripped_map[src_stripped]
        expr = (
            pl.col(src) * scale + shift if (scale != 1 or shift != 0) else pl.col(src)
        )
        new_exprs.append(expr.alias(target))
        return True

    apply_canonical_detection(applier, mapped_targets)
    return data.with_columns(new_exprs) if new_exprs else data


def synthesize_current_a_from_ma(data: pl.DataFrame) -> pl.DataFrame:
    """If the frame has ``Current [mA]`` but no ``Current [A]``, derive the
    canonical ampere column. Otherwise return the frame unchanged.
    """
    if "Current [A]" not in data.columns and "Current [mA]" in data.columns:
        return data.with_columns((pl.col("Current [mA]") / 1000.0).alias("Current [A]"))
    return data


class CSV(BaseReader):
    name: str = "CSV"
    default_options: dict[str, Any] = {
        "cell_metadata": {},
    }

    def run(
        self,
        filename: str | Path,
        extra_column_mappings: dict[str, str] | None = None,
        options: dict[str, str] | None = None,
    ) -> pl.DataFrame:
        """
        Read a CSV file and return a Polars DataFrame with appropriate column names.

        Parameters
        ----------
        filename : str | Path
            Path to the CSV file to be read.
        extra_column_mappings : dict[str, str] | None, optional
            Dictionary of additional column mappings to use when reading the CSV file.
            The keys are the original column names and the values are the new column
            names. Default is None.
        options : dict[str, str] | None, optional
            Dictionary of options to use when reading the CSV file.

            Options are:

                - cell_metadata: dict, optional
                    Additional metadata about the cell. Default is empty dict.

        Returns
        -------
        pl.DataFrame
            Processed data from the CSV file with standardized column names and units. By
            default, only returns the columns "Time [s]", "Voltage [V]",
            "Current [A]",
            "Current [mA.cm-2]", and "Temperature [degC]".
        """
        options = iwutil.check_and_combine_options(self.default_options, options)
        extra_column_mappings = extra_column_mappings or {}

        data_pd = iwutil.read_df(filename).rename(columns=extra_column_mappings)
        data = pl.from_pandas(data_pd)
        data = detect_canonical(data, set(extra_column_mappings.values()))
        data = synthesize_current_a_from_ma(data)
        columns_keep = build_columns_keep(data.columns, extra_column_mappings)
        return self.standard_data_processing(data, columns_keep=columns_keep)

    def read_start_time(
        self,
        filename: str | Path,
        extra_column_mappings: dict[str, str] | None = None,
        options: dict[str, str] | None = None,
    ) -> None:
        warnings.warn(
            "CSV reader does not support reading start time from file",
            stacklevel=2,
        )
        return None


def csv(
    filename: str | Path,
    extra_column_mappings: dict[str, str] | None = None,
    options: dict[str, str] | None = None,
) -> pl.DataFrame:
    return CSV().run(
        filename, extra_column_mappings=extra_column_mappings, options=options
    )
