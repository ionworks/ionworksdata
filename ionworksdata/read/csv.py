from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any
import polars as pl

import iwutil

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


def _find_column(
    data_columns: list[str], options: list[dict]
) -> tuple[str, float, float]:
    """
    Find the first column in a list of options that is present in a DataFrame.
    """
    for values_scale_shift in options:
        for column in values_scale_shift["values"]:
            if column in data_columns:
                return column, values_scale_shift["scale"], values_scale_shift["shift"]
    else:
        raise ValueError(f"Could not find appropriate column out of {options}")


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
        Read a CSV file and return a pandas DataFrame with appropriate column names.

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
        pandas.DataFrame
            Processed data from the CSV file with standardized column names and units. By
            default, only returns the columns "Time [s]", "Voltage [V]",
            "Current [A]",
            "Current [mA.cm-2]", and "Temperature [degC]".
        """
        options = iwutil.check_and_combine_options(self.default_options, options)

        # Read the data
        data = iwutil.read_df(filename)
        extra_column_mappings = extra_column_mappings or {}

        # Apply column mappings first, before any column detection
        data = data.rename(columns=extra_column_mappings)

        # remove spaces from column names to reduce the number of possible column names
        data.columns = [col.replace(" ", "") for col in data.columns]

        # Voltage
        voltage_column, voltage_scale, voltage_shift = _find_column(
            data.columns, VOLTAGE_COLUMNS
        )
        data["Voltage [V]"] = data[voltage_column] * voltage_scale + voltage_shift

        # Current
        try:
            current_column, current_scale, current_shift = _find_column(
                data.columns, CURRENT_COLUMNS
            )
            data["Current [A]"] = data[current_column] * current_scale + current_shift
        except ValueError:
            (
                areal_current_column,
                areal_current_scale,
                areal_current_shift,
            ) = _find_column(data.columns, AREAL_CURRENT_COLUMNS)
            data["Current [mA.cm-2]"] = (
                data[areal_current_column] * areal_current_scale + areal_current_shift
            )

        # Time
        time_column, time_scale, time_shift = _find_column(data.columns, TIME_COLUMNS)
        data["Time [s]"] = data[time_column] * time_scale + time_shift

        # Temperature
        try:
            temperature_column, temperature_scale, temperature_shift = _find_column(
                data.columns, TEMPERATURE_COLUMNS
            )
            data["Temperature [degC]"] = (
                data[temperature_column] * temperature_scale + temperature_shift
            )
        except ValueError:
            pass

        # Get standard and user-supplied columns first
        standard_cols = [
            "Time [s]",
            "Voltage [V]",
            "Current [A]",
            "Current [mA.cm-2]",
            "Temperature [degC]",
        ]
        user_cols = list(extra_column_mappings.values())

        # Restore mapped column names (the values from extra_column_mappings)
        # to preserve user-supplied mapped names, but avoid duplicates
        for mapped_col in extra_column_mappings.values():
            mapped_col_no_space = mapped_col.replace(" ", "")
            if mapped_col_no_space in data.columns and mapped_col not in data.columns:
                data = data.rename(columns={mapped_col_no_space: mapped_col})

        # Build the list of columns to keep:
        # 1. All standard columns present in data and not shadowed by user-mapped columns
        # 2. All user-mapped columns present in data and not already included
        columns_keep = []
        seen = set()
        for col in standard_cols + user_cols:
            if col in data.columns and col not in seen:
                columns_keep.append(col)
                seen.add(col)

        data_pl = pl.from_pandas(data)
        data_pl = self.standard_data_processing(data_pl, columns_keep=columns_keep)

        return data_pl

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
