"""
Core step analysis functions for battery cycling data.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import polars as pl

import ionworksdata as iwdata


def ocp_steps(time_series: pd.DataFrame | pl.DataFrame) -> pl.DataFrame:
    """
    Build a single-step steps DataFrame from OCP (open-circuit potential) data.

    Produces output compatible with :func:`summarize` but without requiring
    ``Time [s]``, ``Current [A]``, or energy columns.

    Parameters
    ----------
    time_series : pd.DataFrame | pl.DataFrame
        OCP data with at least ``Voltage [V]`` and ``Step count`` columns.
        ``Capacity [A.h]`` is used for capacity fields if present.

    Returns
    -------
    pl.DataFrame
        A single-row steps DataFrame with the same column schema as
        :func:`summarize` output.
    """
    if isinstance(time_series, pd.DataFrame):
        ts = pl.from_pandas(time_series)
    else:
        ts = time_series

    voltage = ts["Voltage [V]"]
    n = len(ts)

    has_cap = "Capacity [A.h]" in ts.columns
    if has_cap:
        cap = ts["Capacity [A.h]"]
        total_cap = float(cap[-1]) - float(cap[0])
    else:
        total_cap = 0.0

    v_std = voltage.std()

    step = {
        "Start index": 0,
        "End index": n - 1,
        "Start time [s]": 0.0,
        "End time [s]": 0.0,
        "Start voltage [V]": float(voltage[0]),
        "End voltage [V]": float(voltage[-1]),
        "Discharge capacity [A.h]": max(total_cap, 0.0),
        "Charge capacity [A.h]": max(-total_cap, 0.0),
        "Discharge energy [W.h]": 0.0,
        "Charge energy [W.h]": 0.0,
        "Min voltage [V]": float(voltage.min()),  # type: ignore[arg-type]
        "Max voltage [V]": float(voltage.max()),  # type: ignore[arg-type]
        "Std voltage [V]": float(v_std) if v_std is not None else 0.0,
        "Mean voltage [V]": float(voltage.mean()),  # type: ignore[arg-type]
        "Min current [A]": 0.0,
        "Max current [A]": 0.0,
        "Std current [A]": 0.0,
        "Mean current [A]": 0.0,
        "Min power [W]": 0.0,
        "Max power [W]": 0.0,
        "Std power [W]": 0.0,
        "Mean power [W]": 0.0,
        "Min frequency [Hz]": 0.0,
        "Max frequency [Hz]": 0.0,
        "Std frequency [Hz]": 0.0,
        "Mean frequency [Hz]": 0.0,
        "Step count": 0,
        "Step from cycler": 0,
        "Cycle count": 0,
        "Cycle from cycler": 0,
        "Duration [s]": 0.0,
        "Step type": "OCP",
        "Label": "",
        "Group number": float("nan"),
        "Cycle charge capacity [A.h]": max(-total_cap, 0.0),
        "Cycle discharge capacity [A.h]": max(total_cap, 0.0),
        "Cycle charge energy [W.h]": 0.0,
        "Cycle discharge energy [W.h]": 0.0,
    }

    return pl.DataFrame([step])


def summarize(data: pd.DataFrame | pl.DataFrame) -> pl.DataFrame:
    """
    Returns a DataFrame with information about each step in the data.

    Parameters
    ----------
    data : pd.DataFrame | pl.DataFrame
        The data to get the step types for. Must contain "Step count" column.
        If "Cycle from cycler" is present, it will be used to calculate cycle count.

    Returns
    -------
    pl.DataFrame
        A DataFrame with information about each step in the data. The output always
        includes a "Cycle count" column (defaults to 0 if no cycle information is
        available), "Cycle charge capacity [A.h]" and "Cycle discharge capacity [A.h]"
        columns, "Cycle charge energy [W.h]" and "Cycle discharge energy [W.h]" columns
        (if energy columns are present), and a "Cycle from cycler" column (only if
        provided in the input data).
    """
    steps_list = identify(data)
    steps_pl = pl.DataFrame(steps_list)
    steps_pl = set_cycle_capacity(steps_pl)
    steps_pl = set_cycle_energy(steps_pl)
    return steps_pl


def identify(time_series: pd.DataFrame | pl.DataFrame) -> list[dict]:
    """
    Identify individual steps in battery cycling data.

    This function processes a time series DataFrame and identifies distinct steps
    within battery cycling data by detecting changes in the "Step count" column.
    For each identified step, it extracts and calculates relevant metrics (voltage,
    current, capacity, etc.) and determines the step type.

    Parameters
    ----------
    time_series : pd.DataFrame | pl.DataFrame
        Battery cycling data with columns including "Step count", "Time [s]",
        'Voltage [V]', 'Current [A]', etc.

    Returns
    -------
    list[dict]
        List of dictionaries where each dictionary contains information about a step,
        including start/end indices, voltage, current, capacity, duration, and step
        type.
    """
    # Normalize to Polars
    if isinstance(time_series, pl.DataFrame):
        time_series_pl = time_series
    else:
        time_series_pl = pl.from_pandas(time_series)

    step_column = "Step count"

    # Ensure required columns
    if step_column not in time_series_pl.columns:
        raise KeyError(f"Missing required step column: {step_column}")
    if "Time [s]" not in time_series_pl.columns:
        raise KeyError("Missing required column: 'Time [s]'")
    if "Voltage [V]" not in time_series_pl.columns:
        raise KeyError("Missing required column: 'Voltage [V]'")

    # Ensure capacity present
    time_series_pl = iwdata.transform.set_capacity(time_series_pl)

    # Ensure frequency column exists (fill with zeros if not present)
    if "Frequency [Hz]" not in time_series_pl.columns:
        time_series_pl = time_series_pl.with_columns(
            pl.lit(0.0).alias("Frequency [Hz]")
        )

    # Compute power
    time_series_pl = time_series_pl.with_columns(
        (pl.col("Voltage [V]") * pl.col("Current [A]")).alias("Power [W]")
    )

    # Add a stable row index
    time_series_pl = time_series_pl.with_row_index("__row_id")

    # Identify contiguous step groups via cumulative sum of changes
    change_flag = (
        (pl.col(step_column) != pl.col(step_column).shift())
        .fill_null(True)
        .cast(pl.Int32)
    )
    time_series_pl = time_series_pl.with_columns(
        (change_flag.cum_sum() - 1).alias("__step_group")
    )

    # Cycle count per step (cumulative count of cycle changes at step boundaries)
    # Always use "Cycle from cycler" if it exists
    if "Cycle from cycler" in time_series_pl.columns:
        cycle_change = (
            (pl.col("Cycle from cycler") != pl.col("Cycle from cycler").shift())
            .fill_null(False)
            .cast(pl.Int64)
            .cum_sum()
        )
        time_series_pl = time_series_pl.with_columns(
            cycle_change.alias("__cycle_cumsum")
        )
    else:
        # If no cycle information, set cycle count to 0
        time_series_pl = time_series_pl.with_columns(
            pl.lit(0).cast(pl.Int64).alias("__cycle_cumsum")
        )

    # Aggregate per step group
    agg = time_series_pl.group_by("__step_group").agg(
        [
            pl.col("__row_id").min().alias("Start index"),
            pl.col("__row_id").max().alias("End index"),
            pl.col("Time [s]").first().alias("Start time [s]"),
            pl.col("Time [s]").last().alias("End time [s]"),
            pl.col("Voltage [V]").first().alias("Start voltage [V]"),
            pl.col("Voltage [V]").last().alias("End voltage [V]"),
            (
                pl.col("Discharge capacity [A.h]").last()
                - pl.col("Discharge capacity [A.h]").first()
            ).alias("Discharge capacity [A.h]"),
            (
                pl.col("Charge capacity [A.h]").last()
                - pl.col("Charge capacity [A.h]").first()
            ).alias("Charge capacity [A.h]"),
            (
                pl.col("Discharge energy [W.h]").last()
                - pl.col("Discharge energy [W.h]").first()
            ).alias("Discharge energy [W.h]")
            if "Discharge energy [W.h]" in time_series_pl.columns
            else pl.lit(0.0).alias("Discharge energy [W.h]"),
            (
                pl.col("Charge energy [W.h]").last()
                - pl.col("Charge energy [W.h]").first()
            ).alias("Charge energy [W.h]")
            if "Charge energy [W.h]" in time_series_pl.columns
            else pl.lit(0.0).alias("Charge energy [W.h]"),
            pl.col("Voltage [V]").min().alias("Min voltage [V]"),
            pl.col("Voltage [V]").max().alias("Max voltage [V]"),
            pl.col("Voltage [V]").std().fill_null(0.0).alias("Std voltage [V]"),
            pl.col("Voltage [V]").mean().alias("Mean voltage [V]"),
            pl.col("Current [A]").min().alias("Min current [A]"),
            pl.col("Current [A]").max().alias("Max current [A]"),
            pl.col("Current [A]").std().fill_null(0.0).alias("Std current [A]"),
            pl.col("Current [A]").mean().alias("Mean current [A]"),
            pl.col("Power [W]").min().alias("Min power [W]"),
            pl.col("Power [W]").max().alias("Max power [W]"),
            pl.col("Power [W]").std().fill_null(0.0).alias("Std power [W]"),
            pl.col("Power [W]").mean().alias("Mean power [W]"),
            pl.col("Frequency [Hz]").min().alias("Min frequency [Hz]"),
            pl.col("Frequency [Hz]").max().alias("Max frequency [Hz]"),
            pl.col("Frequency [Hz]").std().fill_null(0.0).alias("Std frequency [Hz]"),
            pl.col("Frequency [Hz]").mean().alias("Mean frequency [Hz]"),
            pl.col(step_column).first().alias(step_column),
            # Always include "Step from cycler" if it exists in the time series
            pl.col("Step from cycler").first().alias("Step from cycler")
            if "Step from cycler" in time_series_pl.columns
            else pl.lit(None).alias("Step from cycler"),
            # Always include "Cycle count" - default to 0 if no cycle information
            pl.col("__cycle_cumsum")
            .first()
            .fill_null(0)
            .cast(pl.Int64)
            .alias("Cycle count"),
            # Always include "Cycle from cycler" if it exists in the time series
            pl.col("Cycle from cycler").first().alias("Cycle from cycler")
            if "Cycle from cycler" in time_series_pl.columns
            else pl.lit(None).alias("Cycle from cycler"),
        ]
    )

    # Order by start index and assign step count
    agg = agg.sort("Start index")

    # Duration
    agg = agg.with_columns(
        (pl.col("End time [s]") - pl.col("Start time [s]")).alias("Duration [s]")
    )

    # Step type inference using thresholds from global settings
    current_tol = iwdata.settings.get_current_std_tol()
    voltage_tol = iwdata.settings.get_voltage_std_tol()
    power_tol = iwdata.settings.get_power_std_tol()
    rest_tol = iwdata.settings.get_rest_tol()
    eis_tol = iwdata.settings.get_eis_tol()
    step_type_expr = (
        pl.when(pl.col("Mean frequency [Hz]") > eis_tol)
        .then(pl.lit("EIS"))
        .when(pl.col("Mean current [A]").abs() < rest_tol)
        .then(pl.lit("Rest"))
        .when(pl.col("Std current [A]") < current_tol)
        .then(
            pl.when(pl.col("Mean current [A]") > 0)
            .then(pl.lit("Constant current discharge"))
            .otherwise(pl.lit("Constant current charge"))
        )
        .when(pl.col("Std voltage [V]").abs() < voltage_tol)
        .then(
            pl.when(pl.col("Mean current [A]") > 0)
            .then(pl.lit("Constant voltage discharge"))
            .otherwise(pl.lit("Constant voltage charge"))
        )
        .when(pl.col("Std power [W]").abs() < power_tol)
        .then(
            pl.when(pl.col("Mean current [A]") > 0)
            .then(pl.lit("Constant power discharge"))
            .otherwise(pl.lit("Constant power charge"))
        )
        .otherwise(pl.lit("Unknown step type"))
    )
    agg = agg.with_columns(step_type_expr.alias("Step type"))

    # Add placeholder columns to match legacy output
    agg = agg.with_columns(
        [
            pl.lit("").alias("Label"),
            pl.lit(float("nan")).alias("Group number"),
        ]
    )

    # Convert to list of dicts for backward compatibility with callers
    return agg.drop(["__step_group"]).to_dicts()


def set_cycle_capacity(steps: pl.DataFrame | dict) -> pl.DataFrame:
    """
    Calculate the cycle capacity for each step in the data.

    Cycles are identified by the "Cycle count" column.

    Parameters
    ----------
    steps : pl.DataFrame | dict
        A DataFrame with information about each step in the data.

    Returns
    -------
    pl.DataFrame
        The original DataFrame with the cycle capacity added.
    """
    # Convert to Polars if needed
    if isinstance(steps, dict):
        steps = pl.DataFrame(steps)

    if "Cycle count" not in steps.columns:
        return steps.with_columns(
            [
                pl.lit(None).cast(pl.Float64).alias("Cycle charge capacity [A.h]"),
                pl.lit(None).cast(pl.Float64).alias("Cycle discharge capacity [A.h]"),
            ]
        )

    # Calculate cycle capacities using group_by
    # Sum up the discharge and charge capacity from each step in the cycle
    cycle_capacities = steps.group_by("Cycle count").agg(
        [
            pl.col("Charge capacity [A.h]").sum().alias("Cycle charge capacity [A.h]"),
            pl.col("Discharge capacity [A.h]")
            .sum()
            .alias("Cycle discharge capacity [A.h]"),
        ]
    )

    # Join back to original steps
    steps = steps.join(cycle_capacities, on="Cycle count", how="left")

    return steps


def set_cycle_energy(steps: pl.DataFrame | dict) -> pl.DataFrame:
    """
    Calculate the cycle energy for each step in the data.

    Cycles are identified by the "Cycle count" column.

    Parameters
    ----------
    steps : pl.DataFrame | dict
        A DataFrame with information about each step in the data.

    Returns
    -------
    pl.DataFrame
        The original DataFrame with the cycle energy added.
    """
    # Convert to Polars if needed
    if isinstance(steps, dict):
        steps = pl.DataFrame(steps)

    if "Cycle count" not in steps.columns:
        return steps.with_columns(
            [
                pl.lit(None).cast(pl.Float64).alias("Cycle charge energy [W.h]"),
                pl.lit(None).cast(pl.Float64).alias("Cycle discharge energy [W.h]"),
            ]
        )

    # Check if energy columns exist
    has_charge_energy = "Charge energy [W.h]" in steps.columns
    has_discharge_energy = "Discharge energy [W.h]" in steps.columns

    if not has_charge_energy or not has_discharge_energy:
        # If energy columns don't exist, set cycle energy to None
        return steps.with_columns(
            [
                pl.lit(None).cast(pl.Float64).alias("Cycle charge energy [W.h]"),
                pl.lit(None).cast(pl.Float64).alias("Cycle discharge energy [W.h]"),
            ]
        )

    # Calculate cycle energies using group_by
    # Sum up the discharge and charge energy from each step in the cycle
    cycle_energies = steps.group_by("Cycle count").agg(
        [
            pl.col("Charge energy [W.h]").sum().alias("Cycle charge energy [W.h]"),
            pl.col("Discharge energy [W.h]")
            .sum()
            .alias("Cycle discharge energy [W.h]"),
        ]
    )

    # Join back to original steps
    steps = steps.join(cycle_energies, on="Cycle count", how="left")

    return steps


def infer_type(
    step: dict,
    current_std_tol: float | None = None,
    voltage_std_tol: float | None = None,
    power_std_tol: float | None = None,
    rest_tol: float | None = None,
    eis_tol: float | None = None,
) -> str:
    """
    Infer the type of step based on its metrics.

    Parameters
    ----------
    step : dict
        A dictionary containing the calculated metrics and properties for the step.
    current_std_tol : float, optional
        The tolerance for the standard deviation of the current below which the step
        is considered a constant current step. If None, uses the value from global
        settings.
    voltage_std_tol : float, optional
        The tolerance for the standard deviation of the voltage below which the step
        is considered a constant voltage step. If None, uses the value from global
        settings.
    power_std_tol : float, optional
        The tolerance for the standard deviation of the power below which the step is
        considered a constant power step. If None, uses the value from global settings.
    rest_tol : float, optional
        The tolerance for the absolute value of the current below which the step is
        considered a rest step. If None, uses the value from global settings.
    eis_tol : float, optional
        The tolerance for the absolute value of the frequency below which the step is
        considered an EIS step. If None, uses the value from global settings.

    Returns
    -------
    str
        The type of step.
    """
    # Use global settings if tolerances are not provided
    if current_std_tol is None:
        current_std_tol = iwdata.settings.get_current_std_tol()
    if voltage_std_tol is None:
        voltage_std_tol = iwdata.settings.get_voltage_std_tol()
    if power_std_tol is None:
        power_std_tol = iwdata.settings.get_power_std_tol()
    if rest_tol is None:
        rest_tol = iwdata.settings.get_rest_tol()
    if eis_tol is None:
        eis_tol = iwdata.settings.get_eis_tol()

    if step["Mean frequency [Hz]"] > eis_tol:
        return "EIS"
    elif np.abs(step["Mean current [A]"]) < rest_tol:
        return "Rest"
    elif np.abs(step["Std current [A]"]) < current_std_tol:
        if step["Mean current [A]"] > 0:
            return "Constant current discharge"
        else:
            return "Constant current charge"
    elif np.abs(step["Std voltage [V]"]) < voltage_std_tol:
        if step["Mean current [A]"] > 0:
            return "Constant voltage discharge"
        else:
            return "Constant voltage charge"
    elif np.abs(step["Std power [W]"]) < power_std_tol:
        if step["Mean current [A]"] > 0:
            return "Constant power discharge"
        else:
            return "Constant power charge"
    else:
        return "Unknown step type"


def _postprocess_step(
    time_series: pd.DataFrame,
    stop_index: int,
    start_index: int,
    current_step: int,
    cycle_count: int | None,
    cycle_column: str | None,
) -> dict:
    """
    Process a single battery cycling step to extract and calculate relevant metrics.

    This function takes a section of time series data identified as a single step and
    calculates various statistical measures and properties for that step, including
    voltage, current, power, duration, and step type.

    Parameters
    ----------
    time_series : pd.DataFrame
        The complete time series DataFrame containing battery cycling data.
    stop_index : int
        The ending index of the step in the time_series DataFrame.
    start_index : int
        The starting index of the step in the time_series DataFrame.
    current_step : int
        The step count identifier.
    cycle_count : int | None
        The cycle count (cumulative cycle number) this step belongs to, or None if
        cycles aren't tracked.
    cycle_column : str | None
        The column name containing cycle numbers, or None if cycles aren't tracked.

    Returns
    -------
    dict
        Dictionary containing calculated metrics and properties for the step including:
        - Start/end indices, times, voltages, and capacities
        - Min/max/std/mean values for voltage, current, and power
        - Duration
        - Step number, count, and type
        - If cycle_column is provided, the value of that column, plus the cycle count
    """
    step: dict[str, Any] = {}

    # Extract the relevant portion of time series data for this step
    step_details = time_series.iloc[start_index : stop_index + 1].copy()

    # Calculate power from voltage and current
    step_details["Power [W]"] = (
        step_details["Voltage [V]"] * step_details["Current [A]"]
    )

    # Record start and end indices
    step["Start index"] = start_index
    step["End index"] = stop_index

    # Extract start and end values for time and voltage
    start_end_cols = [
        "Time [s]",
        "Voltage [V]",
    ]
    for k in start_end_cols:
        lower_k = k[0].lower() + k[1:]
        step["Start " + lower_k] = step_details[k].iloc[0]
        step["End " + lower_k] = step_details[k].iloc[-1]

    # Calculate total discharge and charge capacity for this step
    step["Discharge capacity [A.h]"] = (
        step_details["Discharge capacity [A.h]"].iloc[-1]
        - step_details["Discharge capacity [A.h]"].iloc[0]
    )
    step["Charge capacity [A.h]"] = (
        step_details["Charge capacity [A.h]"].iloc[-1]
        - step_details["Charge capacity [A.h]"].iloc[0]
    )

    # Calculate total discharge and charge energy for this step
    if "Discharge energy [W.h]" in step_details.columns:
        step["Discharge energy [W.h]"] = (
            step_details["Discharge energy [W.h]"].iloc[-1]
            - step_details["Discharge energy [W.h]"].iloc[0]
        )
    else:
        step["Discharge energy [W.h]"] = 0.0

    if "Charge energy [W.h]" in step_details.columns:
        step["Charge energy [W.h]"] = (
            step_details["Charge energy [W.h]"].iloc[-1]
            - step_details["Charge energy [W.h]"].iloc[0]
        )
    else:
        step["Charge energy [W.h]"] = 0.0

    # Calculate statistical metrics for voltage, current, power, and frequency
    stat_columns = [
        "Voltage [V]",
        "Current [A]",
        "Power [W]",
        "Frequency [Hz]",
    ]
    min_vals = {
        k: step_details[k].min() if k in step_details.columns else 0
        for k in stat_columns
    }
    max_vals = {
        k: step_details[k].max() if k in step_details.columns else 0
        for k in stat_columns
    }
    std_vals = {
        k: np.nan_to_num(step_details[k].std(), nan=0.0)
        if k in step_details.columns
        else 0.0
        for k in stat_columns
    }
    mean_vals = {
        k: step_details[k].mean() if k in step_details.columns else 0
        for k in stat_columns
    }
    for k in stat_columns:
        lower_k = k[0].lower() + k[1:]
        # Frequency is not always present, so we set it to 0 if it's not present
        if k == "Frequency [Hz]" and "Frequency [Hz]" not in step_details.columns:
            step["Min " + lower_k] = 0
            step["Max " + lower_k] = 0
            step["Std " + lower_k] = 0
            step["Mean " + lower_k] = 0
        else:
            step["Min " + lower_k] = min_vals[k]
            step["Max " + lower_k] = max_vals[k]
            step["Std " + lower_k] = std_vals[k]
            step["Mean " + lower_k] = mean_vals[k]

    # Calculate step duration
    step["Duration [s]"] = step["End time [s]"] - step["Start time [s]"]

    # Record step count
    step["Step count"] = current_step

    # Infer the type of step based on the calculated metrics
    step["Step type"] = infer_type(step)

    # Record cycle count if available
    step["Cycle count"] = cycle_count
    # Log the original cycle_column if available
    if cycle_column is not None:
        step[cycle_column] = step_details[cycle_column].iloc[0]

    # Initialize label and group number (to be set by the labeler later)
    step["Label"] = ""
    step["Group number"] = np.nan

    return step


def annotate(
    time_series: pl.DataFrame | pd.DataFrame,
    steps: pl.DataFrame | pd.DataFrame,
    column_names: list[str],
) -> pl.DataFrame | pd.DataFrame:
    """
    Apply columns from the steps to the time series.

    Parameters
    ----------
    time_series : pl.DataFrame | pd.DataFrame
        The time series to apply the columns to.
    steps : pl.DataFrame | pd.DataFrame
        The steps to apply the columns from.
    column_names : list[str]
        The columns to apply from the steps to the time series.

    Returns
    -------
    pl.DataFrame | pd.DataFrame
        The time series with the columns applied. Returns the same type as the input
        time_series.
    """
    # Remember the input type
    return_polars = isinstance(time_series, pl.DataFrame)

    # Convert to pandas for row-wise slice assignment via .loc[start:end, col] = value
    # Polars doesn't support efficient in-place row slice assignment by index ranges.
    # This operation requires iterating over step ranges and updating corresponding
    # time series rows, which is a pandas strength.
    if isinstance(steps, pl.DataFrame):
        steps_pd = steps.to_pandas()
    else:
        steps_pd = steps

    if isinstance(time_series, pl.DataFrame):
        time_series_pd = time_series.to_pandas()
    else:
        time_series_pd = time_series

    time_series_pd = time_series_pd.copy()

    for _, row in steps_pd.iterrows():
        for column_name in column_names:
            time_series_pd.loc[row["Start index"] : row["End index"], column_name] = (
                row[column_name]
            )

    # Convert back to Polars if input was Polars
    if return_polars:
        return pl.from_pandas(time_series_pd)
    return time_series_pd
