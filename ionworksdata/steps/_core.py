"""
Core step analysis functions for battery cycling data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl

import ionworksdata as iwdata


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

    # Require at least one x-axis: Time, Capacity, Stoichiometry, or SOC
    if step_column not in time_series_pl.columns:
        raise KeyError(f"Missing required step column: {step_column}")
    has_time = "Time [s]" in time_series_pl.columns
    has_capacity = "Capacity [A.h]" in time_series_pl.columns
    has_stoich = "Stoichiometry" in time_series_pl.columns
    has_soc = "SOC" in time_series_pl.columns
    has_x_axis = has_time or has_capacity or has_stoich or has_soc
    if not has_x_axis:
        raise KeyError(
            "Need at least one of 'Time [s]', 'Capacity [A.h]', "
            "'Stoichiometry', or 'SOC'"
        )
    if "Voltage [V]" not in time_series_pl.columns:
        raise KeyError("Missing required column: 'Voltage [V]'")

    # Current required by set_capacity and Power; use 1.0 so single Capacity = discharge
    if "Current [A]" not in time_series_pl.columns:
        time_series_pl = time_series_pl.with_columns(pl.lit(1.0).alias("Current [A]"))

    # Ensure capacity-derived columns present only when Capacity [A.h] exists
    if has_capacity:
        time_series_pl = iwdata.transform.set_capacity(time_series_pl)

    # Compute step capacity aggs when we have Discharge/Charge columns (from set_capacity here or upstream)
    has_capacity_cols = (
        "Discharge capacity [A.h]" in time_series_pl.columns
        and "Charge capacity [A.h]" in time_series_pl.columns
    )

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

    # Aggregate per step group (time-derived cols are null when Time [s] not present)
    agg_exprs = [
        pl.col("__row_id").min().alias("Start index"),
        pl.col("__row_id").max().alias("End index"),
        pl.col("Time [s]").first().alias("Start time [s]")
        if has_time
        else pl.lit(None).cast(pl.Float64).alias("Start time [s]"),
        pl.col("Time [s]").last().alias("End time [s]")
        if has_time
        else pl.lit(None).cast(pl.Float64).alias("End time [s]"),
    ]
    capacity_aggs = (
        [
            (
                pl.col("Discharge capacity [A.h]").last()
                - pl.col("Discharge capacity [A.h]").first()
            ).alias("Discharge capacity [A.h]"),
            (
                pl.col("Charge capacity [A.h]").last()
                - pl.col("Charge capacity [A.h]").first()
            ).alias("Charge capacity [A.h]"),
        ]
        if has_capacity_cols
        else [
            pl.lit(None).cast(pl.Float64).alias("Discharge capacity [A.h]"),
            pl.lit(None).cast(pl.Float64).alias("Charge capacity [A.h]"),
        ]
    )
    agg_exprs.extend(
        [
            pl.col("Voltage [V]").first().alias("Start voltage [V]"),
            pl.col("Voltage [V]").last().alias("End voltage [V]"),
            *capacity_aggs,
            (
                pl.col("Discharge energy [W.h]").last()
                - pl.col("Discharge energy [W.h]").first()
            ).alias("Discharge energy [W.h]")
            if "Discharge energy [W.h]" in time_series_pl.columns
            else pl.lit(None).cast(pl.Float64).alias("Discharge energy [W.h]")
            if not has_capacity_cols
            else pl.lit(0.0).alias("Discharge energy [W.h]"),
            (
                pl.col("Charge energy [W.h]").last()
                - pl.col("Charge energy [W.h]").first()
            ).alias("Charge energy [W.h]")
            if "Charge energy [W.h]" in time_series_pl.columns
            else pl.lit(None).cast(pl.Float64).alias("Charge energy [W.h]")
            if not has_capacity_cols
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
    agg = time_series_pl.group_by("__step_group").agg(agg_exprs)

    # Order by start index and assign step count
    agg = agg.sort("Start index")

    # Duration (null when Time [s] was not present, so DB gets explicit NULL)
    if has_time:
        agg = agg.with_columns(
            (pl.col("End time [s]") - pl.col("Start time [s]")).alias("Duration [s]")
        )
    else:
        agg = agg.with_columns(pl.lit(None).cast(pl.Float64).alias("Duration [s]"))

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


def annotate(
    time_series: pl.DataFrame | pd.DataFrame,
    steps: pl.DataFrame | pd.DataFrame,
    column_names: list[str],
) -> pl.DataFrame:
    """
    Apply columns from the steps table to the time series.

    Each time-series row is assigned the step's value for each requested column,
    based on step "Start index" and "End index" (inclusive). Use this to attach
    step-level info (e.g. "Step count", "Label", "Step type") to every row for
    downstream transforms or filtering.

    Parameters
    ----------
    time_series : pl.DataFrame | pd.DataFrame
        The time series to annotate. Row indices are 0 to n-1; steps must use
        the same index space (e.g. slice coordinates).
    steps : pl.DataFrame | pd.DataFrame
        Steps with "Start index" and "End index" and the columns in column_names.
    column_names : list[str]
        Columns to copy from steps onto the time series.

    Returns
    -------
    pl.DataFrame
        The time series with the requested step columns added.
    """
    if isinstance(time_series, pd.DataFrame):
        time_series_pl = pl.from_pandas(time_series)
    else:
        time_series_pl = time_series
    if isinstance(steps, pd.DataFrame):
        steps_pl = pl.from_pandas(steps)
    else:
        steps_pl = steps

    time_series_pl = time_series_pl.with_row_index("__row_id")

    # Build a long-format frame: __row_id and one column per column_name
    # Each step contributes rows from Start index to End index (inclusive) with
    # the step's values for column_names.
    parts: list[pl.DataFrame] = []
    for row in steps_pl.iter_rows(named=True):
        start = int(row["Start index"])
        end = int(row["End index"])
        row_ids = pl.Series("__row_id", range(start, end + 1))
        cols = [row_ids]
        for col_name in column_names:
            val = row[col_name]
            cols.append(pl.Series(col_name, [val] * (end - start + 1)))
        parts.append(pl.DataFrame(cols))
    if not parts:
        for col_name in column_names:
            time_series_pl = time_series_pl.with_columns(pl.lit(None).alias(col_name))
        return time_series_pl.drop("__row_id")

    steps_by_row = pl.concat(parts)

    time_series_pl = time_series_pl.join(steps_by_row, on="__row_id", how="left")
    return time_series_pl.drop("__row_id")
