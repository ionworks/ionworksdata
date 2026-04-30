from __future__ import annotations

from ionworks.validators import positive_current_is_charge  # noqa: F401
import iwutil
import numpy as np
import pandas as pd
import polars as pl
from scipy.integrate import cumulative_trapezoid

import ionworksdata as iwdata


def _sign_with_tolerance(x: np.ndarray, tol: float | None = None) -> np.ndarray:
    if tol is None:
        tol = iwdata.settings.get_sign_tolerance()
    return np.sign(x * (abs(x) > tol))


def get_cumulative_step_number(
    data: pl.DataFrame | pd.DataFrame, options: dict | None = None
) -> pl.Series:
    """
    Assign a cumulative step number to each row in the data.

    Parameters
    ----------
    data : pl.DataFrame | pd.DataFrame
        The data to assign step numbers to.
    options : dict, optional
        Options for assigning step numbers. The default is None, which uses the following
        default options:

        - ``method``: The method to use for assigning step numbers. Default is ``status``.
          Options are:

            - ``status``: Assigns a new step number each time the status changes.
            - ``current sign``: Assigns a new step number each time the sign of the
              current divided by the absolute maximum current changes more than 1e-2.
            - ``step column``: Assigns a new step number each time the numeric value in the
              step column changes (see ``step column`` option).

        - ``current units``: The format of the current. Default is ``total``. Options are:
            - ``total``: The current is in units of A.
            - ``density``: The current is in units of mA.cm-2.
        - ``zero current tol``: Tolerance for considering current as zero when using
          ``current sign`` method. Values below this fraction of max current are treated
          as zero. Default: value from global settings.
        - ``step column``: The column to use for assigning step numbers if using the
          ``step column`` method. Default is ``Step number``.
        - ``group EIS steps``: Whether to group EIS steps together as a single step. If False,
          and ``method`` is ``current sign`` an EIS experiment at a single state of charge will
          have a large number of steps (the step number will change each time the current
          sign changes). Default: True.
        - ``EIS tolerance``: The tolerance for considering a frequency as an EIS step.
          Default: 1e-8.

    Returns
    -------
    pl.Series
        The step numbers as an integer Series starting at 0.
    """
    # Convert pandas to polars if needed
    if isinstance(data, pd.DataFrame):
        data = pl.from_pandas(data)

    current_units, _ = iwdata.util.get_current_and_capacity_units(options)
    default_options = {
        "method": iwutil.OptionSpec("status", ["current sign", "step column"]),
        "current units": iwutil.OptionSpec("total", ["density"]),
        "zero current tol": iwdata.settings.get_zero_current_percent_tol(),
        "step column": iwutil.OptionSpec("Step number"),
        "group EIS steps": iwutil.OptionSpec(True, [False]),
        "EIS tolerance": iwdata.settings.get_eis_tolerance(),
    }
    combined_options = iwutil.check_and_combine_options(
        default_options, options, filter_unknown=True
    )
    if combined_options["method"] == "status":
        status_values = data.get_column("Status").to_list()
        unique_status = []
        seen = set()
        for s in status_values:
            if s not in seen:
                seen.add(s)
                unique_status.append(s)
        status_map = {s: i for i, s in enumerate(unique_status)}
        index = np.array([status_map[s] for s in status_values], dtype=float)
    elif combined_options["method"] == "current sign":
        # Use numpy for numerical operations (sign detection with tolerance)
        current_data = data.get_column(f"Current [{current_units}]").to_numpy()
        max_abs = np.max(np.abs(current_data)) if len(current_data) > 0 else 1.0
        # Avoid division by zero - if max_abs is 0, all values are 0
        if max_abs == 0:
            normalized_current = np.zeros_like(current_data)
        else:
            normalized_current = current_data / max_abs
        index = _sign_with_tolerance(
            normalized_current, tol=combined_options["zero current tol"]
        )
    elif combined_options["method"] == "step column":
        # Use numpy for diff operations and array manipulation
        step_column = combined_options["step column"]
        index = data.get_column(step_column).to_numpy()

    # Detect changes in index using numpy diff and cumsum for efficiency
    # Compute step changes: 1 when index changes, else 0
    diff = np.diff(index, prepend=index[0])
    step_changes = np.abs(np.sign(diff))
    step_changes[0] = 0

    # Handle EIS steps if needed
    if combined_options["group EIS steps"] and ("Frequency [Hz]" in data.columns):
        freq = data.get_column("Frequency [Hz]")
        # Use numpy for boolean array operations and boundary detection
        is_eis = (freq > combined_options["EIS tolerance"]).to_numpy()
        eis_boundary = np.concatenate([[False], is_eis[1:] != is_eis[:-1]])
        step_changes[is_eis] = 0
        step_changes[eis_boundary] = 1

    # Calculate cumulative step number using numpy cumsum
    # Filter out NaN values before casting to avoid RuntimeWarning
    step_changes_clean = np.nan_to_num(step_changes, nan=0.0)
    step_number = step_changes_clean.cumsum().astype(np.int64)
    step_number = step_number - step_number[0]
    return pl.Series(step_number)


def set_cumulative_step_number(data: pl.DataFrame, **kwargs) -> pl.DataFrame:
    """
    Add a column with the cumulative step number to the data.

    Parameters
    ----------
    data : pl.DataFrame
        The data to add the step number to.
    kwargs
        Additional keyword arguments to pass to get_cumulative_step_number.

    Returns
    -------
    pl.DataFrame
        The data with the step number added.
    """
    step_series = get_cumulative_step_number(data, **kwargs)
    # Always overwrite/define the column
    out = data.with_columns(step_series.alias("Step number"))
    return out


def set_step_count(
    data: pl.DataFrame | pd.DataFrame, options: dict | None = None
) -> pl.DataFrame:
    """
    Assign a cumulative step number "Step count" to each row in the data by detecting
    changes in the "Step from cycler" column.

    Parameters
    ----------
    data : pl.DataFrame | pd.DataFrame
        The data to assign step count to.
    options : dict, optional
        Additional options to pass to the function. The default is None, which uses the
        following default options:

        - ``method``: The method to use for assigning step count. Default is ``step column``.
          Options are:

            - ``step column``: Assigns a new step number each time the numeric value in the
              step column changes (see ``step column`` option).

        - ``step column``: The column to use for assigning step numbers if using the
          ``step column`` method. Default is ``Step from cycler``.

    Returns
    -------
    pl.DataFrame
        The data with "Step count" column added.
    """
    if isinstance(data, pd.DataFrame):
        data = pl.from_pandas(data)

    default_options = {
        "method": iwutil.OptionSpec("step column", ["step column"]),
        "step column": iwutil.OptionSpec(["Step from cycler"]),
    }
    options = iwutil.check_and_combine_options(
        default_options, options, filter_unknown=True
    )
    if options is None:
        options = {"method": "step column", "step column": "Step from cycler"}
    else:
        col = options.get("step column", "Step from cycler")
        if isinstance(col, list):
            options["step column"] = col[0]
    step_series = get_cumulative_step_number(data, options)
    return data.with_columns(step_series.alias("Step count"))


def get_cumulative_cycle_number(
    data: pl.DataFrame, options: dict | None = None
) -> pl.Series:
    """
    Assign a cumulative cycle number "Cycle count" to each row in the data.

    Parameters
    ----------
    data : pl.DataFrame
        The data to assign cycle count to.
    options : dict, optional
        Additional options to pass to the function. The default is None, which uses the
        following default options:

        - ``method``: The method to use for assigning cycle count. Default is ``cycle column``.
          Options are:

            - ``cycle column``: Assigns a new cycle number each time the numeric value in the
              cycle column changes (see ``cycle column`` option).

        - ``cycle column``: The column to use for assigning cycle numbers. Default is
          ``Cycle number``.

    Returns
    -------
    pl.Series
        The cumulative cycle numbers.
    """
    default_options = {
        "method": iwutil.OptionSpec("cycle column", ["cycle column"]),
        "cycle column": iwutil.OptionSpec(["Cycle number"]),
    }
    combined_options = iwutil.check_and_combine_options(
        default_options, options, filter_unknown=True
    )
    if combined_options["method"] == "cycle column":
        cycle_column = combined_options["cycle column"]
        if isinstance(cycle_column, list):
            cycle_column = cycle_column[0]

        # Use numpy for diff and cumsum operations for efficient change detection
        cycle_values = data.get_column(cycle_column).to_numpy()
        changes = np.diff(cycle_values, prepend=cycle_values[0]) != 0
        cumulative_cycles = np.cumsum(changes).astype(np.int64)
        return pl.Series(cumulative_cycles)

    raise ValueError(f"Unsupported method: {combined_options['method']}")


def set_cumulative_cycle_number(data: pl.DataFrame, **kwargs) -> pl.DataFrame:
    """
    Add a column with the cumulative cycle number to the data.

    Parameters
    ----------
    data : pl.DataFrame
        The data to add the cycle number to.
    kwargs
        Additional keyword arguments to pass to get_cumulative_cycle_number.

    Returns
    -------
    pl.DataFrame
        The data with the cycle number added.
    """
    cycle_series = get_cumulative_cycle_number(data, **kwargs)
    return data.with_columns(cycle_series.alias("Cycle number"))


def set_cycle_count(data: pl.DataFrame) -> pl.DataFrame:
    """
    Assign a cumulative cycle number "Cycle count" to each row in the data by detecting
    changes in the "Cycle from cycler" column. If "Cycle from cycler" doesn't exist,
    sets all values to 0.

    Parameters
    ----------
    data : pl.DataFrame
        The data to assign cycle count to.

    Returns
    -------
    pl.DataFrame
        The data with "Cycle count" column added.
    """
    if "Cycle from cycler" in data.columns:
        options = {"method": "cycle column", "cycle column": "Cycle from cycler"}
        cycle_series = get_cumulative_cycle_number(data, options)
    else:
        # Set all to 0 if no cycle information available
        cycle_series = pl.Series([0] * data.height)
    return data.with_columns(cycle_series.alias("Cycle count"))


def reset_time(data: pl.DataFrame) -> pl.DataFrame:
    """
    Reset the time to start at zero

    Parameters
    ----------
    data : pl.DataFrame
        The data to reset the time for.
    """
    # Extract first value using numpy (simpler than Polars head/item)
    first = data.get_column("Time [s]").to_numpy()[0]
    return data.with_columns((pl.col("Time [s]") - pl.lit(first)).alias("Time [s]"))


def offset_duplicate_times(data: pl.DataFrame, offset: float = 1e-6) -> pl.DataFrame:
    """
    Offset duplicate time values by a small amount. This is preferable to
    removing the duplicate time values because removing duplicate time values can
    lead to missing steps in the data.

    Parameters
    ----------
    data : pl.DataFrame
        The data to remove duplicate time values from.
    offset : float, optional
        The amount to offset the duplicate time values by.
    """
    t = data.get_column("Time [s]").to_numpy().astype(float)

    if len(t) <= 1:
        return data

    max_iter = 100
    while True:
        # Sort to make duplicates adjacent (stable sort preserves relative order)
        sort_idx = np.argsort(t, kind="stable")
        sorted_t = t[sort_idx]

        # Find group boundaries (where values change)
        is_new_group = np.concatenate([[True], sorted_t[1:] != sorted_t[:-1]])

        # Check if there are any duplicates (all True means no duplicates)
        if is_new_group.all():
            break

        # Compute group IDs and group start indices
        group_id = np.cumsum(is_new_group) - 1
        group_starts = np.where(is_new_group)[0]

        # Compute cumulative count within each group (0 for first, 1 for second, etc)
        # e.g. for sorted [1, 1, 1, 2, 2] -> cumcount = [0, 1, 2, 0, 1]
        cumcount_sorted = np.arange(len(sorted_t)) - group_starts[group_id]

        # Unsort to get back to original order
        inv_sort_idx = np.empty_like(sort_idx)
        inv_sort_idx[sort_idx] = np.arange(len(sort_idx))
        cumcount = cumcount_sorted[inv_sort_idx]

        # Apply offset: each duplicate gets offset * its position within the group
        # This resolves most duplicates in a single pass (unlike fixed offset approach)
        t = t + cumcount * offset

        max_iter -= 1
        if max_iter == 0:
            raise ValueError("Offsetting duplicate times failed")

    return data.with_columns(pl.Series("Time [s]", t))


def _apply_step_resets(
    discharge_values: np.ndarray,
    charge_values: np.ndarray,
    step_numbers: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Reset cumulative values to 0 at each step boundary.

    Vectorized implementation that subtracts the starting value of each step
    from all points within that step, ensuring each step starts at 0.

    Parameters
    ----------
    discharge_values : np.ndarray
        Cumulative discharge values to reset.
    charge_values : np.ndarray
        Cumulative charge values to reset.
    step_numbers : np.ndarray
        Step numbers for each data point.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Reset discharge and charge values.
    """
    # Detect step boundaries (first row is always a step start)
    step_changes = np.concatenate(([True], np.diff(step_numbers) != 0))

    # For each row, subtract the cumulative value at the start of its step
    step_start_indices = np.where(step_changes)[0]
    step_group_indices = (
        np.searchsorted(
            step_start_indices, np.arange(len(discharge_values)), side="right"
        )
        - 1
    )

    # Get the starting cumulative value for each step
    discharge_start_values = discharge_values[step_start_indices[step_group_indices]]
    charge_start_values = charge_values[step_start_indices[step_group_indices]]

    # Subtract the start value from each row
    return (
        discharge_values - discharge_start_values,
        charge_values - charge_start_values,
    )


def _split_cumulative_by_direction(
    cumulative_values: np.ndarray,
    direction_indicator: np.ndarray,
    step_numbers: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split cumulative values into positive and negative components based on
    direction indicator (e.g., current or power).

    Vectorized implementation that splits a cumulative metric (capacity/energy)
    into discharge and charge components based on the sign of a direction
    indicator. Resets accumulation at step boundaries if step_numbers provided.

    Parameters
    ----------
    cumulative_values : np.ndarray
        Cumulative values to split (e.g., capacity or energy).
    direction_indicator : np.ndarray
        Direction indicator (e.g., current or power). Positive values indicate
        discharge, negative indicate charge.
    step_numbers : np.ndarray, optional
        Step numbers for each data point. If provided, accumulation resets at
        each step boundary.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Discharge and charge components as numpy arrays.
    """
    # Calculate deltas
    deltas = np.diff(cumulative_values, prepend=0)

    # Create masks for discharge (positive), charge (negative), and rest
    is_discharge = direction_indicator > 0
    is_charge = direction_indicator < 0

    # Split deltas based on direction
    discharge_deltas = np.where(is_discharge, deltas, 0)
    charge_deltas = np.where(is_charge, np.abs(deltas), 0)

    # Accumulate
    discharge_cumulative = np.cumsum(discharge_deltas)
    charge_cumulative = np.cumsum(charge_deltas)

    # Apply step resets if step_numbers provided
    if step_numbers is not None:
        discharge_cumulative, charge_cumulative = _apply_step_resets(
            discharge_cumulative, charge_cumulative, step_numbers
        )
        # Ensure non-negative (take absolute value after step resets)
        # This handles cases like Novonix where capacity decreases during discharge
        discharge_cumulative = np.abs(discharge_cumulative)
        charge_cumulative = np.abs(charge_cumulative)

    return discharge_cumulative, charge_cumulative


def _calculate_capacity(
    data: pl.DataFrame, options: dict | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate discharge and charge capacity from the data.

    First checks if charge/discharge capacity columns exist. If so, uses them
    directly. Otherwise checks if a single "Capacity" column exists and splits
    it into discharge and charge based on current direction. Finally, calculates
    capacities using cumulative trapezoidal integration if no capacity columns
    are found. Resets capacity to zero at each step boundary if step information
    is available.

    Parameters
    ----------
    data : pl.DataFrame
        The data to get the capacity columns from.
    options : dict, optional
        Additional options to pass to the function. The default is None.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Discharge capacity and charge capacity as numpy arrays.
    """
    current_units, capacity_units = iwdata.util.get_current_and_capacity_units(options)
    current = data.get_column(f"Current [{current_units}]").to_numpy()

    # Check for step column (only Step count)
    step_numbers = None
    if "Step count" in data.columns:
        step_numbers = data.get_column("Step count").to_numpy()

    # First check if charge/discharge capacity columns exist
    discharge_cap_col = f"Discharge capacity [{capacity_units}]"
    charge_cap_col = f"Charge capacity [{capacity_units}]"
    if discharge_cap_col in data.columns and charge_cap_col in data.columns:
        discharge_capacity = data.get_column(discharge_cap_col).to_numpy()
        charge_capacity = data.get_column(charge_cap_col).to_numpy()
        # Apply step resets if step numbers available
        if step_numbers is not None:
            discharge_capacity, charge_capacity = _apply_step_resets(
                discharge_capacity, charge_capacity, step_numbers
            )
        # Ensure non-negative (take absolute value)
        discharge_capacity = np.abs(discharge_capacity)
        charge_capacity = np.abs(charge_capacity)
        return discharge_capacity, charge_capacity

    # Check if single capacity column exists
    capacity_col = f"Capacity [{capacity_units}]"
    if capacity_col in data.columns:
        capacity = data.get_column(capacity_col).to_numpy()
        return _split_cumulative_by_direction(capacity, current, step_numbers)

    # Otherwise, calculate using cumulative trapezoidal integration
    t = data.get_column("Time [s]").to_numpy()
    discharge_current = np.where(current > 0, current, 0)
    charge_current = np.where(current < 0, -current, 0)
    discharge_capacity = cumulative_trapezoid(discharge_current, t / 3600.0, initial=0)
    charge_capacity = cumulative_trapezoid(charge_current, t / 3600.0, initial=0)

    # Apply step resets if step numbers available
    if step_numbers is not None:
        discharge_capacity, charge_capacity = _apply_step_resets(
            discharge_capacity, charge_capacity, step_numbers
        )

    return discharge_capacity, charge_capacity


def set_capacity(data: pl.DataFrame, options: dict | None = None) -> pl.DataFrame:
    """
    Calculate discharge and charge capacity for the data and assign them to new
    columns called "Discharge capacity [A.h]" and "Charge capacity [A.h]"
    Drops the single "Capacity [A.h]" column if it exists.

    Parameters
    ----------
    data : pl.DataFrame
        The data to calculate the capacity for.
    options : dict, optional
        Additional options to pass to the function. The default is None.

    Returns
    -------
    pl.DataFrame
        The data with discharge and charge capacity columns added, and single
        capacity column removed if it existed.
    """
    _, capacity_units = iwdata.util.get_current_and_capacity_units(options)
    discharge_cap, charge_cap = _calculate_capacity(data, options)
    result = data.with_columns(
        [
            pl.Series(f"Discharge capacity [{capacity_units}]", discharge_cap),
            pl.Series(f"Charge capacity [{capacity_units}]", charge_cap),
        ]
    )
    # Drop single capacity column if it exists
    single_cap_col = f"Capacity [{capacity_units}]"
    if single_cap_col in result.columns:
        result = result.drop(single_cap_col)
    return result


def _calculate_energy(
    data: pl.DataFrame, options: dict | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate discharge and charge energy from the data.

    First checks if charge/discharge energy columns exist. If so, uses them
    directly. Otherwise checks if a single "Energy [W.h]" column exists and splits
    it into discharge and charge based on power direction. Finally, calculates
    energies using cumulative trapezoidal integration if no energy columns are
    found. Resets energy to zero at each step boundary if step information is
    available.

    Parameters
    ----------
    data : pl.DataFrame
        The data to get the energy columns from.
    options : dict, optional
        Additional options to pass to the function. The default is None.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Discharge energy and charge energy as numpy arrays.
    """
    # Calculate power from current and voltage if Power column doesn't exist
    if "Power [W]" in data.columns:
        power = data.get_column("Power [W]").to_numpy()
    else:
        current_units, _ = iwdata.util.get_current_and_capacity_units(options)
        current = data.get_column(f"Current [{current_units}]").to_numpy()
        voltage = data.get_column("Voltage [V]").to_numpy()
        power = current * voltage

    # Check for step column (only Step count)
    step_numbers = None
    if "Step count" in data.columns:
        step_numbers = data.get_column("Step count").to_numpy()

    # First check if charge/discharge energy columns exist
    if (
        "Discharge energy [W.h]" in data.columns
        and "Charge energy [W.h]" in data.columns
    ):
        discharge_energy = data.get_column("Discharge energy [W.h]").to_numpy()
        charge_energy = data.get_column("Charge energy [W.h]").to_numpy()
        # Apply step resets if step numbers available
        if step_numbers is not None:
            discharge_energy, charge_energy = _apply_step_resets(
                discharge_energy, charge_energy, step_numbers
            )
        # Ensure non-negative (take absolute value)
        discharge_energy = np.abs(discharge_energy)
        charge_energy = np.abs(charge_energy)
        return discharge_energy, charge_energy

    # Check if single energy column exists
    if "Energy [W.h]" in data.columns:
        energy = data.get_column("Energy [W.h]").to_numpy()
        return _split_cumulative_by_direction(energy, power, step_numbers)

    # Otherwise, calculate using cumulative trapezoidal integration
    t = data.get_column("Time [s]").to_numpy()
    discharge_power = np.where(power > 0, power, 0)
    charge_power = np.where(power < 0, -power, 0)
    discharge_energy = cumulative_trapezoid(discharge_power, t / 3600.0, initial=0)
    charge_energy = cumulative_trapezoid(charge_power, t / 3600.0, initial=0)

    # Apply step resets if step numbers available
    if step_numbers is not None:
        discharge_energy, charge_energy = _apply_step_resets(
            discharge_energy, charge_energy, step_numbers
        )

    return discharge_energy, charge_energy


def set_energy(data: pl.DataFrame, options: dict | None = None) -> pl.DataFrame:
    """
    Calculate discharge and charge energy for the data and assign them to new
    columns called "Discharge energy [W.h]" and "Charge energy [W.h]"
    Drops the single "Energy [W.h]" column if it exists.

    Parameters
    ----------
    data : pl.DataFrame
        The data to calculate the energy for.
    options : dict, optional
        Additional options to pass to the function. The default is None.

    Returns
    -------
    pl.DataFrame
        The data with discharge and charge energy columns added, and single
        energy column removed if it existed.
    """
    discharge_energy, charge_energy = _calculate_energy(data, options)
    result = data.with_columns(
        [
            pl.Series("Discharge energy [W.h]", discharge_energy),
            pl.Series("Charge energy [W.h]", charge_energy),
        ]
    )
    # Drop single energy column if it exists
    if "Energy [W.h]" in result.columns:
        result = result.drop("Energy [W.h]")
    return result


def _calculate_net_capacity(
    data: pl.DataFrame, options: dict | None = None
) -> np.ndarray:
    """
    Calculate net capacity (discharge minus charge) from the data.

    Net capacity represents the net amount of charge removed from the battery.
    Positive values indicate net discharge, negative values indicate net charge.

    Parameters
    ----------
    data : pl.DataFrame
        The data to calculate net capacity from.
    options : dict, optional
        Additional options to pass to the function. The default is None.

    Returns
    -------
    np.ndarray
        Net capacity as a numpy array.
    """
    _, capacity_units = iwdata.util.get_current_and_capacity_units(options)
    discharge_cap_col = f"Discharge capacity [{capacity_units}]"
    charge_cap_col = f"Charge capacity [{capacity_units}]"

    # Check if columns exist
    if discharge_cap_col in data.columns and charge_cap_col in data.columns:
        discharge_cap = data.get_column(discharge_cap_col).to_numpy()
        charge_cap = data.get_column(charge_cap_col).to_numpy()
    else:
        # Calculate if not present
        discharge_cap, charge_cap = _calculate_capacity(data, options)

    return discharge_cap - charge_cap


def get_cumulative_net_capacity(
    data: pl.DataFrame,
    options: dict | None = None,
) -> np.ndarray:
    """
    Cumulative net capacity (no reset) at each row.

    Discharge/charge capacity columns reset to 0 at each step boundary.
    We recover true cumulative values by diffing each column, clipping
    negative diffs (the resets) to zero, and cumsumming. The result is
    cumulative discharge minus cumulative charge.

    Parameters
    ----------
    data : pl.DataFrame
        Data with discharge/charge capacity columns (or current for integration).
    options : dict, optional
        Passed to get_current_and_capacity_units / _calculate_capacity.

    Returns
    -------
    np.ndarray
        Cumulative net capacity at each row, same length as data.
    """
    _, capacity_units = iwdata.util.get_current_and_capacity_units(options)
    discharge_cap_col = f"Discharge capacity [{capacity_units}]"
    charge_cap_col = f"Charge capacity [{capacity_units}]"

    if discharge_cap_col in data.columns and charge_cap_col in data.columns:
        discharge_cap = data.get_column(discharge_cap_col).to_numpy()
        charge_cap = data.get_column(charge_cap_col).to_numpy()
    else:
        discharge_cap, charge_cap = _calculate_capacity(data, options)

    cumul_discharge = np.cumsum(np.maximum(np.diff(discharge_cap, prepend=0), 0))
    cumul_charge = np.cumsum(np.maximum(np.diff(charge_cap, prepend=0), 0))
    return cumul_discharge - cumul_charge


def set_net_capacity(data: pl.DataFrame, options: dict | None = None) -> pl.DataFrame:
    """
    Calculate the net capacity for the data and assign it to a new column called
    "Capacity [A.h]".

    Parameters
    ----------
    data : pl.DataFrame
        The data to calculate the net capacity for.
    options : dict, optional
        Additional options to pass to the function. The default is None.

    Returns
    -------
    pl.DataFrame
        The data with the net capacity added.
    """
    _, capacity_units = iwdata.util.get_current_and_capacity_units(options)
    cap_col = f"Capacity [{capacity_units}]"
    if cap_col in data.columns:
        raise ValueError(f"Column '{cap_col}' already exists in data.")
    net_capacity = _calculate_net_capacity(data, options)
    return data.with_columns(pl.Series(cap_col, net_capacity))


def set_nominal_soc(
    data: pl.DataFrame, cell_metadata: dict, options: dict | None = None
) -> pl.DataFrame:
    """
    Calculate the nominal SOC for the data and assign it to a new column called
    "Nominal SOC". SOC is calculated based on net capacity (discharge - charge).

    Parameters
    ----------
    data : pl.DataFrame
        The data to calculate the nominal SOC for. Must have columns
        "Discharge capacity [A.h]" and "Charge capacity [A.h]" (or mA.h.cm-2).
        If they don't exist, use set_capacity to calculate them first.
    cell_metadata : dict
        The metadata for the cell. Should have a key "Nominal cell capacity [A.h]" or
        "Nominal cell capacity [mA.h.cm-2]"
    options : dict, optional
        Additional options to pass to the function. The default is None.

    Returns
    -------
    pl.DataFrame
        The data with the nominal SOC added.
    """
    _, capacity_units = iwdata.util.get_current_and_capacity_units(options)
    net_capacity = _calculate_net_capacity(data, options)
    Q = cell_metadata[f"Nominal cell capacity [{capacity_units}]"]
    soc_nom = 1 - net_capacity / Q
    max_soc = float(np.max(soc_nom)) if len(soc_nom) > 0 else 1.0
    soc_nom_adjusted = soc_nom + (1 - max_soc)
    return data.with_columns(pl.Series("Nominal SOC", soc_nom_adjusted))


def convert_current_density_to_total_current(
    data: pl.DataFrame, metadata: dict
) -> pl.DataFrame:
    """
    Convert the current density from mA.cm-2 to A

    Parameters
    ----------
    data : pl.DataFrame
        The data to convert. Should have a column "Current [mA.cm-2]".
    metadata : dict
        The metadata for the data. Should have a key "Electrode area [cm2]".

    Returns
    -------
    pl.DataFrame
        The data with the current converted to A.
    """
    return data.with_columns(
        (
            pl.col("Current [mA.cm-2]")
            * float(metadata["Electrode area [cm2]"])
            / 1000.0
        ).alias("Current [A]")
    ).drop(["Current [mA.cm-2]"])


def convert_total_current_to_current_density(
    data: pl.DataFrame, metadata: dict
) -> pl.DataFrame:
    """
    Convert the total current from A to mA.cm-2

    Parameters
    ----------
    data : pl.DataFrame
        The data to convert. Should have a column "Current [A]".
    metadata : dict
        The metadata for the data. Should have a key "Electrode area [cm2]".

    Returns
    -------
    pl.DataFrame
        The data with the current converted to mA.cm-2.
    """
    return data.with_columns(
        (
            pl.col("Current [A]") / float(metadata["Electrode area [cm2]"]) * 1000.0
        ).alias("Current [mA.cm-2]")
    ).drop(["Current [A]"])


def set_positive_current_for_discharge(
    data: pl.DataFrame, options: dict | None = None
) -> pl.DataFrame:
    """
    Identify whether positive current is charging or discharging, then make sure that
    positive current is discharging and negative current is charging.

    Three paths:

    1. **All currents same sign** (cycler recording absolute values) *and*
       a mode column (e.g. ``"Status"``) is present with charge/discharge
       labels — use the mode column to negate charge rows.
    2. **All currents same sign** but no mode column — apply per-step sign
       correction using voltage response to classify each step as charge
       or discharge.
    3. **Mixed-sign currents** (normal case) — apply the original global
       mean voltage-response transform across the entire measurement.

    Parameters
    ----------
    data : pl.DataFrame
        The data to set the current direction for.
    options : dict, optional
        Additional options to pass to the function. The default is None.

    Returns
    -------
    pl.DataFrame
        The data with the current direction set to positive current is discharging.
    """
    options = options or {}
    options["method"] = "current sign"

    current_units, _ = iwdata.util.get_current_and_capacity_units(options)
    current_col = f"Current [{current_units}]"

    # --- Step 1: detect if all non-rest currents are the same sign ----------
    rest_tol = iwdata.settings.get_rest_tol()
    current_np = data.get_column(current_col).to_numpy()
    non_rest_current = current_np[np.abs(current_np) > rest_tol]

    if len(non_rest_current) == 0:
        return data

    all_positive = bool((non_rest_current > 0).all())
    all_negative = bool((non_rest_current < 0).all())
    all_same_sign = all_positive or all_negative

    if all_same_sign:
        # --- Step 2: check for a mode column (e.g. "Status") ---------------
        mode_result = _fix_sign_from_mode_column(data, current_col)
        if mode_result is not None:
            return mode_result

        # No mode column — fall back to per-step voltage-response correction
        return _fix_sign_per_step(data, current_col, options)

    # --- Step 3: mixed-sign currents — apply old global transform -----------
    return _fix_sign_global(data, current_col, options)


def _fix_sign_from_mode_column(
    data: pl.DataFrame, current_col: str
) -> pl.DataFrame | None:
    """Use a mode column (``"Status"``) to negate charge rows.

    Returns the corrected DataFrame, or ``None`` if no usable mode column
    is present.
    """
    if "Status" not in data.columns:
        return None

    statuses = set(data.get_column("Status").unique().to_list())
    # Need both charge ("C") and discharge ("D") labels to be useful
    if "C" not in statuses or "D" not in statuses:
        return None

    # All non-rest current is the same sign (caller already checked).
    # Negate the charge rows so that discharge stays positive and charge
    # becomes negative.
    return data.with_columns(
        pl.when(pl.col("Status") == "C")
        .then(-pl.col(current_col))
        .otherwise(pl.col(current_col))
        .alias(current_col)
    )


def _prepare_step_data(
    data: pl.DataFrame, current_col: str, options: dict
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Shared step detection and filtering for sign correction helpers.

    Returns ``(work, summary_no_rest)`` where *work* has step columns and
    *summary_no_rest* contains only non-rest steps.
    """
    work = data.select(
        [
            c
            for c in ["Time [s]", current_col, "Voltage [V]", "Frequency [Hz]"]
            if c in data.columns
        ]
    )

    # Filter out EIS rows before step detection
    eis_tol = options.get("EIS tolerance", iwdata.settings.get_eis_tolerance())
    if "Frequency [Hz]" in work.columns:
        freq_np = work.get_column("Frequency [Hz]").to_numpy()
        is_eis_np = freq_np > eis_tol
        if bool(is_eis_np.any()) and bool((~is_eis_np).any()):
            work = work.filter(pl.col("Frequency [Hz]") <= eis_tol)

    work = iwdata.transform.set_cumulative_step_number(work, options=options)
    work = iwdata.transform.set_step_count(
        work, options={"method": "step column", "step column": "Step number"}
    )

    step_summary = work.group_by("Step count").agg(
        pl.col(current_col).mean().alias("mean_current"),
    )

    rest_tol = iwdata.settings.get_rest_tol()
    summary_no_rest = step_summary.filter(pl.col("mean_current").abs() > rest_tol)

    return work, summary_no_rest


def _fix_sign_per_step(
    data: pl.DataFrame, current_col: str, options: dict
) -> pl.DataFrame:
    """Per-step sign correction using WLS V-vs-Q classification.

    Used when all currents are the same sign (absolute-value cycler) but
    no mode column is available.  Each step is classified individually
    via ``positive_current_is_charge``; charge steps are negated.

    If all steps agree on a single direction, falls back to a global
    classification on the full time series.
    """
    work, summary_no_rest = _prepare_step_data(data, current_col, options)

    if summary_no_rest.height == 0:
        return data

    t_arr = work.get_column("Time [s]").to_numpy()
    current_arr = work.get_column(current_col).to_numpy()
    voltage_arr = work.get_column("Voltage [V]").to_numpy()
    step_counts = work.get_column("Step count").to_numpy()
    non_rest_step_ids = set(summary_no_rest.get_column("Step count").to_list())

    if summary_no_rest.height > 1:
        # Per-step classification
        steps_to_negate = set()
        has_discharge = False
        has_charge = False
        for step_id in non_rest_step_ids:
            mask = step_counts == step_id
            is_charge, _ = positive_current_is_charge(
                t_arr[mask], current_arr[mask], voltage_arr[mask]
            )
            if is_charge:
                steps_to_negate.add(step_id)
                has_charge = True
            else:
                has_discharge = True

        if has_discharge and has_charge:
            data_current = data.get_column(current_col).to_numpy().copy()
            negate_mask = np.isin(step_counts, list(steps_to_negate))
            data_current[negate_mask] = -data_current[negate_mask]
            return data.with_columns(pl.Series(current_col, data_current))

    # Single-direction data or all steps agreed: classify on full series
    is_charge, _ = positive_current_is_charge(t_arr, current_arr, voltage_arr)
    if is_charge:
        return data.with_columns(((-1) * pl.col(current_col)).alias(current_col))
    return data


def _fix_sign_global(
    data: pl.DataFrame, current_col: str, options: dict
) -> pl.DataFrame:
    """Global sign correction using confidence-weighted WLS vote.

    Used when currents are already mixed-sign (normal case).  Classifies
    each non-rest step individually via ``positive_current_is_charge``
    and uses a confidence-weighted vote (weight = 1 - p_value) to decide
    whether to flip the entire current column.
    """
    work, summary_no_rest = _prepare_step_data(data, current_col, options)

    if summary_no_rest.height == 0:
        return data

    t_arr = work.get_column("Time [s]").to_numpy()
    current_arr = work.get_column(current_col).to_numpy()
    voltage_arr = work.get_column("Voltage [V]").to_numpy()
    step_counts = work.get_column("Step count").to_numpy()
    non_rest_step_ids = set(summary_no_rest.get_column("Step count").to_list())

    charge_weight = 0.0
    discharge_weight = 0.0
    charge_count = 0
    discharge_count = 0
    for step_id in non_rest_step_ids:
        mask = step_counts == step_id
        is_charge, p_value = positive_current_is_charge(
            t_arr[mask], current_arr[mask], voltage_arr[mask]
        )
        confidence = 1.0 - p_value
        if is_charge:
            charge_weight += confidence
            charge_count += 1
        else:
            discharge_weight += confidence
            discharge_count += 1

    if charge_weight + discharge_weight > 0:
        should_negate = charge_weight > discharge_weight
    else:
        should_negate = charge_count > discharge_count

    if should_negate:
        return data.with_columns(((-1) * pl.col(current_col)).alias(current_col))
    return data


def derive_impedance_components(data: pl.DataFrame) -> pl.DataFrame:
    """Derive missing impedance components from modulus and phase.

    If ``Z_Mod [Ohm]`` and ``Z_Phase [deg]`` are present but ``Z_Re [Ohm]``
    or ``Z_Im [Ohm]`` are missing, compute them via:

    - ``Z_Re = Z_Mod * cos(Z_Phase)``
    - ``Z_Im = Z_Mod * sin(Z_Phase)``

    Columns that already exist are left untouched. If modulus or phase are
    absent the dataframe is returned unchanged.

    Parameters
    ----------
    data : pl.DataFrame
        Time-series dataframe, potentially containing impedance columns.

    Returns
    -------
    pl.DataFrame
        The dataframe with ``Z_Re [Ohm]`` and ``Z_Im [Ohm]`` added when
        they can be derived.
    """
    if "Z_Mod [Ohm]" not in data.columns or "Z_Phase [deg]" not in data.columns:
        return data

    phase_rad = pl.col("Z_Phase [deg]") * (np.pi / 180.0)
    exprs = []
    if "Z_Re [Ohm]" not in data.columns:
        exprs.append((pl.col("Z_Mod [Ohm]") * phase_rad.cos()).alias("Z_Re [Ohm]"))
    if "Z_Im [Ohm]" not in data.columns:
        exprs.append((pl.col("Z_Mod [Ohm]") * phase_rad.sin()).alias("Z_Im [Ohm]"))
    if exprs:
        data = data.with_columns(exprs)
    return data


def remove_outliers(
    data: pl.DataFrame,
    column: str,
    z_threshold: float = 3,
    data_range: slice | None = None,
) -> pl.DataFrame:
    """
    Remove outliers from the data based on the z-score of a column

    Parameters
    ----------
    data : pl.DataFrame
        The data to remove outliers from.
    column : str
        The column to calculate the z-score for.
    z_threshold : float, optional
        The z-score threshold to use for removing outliers. The default is 3.
    data_range : slice, optional
        The range of data points to consider for outlier detection.
        If None, all points are used. Use Python's slice notation, e.g.,
        slice(0, 100) for first 100 points, slice(-100, None) for last 100 points.

    Returns
    -------
    pl.DataFrame
        The data with the outliers removed.
    """
    # Get unique step numbers
    processed_frames: list[pl.DataFrame] = []
    for step in data.get_column("Step number").unique().to_list():
        step_df = data.filter(pl.col("Step number") == step)
        # Use numpy for statistical operations (nanmean, nanstd) and z-score calculation
        vals = step_df.get_column(column).to_numpy()
        mu = float(np.nanmean(vals)) if len(vals) else 0.0
        sigma = float(np.nanstd(vals)) if len(vals) else 1.0
        z = np.abs((vals - mu) / (sigma if sigma != 0 else 1.0))
        if data_range is not None:
            range_start, range_end, _ = data_range.indices(len(step_df))
            mask = np.zeros(len(step_df), dtype=bool)
            mask[range_start:range_end] = True
        else:
            mask = np.ones(len(step_df), dtype=bool)
        keep = ~((z >= z_threshold) & mask)
        processed_frames.append(
            step_df.with_row_index("__row").filter(pl.Series(keep)).drop("__row")
        )
    return pl.concat(processed_frames)
