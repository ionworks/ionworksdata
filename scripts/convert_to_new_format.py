#!/usr/bin/env python
"""
Convert time series and steps data from old format to new format.

Old Format:
- Single "Capacity [A.h]" column (cumulative across all steps)
- Single "Energy [W.h]" column (cumulative across all steps)
- Steps have "Delta capacity [A.h]" column

New Format:
- "Discharge capacity [A.h]" and "Charge capacity [A.h]" columns
- "Discharge energy [W.h]" and "Charge energy [W.h]" columns
- Capacities and energies reset at each step boundary
- Steps have "Discharge capacity [A.h]" and "Charge capacity [A.h]" columns

Usage:
    python convert_to_new_format.py input_time_series.csv input_steps.csv \\
        output_time_series.csv output_steps.csv

    Or for directories:
    python convert_to_new_format.py input_dir/ output_dir/
"""

import sys
from pathlib import Path
import polars as pl
import numpy as np


def convert_time_series(data: pl.DataFrame) -> pl.DataFrame:
    """
    Convert time series from old format to new format.

    Parameters
    ----------
    data : pl.DataFrame
        Time series data in old format. May or may not have Capacity/Energy columns.

    Returns
    -------
    pl.DataFrame
        Time series data in new format with separate discharge/charge columns.
    """
    result = data.clone()

    # Determine units based on what columns exist
    if "Current [A]" in data.columns:
        current_col = "Current [A]"
        capacity_units = "A.h"
    elif "Current [mA.cm-2]" in data.columns:
        current_col = "Current [mA.cm-2]"
        capacity_units = "mA.h.cm-2"
    else:
        print("  Warning: No current column found, skipping capacity conversion")
        current_col = None

    # Check for capacity columns
    old_capacity_col = f"Capacity [{capacity_units}]"

    if current_col and old_capacity_col in data.columns:
        print(f"  Converting capacity column: {old_capacity_col}")

        capacity = data.get_column(old_capacity_col).to_numpy()
        current = data.get_column(current_col).to_numpy()

        # Get step numbers if available
        step_numbers = None
        for step_col in ["Step from cycler", "Step number"]:
            if step_col in data.columns:
                step_numbers = data.get_column(step_col).to_numpy()
                print(f"  Using step column: {step_col}")
                break

        # Calculate deltas
        deltas = np.diff(capacity, prepend=0)

        # Split by current direction
        is_discharge = current > 0
        is_charge = current < 0

        discharge_deltas = np.where(is_discharge, deltas, 0)
        charge_deltas = np.where(is_charge, np.abs(deltas), 0)

        # Accumulate
        discharge_capacity = np.cumsum(discharge_deltas)
        charge_capacity = np.cumsum(charge_deltas)

        # Apply step resets if available
        if step_numbers is not None:
            step_changes = np.concatenate(([True], np.diff(step_numbers) != 0))
            step_start_indices = np.where(step_changes)[0]
            step_group_indices = (
                np.searchsorted(
                    step_start_indices,
                    np.arange(len(discharge_capacity)),
                    side="right",
                )
                - 1
            )

            discharge_start = discharge_capacity[step_start_indices[step_group_indices]]
            charge_start = charge_capacity[step_start_indices[step_group_indices]]

            discharge_capacity -= discharge_start
            charge_capacity -= charge_start

        # Add new columns and remove old
        result = result.with_columns(
            [
                pl.Series(f"Discharge capacity [{capacity_units}]", discharge_capacity),
                pl.Series(f"Charge capacity [{capacity_units}]", charge_capacity),
            ]
        ).drop(old_capacity_col)

    elif current_col and "Time [s]" in data.columns:
        # Calculate capacity from scratch if not present
        print("  Calculating capacity from current (no existing column)")

        current = data.get_column(current_col).to_numpy()
        time = data.get_column("Time [s]").to_numpy()

        # Get step numbers if available
        step_numbers = None
        for step_col in ["Step from cycler", "Step number"]:
            if step_col in data.columns:
                step_numbers = data.get_column(step_col).to_numpy()
                print(f"  Using step column: {step_col}")
                break

        # Separate discharge and charge currents
        discharge_current = np.where(current > 0, current, 0)
        charge_current = np.where(current < 0, -current, 0)

        # Integrate using trapezoidal rule (convert seconds to hours)
        from scipy.integrate import cumulative_trapezoid

        discharge_capacity = cumulative_trapezoid(
            discharge_current, time / 3600.0, initial=0
        )
        charge_capacity = cumulative_trapezoid(charge_current, time / 3600.0, initial=0)

        # Apply step resets if available
        if step_numbers is not None:
            step_changes = np.concatenate(([True], np.diff(step_numbers) != 0))
            step_start_indices = np.where(step_changes)[0]
            step_group_indices = (
                np.searchsorted(
                    step_start_indices,
                    np.arange(len(discharge_capacity)),
                    side="right",
                )
                - 1
            )

            discharge_start = discharge_capacity[step_start_indices[step_group_indices]]
            charge_start = charge_capacity[step_start_indices[step_group_indices]]

            discharge_capacity -= discharge_start
            charge_capacity -= charge_start

        # Add new columns
        result = result.with_columns(
            [
                pl.Series(f"Discharge capacity [{capacity_units}]", discharge_capacity),
                pl.Series(f"Charge capacity [{capacity_units}]", charge_capacity),
            ]
        )

    # Check for energy columns
    if "Energy [W.h]" in data.columns and "Power [W]" in data.columns:
        print("  Converting energy column: Energy [W.h]")

        energy = data.get_column("Energy [W.h]").to_numpy()
        power = data.get_column("Power [W]").to_numpy()

        # Get step numbers if available
        step_numbers = None
        for step_col in ["Step from cycler", "Step number"]:
            if step_col in result.columns:
                step_numbers = result.get_column(step_col).to_numpy()
                break

        # Calculate deltas
        deltas = np.diff(energy, prepend=0)

        # Split by power direction
        is_discharge = power > 0
        is_charge = power < 0

        discharge_deltas = np.where(is_discharge, deltas, 0)
        charge_deltas = np.where(is_charge, np.abs(deltas), 0)

        # Accumulate
        discharge_energy = np.cumsum(discharge_deltas)
        charge_energy = np.cumsum(charge_deltas)

        # Apply step resets if available
        if step_numbers is not None:
            step_changes = np.concatenate(([True], np.diff(step_numbers) != 0))
            step_start_indices = np.where(step_changes)[0]
            step_group_indices = (
                np.searchsorted(
                    step_start_indices, np.arange(len(discharge_energy)), side="right"
                )
                - 1
            )

            discharge_start = discharge_energy[step_start_indices[step_group_indices]]
            charge_start = charge_energy[step_start_indices[step_group_indices]]

            discharge_energy -= discharge_start
            charge_energy -= charge_start

        # Add new columns and remove old
        result = result.with_columns(
            [
                pl.Series("Discharge energy [W.h]", discharge_energy),
                pl.Series("Charge energy [W.h]", charge_energy),
            ]
        ).drop("Energy [W.h]")

    elif "Power [W]" in result.columns and "Time [s]" in result.columns:
        # Calculate energy from scratch if not present
        print("  Calculating energy from power (no existing column)")

        power = result.get_column("Power [W]").to_numpy()
        time = result.get_column("Time [s]").to_numpy()

        # Get step numbers if available
        step_numbers = None
        for step_col in ["Step from cycler", "Step number"]:
            if step_col in result.columns:
                step_numbers = result.get_column(step_col).to_numpy()
                break

        # Separate discharge and charge power
        discharge_power = np.where(power > 0, power, 0)
        charge_power = np.where(power < 0, -power, 0)

        # Integrate using trapezoidal rule (convert seconds to hours)
        from scipy.integrate import cumulative_trapezoid

        discharge_energy = cumulative_trapezoid(
            discharge_power, time / 3600.0, initial=0
        )
        charge_energy = cumulative_trapezoid(charge_power, time / 3600.0, initial=0)

        # Apply step resets if available
        if step_numbers is not None:
            step_changes = np.concatenate(([True], np.diff(step_numbers) != 0))
            step_start_indices = np.where(step_changes)[0]
            step_group_indices = (
                np.searchsorted(
                    step_start_indices, np.arange(len(discharge_energy)), side="right"
                )
                - 1
            )

            discharge_start = discharge_energy[step_start_indices[step_group_indices]]
            charge_start = charge_energy[step_start_indices[step_group_indices]]

            discharge_energy -= discharge_start
            charge_energy -= charge_start

        # Add new columns
        result = result.with_columns(
            [
                pl.Series("Discharge energy [W.h]", discharge_energy),
                pl.Series("Charge energy [W.h]", charge_energy),
            ]
        )

    return result


def convert_steps(steps: pl.DataFrame) -> pl.DataFrame:
    """
    Convert steps from old format to new format.

    Parameters
    ----------
    steps : pl.DataFrame
        Steps data in old format with Delta capacity column.

    Returns
    -------
    pl.DataFrame
        Steps data in new format with separate discharge/charge columns.
    """
    result = steps.clone()

    # Check for capacity columns
    capacity_units = "A.h" if "Delta capacity [A.h]" in steps.columns else "mA.h.cm-2"
    old_delta_col = f"Delta capacity [{capacity_units}]"

    if old_delta_col in steps.columns:
        print(f"  Converting steps column: {old_delta_col}")

        # For steps, we need to determine if each step was discharge or charge
        # We can use the "Mean current" or "Type" column if available
        delta_capacity = steps.get_column(old_delta_col).to_numpy()

        if "Mean current [A]" in steps.columns:
            mean_current = steps.get_column("Mean current [A]").to_numpy()
        elif "Mean current [mA.cm-2]" in steps.columns:
            mean_current = steps.get_column("Mean current [mA.cm-2]").to_numpy()
        else:
            # Try to infer from Type column
            if "Type" in steps.columns:
                step_type = steps.get_column("Type").to_list()
                mean_current = np.array(
                    [
                        1.0
                        if "discharge" in t.lower()
                        else -1.0
                        if "charge" in t.lower()
                        else 0.0
                        for t in step_type
                    ]
                )
            else:
                print(
                    "  Warning: Cannot determine step direction. "
                    "Setting all as discharge."
                )
                mean_current = np.ones_like(delta_capacity)

        # Split delta capacity based on current direction
        discharge_capacity = np.where(mean_current > 0, np.abs(delta_capacity), 0.0)
        charge_capacity = np.where(mean_current < 0, np.abs(delta_capacity), 0.0)

        # Add new columns and remove old
        result = result.with_columns(
            [
                pl.Series(f"Discharge capacity [{capacity_units}]", discharge_capacity),
                pl.Series(f"Charge capacity [{capacity_units}]", charge_capacity),
            ]
        ).drop(old_delta_col)

    # Remove obsolete capacity columns from old format
    obsolete_columns = []
    for units in ["A.h", "mA.h.cm-2"]:
        for col_name in [
            f"Start capacity [{units}]",
            f"End capacity [{units}]",
            f"Delta capacity [{units}]",  # Already handled above, but include for safety
        ]:
            if col_name in result.columns:
                obsolete_columns.append(col_name)

    if obsolete_columns:
        print(f"  Removing obsolete columns: {', '.join(obsolete_columns)}")
        result = result.drop(obsolete_columns)

    # Handle cycle capacity columns if they exist
    if "Cycle charge capacity [A.h]" not in result.columns:
        # These might need to be recalculated from the new format
        if "Cycle number" in result.columns:
            print("  Note: Cycle capacity columns may need recalculation")

    return result


def convert_files(
    input_time_series: Path,
    input_steps: Path,
    output_time_series: Path,
    output_steps: Path,
):
    """Convert a pair of time series and steps files."""
    print("\nConverting files:")
    print(f"  Input time series: {input_time_series}")
    print(f"  Input steps: {input_steps}")

    # Load data
    print("\nLoading data...")
    time_series = pl.read_csv(input_time_series)
    steps = pl.read_csv(input_steps)

    print(f"  Time series: {time_series.shape[0]} rows, {time_series.shape[1]} columns")
    print(f"  Steps: {steps.shape[0]} rows, {steps.shape[1]} columns")

    # Convert
    print("\nConverting time series...")
    new_time_series = convert_time_series(time_series)

    print("\nConverting steps...")
    new_steps = convert_steps(steps)

    # Save
    print("\nSaving converted data...")
    output_time_series.parent.mkdir(parents=True, exist_ok=True)
    output_steps.parent.mkdir(parents=True, exist_ok=True)

    new_time_series.write_csv(output_time_series)
    new_steps.write_csv(output_steps)

    print(f"  Saved time series: {output_time_series}")
    print(f"  Saved steps: {output_steps}")

    # Print summary
    print("\n" + "=" * 80)
    print("CONVERSION SUMMARY")
    print("=" * 80)

    print("\nTime Series Columns Added:")
    old_cols = set(time_series.columns)
    new_cols = set(new_time_series.columns)
    added_cols = new_cols - old_cols
    removed_cols = old_cols - new_cols

    if removed_cols:
        print(f"  Removed: {', '.join(removed_cols)}")
    if added_cols:
        print(f"  Added: {', '.join(added_cols)}")

    print("\nSteps Columns Added:")
    old_steps_cols = set(steps.columns)
    new_steps_cols = set(new_steps.columns)
    added_steps_cols = new_steps_cols - old_steps_cols
    removed_steps_cols = old_steps_cols - new_steps_cols

    if removed_steps_cols:
        print(f"  Removed: {', '.join(removed_steps_cols)}")
    if added_steps_cols:
        print(f"  Added: {', '.join(added_steps_cols)}")

    print("\n✅ Conversion complete!")


def convert_directory(input_dir: Path, output_dir: Path):
    """Convert all time_series.csv and steps.csv files in a directory."""
    print(f"\nScanning directory: {input_dir}")

    # Find all time_series.csv files
    time_series_files = list(input_dir.rglob("time_series.csv"))
    print(f"Found {len(time_series_files)} time_series.csv files")

    for ts_file in time_series_files:
        # Find corresponding steps.csv
        steps_file = ts_file.parent / "steps.csv"

        if not steps_file.exists():
            print(f"\n⚠️  Skipping {ts_file} (no corresponding steps.csv)")
            continue

        # Create output paths maintaining directory structure
        rel_path = ts_file.parent.relative_to(input_dir)
        output_ts = output_dir / rel_path / "time_series.csv"
        output_st = output_dir / rel_path / "steps.csv"

        try:
            convert_files(ts_file, steps_file, output_ts, output_st)
        except Exception as e:
            print(f"\n❌ Error converting {ts_file}: {e}")
            continue


def main():
    """Main entry point."""
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    # Check if we're converting directories or files
    if input_path.is_dir():
        if len(sys.argv) != 3:
            print("Error: When converting directories, provide input_dir output_dir")
            sys.exit(1)
        convert_directory(input_path, output_path)
    else:
        # Individual files
        if len(sys.argv) != 5:
            print(
                "Error: When converting files, provide:\n"
                "  input_time_series.csv input_steps.csv "
                "output_time_series.csv output_steps.csv"
            )
            sys.exit(1)

        input_ts = Path(sys.argv[1])
        input_st = Path(sys.argv[2])
        output_ts = Path(sys.argv[3])
        output_st = Path(sys.argv[4])

        if not input_ts.exists():
            print(f"Error: {input_ts} does not exist")
            sys.exit(1)
        if not input_st.exists():
            print(f"Error: {input_st} does not exist")
            sys.exit(1)

        convert_files(input_ts, input_st, output_ts, output_st)


if __name__ == "__main__":
    main()
