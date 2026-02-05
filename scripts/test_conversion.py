#!/usr/bin/env python
"""
Test script to demonstrate and verify the conversion from old to new format.

Creates sample data in old format, converts it, and verifies the results.
"""
#test

import polars as pl
from pathlib import Path
import sys

# Add parent to path to import conversion script
sys.path.insert(0, str(Path(__file__).parent))
from convert_to_new_format import convert_time_series, convert_steps


def create_sample_old_format_data():
    """Create sample data in old format (single Capacity column)."""
    print("Creating sample data in old format...")

    # Time series with cumulative capacity (no step resets)
    time_series = pl.DataFrame(
        {
            "Time [s]": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90],
            "Current [A]": [1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
            "Voltage [V]": [3.8] * 10,
            "Capacity [A.h]": [
                0.0,
                0.003,
                0.006,
                0.009,
                0.012,  # Discharge
                0.009,
                0.006,
                0.003,
                0.0,
                -0.003,
            ],  # Charge (cumulative goes down)
            "Energy [W.h]": [
                0.0,
                0.011,
                0.023,
                0.034,
                0.046,  # Discharge
                0.034,
                0.023,
                0.011,
                0.0,
                -0.011,
            ],  # Charge
            "Power [W]": [3.8, 3.8, 3.8, 3.8, 3.8, -3.8, -3.8, -3.8, -3.8, -3.8],
            "Step from cycler": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
        }
    )

    # Steps with delta capacity
    steps = pl.DataFrame(
        {
            "Step number": [1, 2],
            "Type": ["discharge", "charge"],
            "Mean current [A]": [1.0, -1.0],
            "Mean voltage [V]": [3.8, 3.8],
            "Duration [s]": [40.0, 40.0],
            "Delta capacity [A.h]": [0.012, -0.012],
        }
    )

    return time_series, steps


def verify_conversion(old_ts, old_steps, new_ts, new_steps):
    """Verify the conversion was successful."""
    print("\n" + "=" * 80)
    print("VERIFICATION")
    print("=" * 80)

    errors = []

    # Check time series
    print("\n1. Time Series Conversion:")

    # Check old columns removed
    if "Capacity [A.h]" in new_ts.columns:
        errors.append("  ❌ Old 'Capacity [A.h]' column not removed")
    else:
        print("  ✅ Old 'Capacity [A.h]' column removed")

    if "Energy [W.h]" in new_ts.columns:
        errors.append("  ❌ Old 'Energy [W.h]' column not removed")
    else:
        print("  ✅ Old 'Energy [W.h]' column removed")

    # Check new columns added
    if "Discharge capacity [A.h]" not in new_ts.columns:
        errors.append("  ❌ 'Discharge capacity [A.h]' column not added")
    else:
        print("  ✅ 'Discharge capacity [A.h]' column added")

    if "Charge capacity [A.h]" not in new_ts.columns:
        errors.append("  ❌ 'Charge capacity [A.h]' column not added")
    else:
        print("  ✅ 'Charge capacity [A.h]' column added")

    if "Discharge energy [W.h]" not in new_ts.columns:
        errors.append("  ❌ 'Discharge energy [W.h]' column not added")
    else:
        print("  ✅ 'Discharge energy [W.h]' column added")

    if "Charge energy [W.h]" not in new_ts.columns:
        errors.append("  ❌ 'Charge energy [W.h]' column not added")
    else:
        print("  ✅ 'Charge energy [W.h]' column added")

    # Check step resets
    if "Discharge capacity [A.h]" in new_ts.columns:
        discharge = new_ts.get_column("Discharge capacity [A.h]").to_numpy()
        # First row of step 2 (index 5) should be 0
        if abs(discharge[5]) < 1e-6:
            print("  ✅ Capacity resets at step boundary")
        else:
            errors.append(
                f"  ❌ Capacity doesn't reset at step boundary (got {discharge[5]})"
            )

    # Check steps
    print("\n2. Steps Conversion:")

    # Check old columns removed
    if "Delta capacity [A.h]" in new_steps.columns:
        errors.append("  ❌ Old 'Delta capacity [A.h]' column not removed")
    else:
        print("  ✅ Old 'Delta capacity [A.h]' column removed")

    # Check new columns added
    if "Discharge capacity [A.h]" not in new_steps.columns:
        errors.append("  ❌ 'Discharge capacity [A.h]' column not added")
    else:
        print("  ✅ 'Discharge capacity [A.h]' column added")

    if "Charge capacity [A.h]" not in new_steps.columns:
        errors.append("  ❌ 'Charge capacity [A.h]' column not added")
    else:
        print("  ✅ 'Charge capacity [A.h]' column added")

    # Check values make sense
    if "Discharge capacity [A.h]" in new_steps.columns:
        discharge_cap = new_steps.get_column("Discharge capacity [A.h]").to_numpy()
        charge_cap = new_steps.get_column("Charge capacity [A.h]").to_numpy()

        # Step 1 should be discharge only
        if discharge_cap[0] > 0 and charge_cap[0] == 0:
            print("  ✅ Step 1 correctly identified as discharge")
        else:
            errors.append("  ❌ Step 1 not correctly identified as discharge")

        # Step 2 should be charge only
        if charge_cap[1] > 0 and discharge_cap[1] == 0:
            print("  ✅ Step 2 correctly identified as charge")
        else:
            errors.append("  ❌ Step 2 not correctly identified as charge")

    # Summary
    print("\n" + "=" * 80)
    if errors:
        print("❌ VERIFICATION FAILED")
        for error in errors:
            print(error)
        return False
    else:
        print("✅ ALL CHECKS PASSED")
        return True


def main():
    """Run the test."""
    print("=" * 80)
    print("CONVERSION TEST")
    print("=" * 80)

    # Create sample data
    old_ts, old_steps = create_sample_old_format_data()

    print("\nOld format time series:")
    print(old_ts)

    print("\nOld format steps:")
    print(old_steps)

    # Convert
    print("\n" + "=" * 80)
    print("CONVERTING...")
    print("=" * 80)

    new_ts = convert_time_series(old_ts)
    new_steps = convert_steps(old_steps)

    print("\nNew format time series:")
    print(new_ts)

    print("\nNew format steps:")
    print(new_steps)

    # Verify
    success = verify_conversion(old_ts, old_steps, new_ts, new_steps)

    # Show detailed comparison
    print("\n" + "=" * 80)
    print("DETAILED COMPARISON")
    print("=" * 80)

    print("\nTime Series - Step 1 (Discharge):")
    print(
        new_ts.select(
            [
                "Time [s]",
                "Current [A]",
                "Discharge capacity [A.h]",
                "Charge capacity [A.h]",
            ]
        ).head(5)
    )

    print("\nTime Series - Step 2 (Charge) - showing capacity reset:")
    print(
        new_ts.select(
            [
                "Time [s]",
                "Current [A]",
                "Discharge capacity [A.h]",
                "Charge capacity [A.h]",
            ]
        ).tail(5)
    )

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
