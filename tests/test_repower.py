# Test file for Repower reader
from pathlib import Path

import polars as pl
from datetime import datetime

import ionworksdata as iwdata


def test_repower_basic():
    """Test basic Repower file reading."""
    # Read the Repower file (test auto-detection)
    data = iwdata.read.time_series(
        Path("tests/test_data/repower.csv"),
        options={
            "cell_metadata": {
                "Lower voltage cut-off [V]": 2.5,
                "Upper voltage cut-off [V]": 4.2,
                "Nominal cell capacity [A.h]": 5.0,
            }
        },
    )

    # Verify it returns a Polars DataFrame
    assert isinstance(data, pl.DataFrame), (
        "Repower reader should return Polars DataFrame"
    )

    # Verify expected columns are present
    expected_columns = [
        "Time [s]",
        "Voltage [V]",
        "Current [A]",
        "Temperature [degC]",
        "Status",
        "Step from cycler",
        "Cycle from cycler",
        "Step count",
        "Discharge capacity [A.h]",
        "Charge capacity [A.h]",
        "Discharge energy [W.h]",
        "Charge energy [W.h]",
    ]
    for col in expected_columns:
        assert col in data.columns, f"Expected column '{col}' not found in data"

    # Verify data is not empty
    assert len(data) > 0, "Data should not be empty"

    # Verify Time [s] starts at 0
    assert data["Time [s]"][0] == 0.0, "Time should start at 0"

    # Verify Time [s] is monotonically increasing
    time_values = data["Time [s]"].to_list()
    assert all(
        time_values[i] <= time_values[i + 1] for i in range(len(time_values) - 1)
    ), "Time should be monotonically increasing"


def test_repower_column_mappings():
    """Test that Repower column mappings are applied correctly."""
    data = iwdata.read.time_series(
        Path("tests/test_data/repower.csv"),
        "repower",
        options={
            "cell_metadata": {
                "Lower voltage cut-off [V]": 2.5,
                "Upper voltage cut-off [V]": 4.2,
                "Nominal cell capacity [A.h]": 5.0,
            }
        },
    )

    # Verify column renamings were applied
    assert "Voltage [V]" in data.columns
    assert "Current [A]" in data.columns
    assert "Step from cycler" in data.columns
    assert "Cycle from cycler" in data.columns
    assert "Status" in data.columns

    # Verify old column names are not present
    assert "Voltage(V)" not in data.columns
    assert "Current(A)" not in data.columns
    assert "Step ID" not in data.columns
    assert "Cycle ID" not in data.columns


def test_repower_voltage_filtering():
    """Test that voltage outliers are filtered."""
    # Read with tight voltage limits
    data = iwdata.read.time_series(
        Path("tests/test_data/repower.csv"),
        "repower",
        options={
            "cell_metadata": {
                "Lower voltage cut-off [V]": 3.0,
                "Upper voltage cut-off [V]": 4.0,
                "Nominal cell capacity [A.h]": 5.0,
            }
        },
    )

    # Verify all voltages are within the specified range (with small tolerance)
    voltages = data["Voltage [V]"].to_list()
    assert all(v >= 3.0 * (1 - 1e-3) and v <= 4.0 * (1 + 1e-3) for v in voltages), (
        "All voltages should be within the specified range"
    )


def test_repower_temperature_column():
    """Test that temperature column is present."""
    data = iwdata.read.time_series(
        Path("tests/test_data/repower.csv"),
        "repower",
        options={
            "cell_metadata": {
                "Lower voltage cut-off [V]": 2.5,
                "Upper voltage cut-off [V]": 4.2,
                "Nominal cell capacity [A.h]": 5.0,
            }
        },
    )

    # Verify temperature column exists
    assert "Temperature [degC]" in data.columns

    # Check if temperature values are present (MTV column in the file)
    temp_values = data["Temperature [degC]"]
    # Temperature might be NaN if MTV column wasn't in the file, or have values if it was
    assert temp_values is not None


def test_repower_extra_column_mappings():
    """Test that extra column mappings work."""
    data = iwdata.read.time_series(
        Path("tests/test_data/repower.csv"),
        "repower",
        extra_column_mappings={"POWER(W)": "Power [W]"},
        options={
            "cell_metadata": {
                "Lower voltage cut-off [V]": 2.5,
                "Upper voltage cut-off [V]": 4.2,
                "Nominal cell capacity [A.h]": 5.0,
            }
        },
    )

    # Verify extra column mapping was applied
    assert "Power [W]" in data.columns, "Extra column mapping should be applied"


def test_repower_start_time():
    """Test reading start time from Repower file."""
    start_time = iwdata.read.start_time(
        Path("tests/test_data/repower.csv"),
        "repower",
        options={
            "cell_metadata": {
                "Lower voltage cut-off [V]": 2.5,
                "Upper voltage cut-off [V]": 4.2,
                "Nominal cell capacity [A.h]": 5.0,
            }
        },
    )

    # Verify start time is returned
    assert start_time is not None, "Start time should be returned"
    assert isinstance(start_time, datetime), "Start time should be a datetime object"

    # Verify it's timezone-aware
    assert start_time.tzinfo is not None, "Start time should be timezone-aware"

    # Verify it matches the first timestamp in the file (2024-04-30 12:51:30)
    assert start_time.year == 2024
    assert start_time.month == 4
    assert start_time.day == 30


def test_repower_data_types():
    """Test that data types are correct."""
    data = iwdata.read.time_series(
        Path("tests/test_data/repower.csv"),
        "repower",
        options={
            "cell_metadata": {
                "Lower voltage cut-off [V]": 2.5,
                "Upper voltage cut-off [V]": 4.2,
                "Nominal cell capacity [A.h]": 5.0,
            }
        },
    )

    # Verify numeric columns are numeric
    assert data["Time [s]"].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]
    assert data["Voltage [V]"].dtype in [pl.Float64, pl.Float32]
    assert data["Current [A]"].dtype in [pl.Float64, pl.Float32]

    # Verify Step from cycler can be integer
    assert data["Step from cycler"].dtype in [
        pl.Int64,
        pl.Int32,
        pl.Float64,
        pl.Float32,
    ]


def test_repower_decimal_current_values_preserved():
    """Test that decimal current values are not truncated to integers.

    This is a regression test for a bug where Polars would infer Current(A)
    as Int64 when the first rows had integer values (0), causing subsequent
    decimal values like 0.60531 to be truncated to 0.
    """
    data = iwdata.read.time_series(
        Path("tests/test_data/repower.csv"),
        "repower",
        options={
            "cell_metadata": {
                "Lower voltage cut-off [V]": 2.5,
                "Upper voltage cut-off [V]": 4.2,
                "Nominal cell capacity [A.h]": 5.0,
            }
        },
    )

    # The test file has decimal current values like 0.60531 and 1.66961
    # that should be preserved, not truncated to 0
    current = data["Current [A]"]

    # Verify there are non-zero current values
    non_zero_current = current.filter(current.abs() > 0.01)
    assert len(non_zero_current) > 0, (
        "Expected non-zero current values but found none. "
        "This may indicate decimal values were truncated to integers."
    )

    # Verify the non-zero values have decimal precision
    # (i.e., they're not just integers like 1 or 2)
    has_decimal_values = any(
        abs(v) > 0 and abs(v) != round(abs(v)) for v in non_zero_current.to_list()
    )
    assert has_decimal_values, (
        "Non-zero current values appear to be integers, not decimals. "
        "Expected values like 0.60531 or 1.66961 to be preserved."
    )


def test_repower_returns_polars():
    """Test that Repower reader returns Polars DataFrame."""
    data = iwdata.read.time_series(
        Path("tests/test_data/repower.csv"),
        "repower",
        options={
            "cell_metadata": {
                "Lower voltage cut-off [V]": 2.5,
                "Upper voltage cut-off [V]": 4.2,
                "Nominal cell capacity [A.h]": 5.0,
            }
        },
    )

    assert isinstance(data, pl.DataFrame), (
        "Repower reader should return Polars DataFrame"
    )
    assert len(data) > 0, "Data should not be empty"
    assert "Time [s]" in data.columns
    assert "Voltage [V]" in data.columns
    assert "Current [A]" in data.columns


def test_repower_real_data():
    """Test reading a real Repower file using measurement_details"""
    import pytest
    import pandas as pd

    p = Path("tests/test_data/repower.csv")

    # Read raw file once to get original capacity columns for comparison
    raw_df = pd.read_csv(p, encoding="latin1", on_bad_lines="skip")
    # Filter out header rows and rows with NaN cycle ID
    raw_df = raw_df[raw_df["Cycle ID"].notna()]
    raw_df = raw_df[raw_df["Charge Capacity(Ah)"].notna()]
    raw_df = raw_df[raw_df["Discharge Capacity(Ah)"].notna()]

    # Generate measurement_details once and use it throughout the test
    measurement = {"name": "00"}
    measurement_details = iwdata.read.measurement_details(
        p,
        measurement,
        "repower",
        options={
            "cell_metadata": {
                "Lower voltage cut-off [V]": 2.5,
                "Upper voltage cut-off [V]": 4.2,
                "Nominal cell capacity [A.h]": 5.0,
            },
            "timezone": "UTC",
        },
    )

    # Check that measurement_details returns the expected structure
    assert "time_series" in measurement_details
    assert "steps" in measurement_details
    assert "measurement" in measurement_details

    # Extract components
    df = measurement_details["time_series"]
    steps = measurement_details["steps"]
    measurement_dict = measurement_details["measurement"]

    # Check measurement metadata
    assert measurement_dict["cycler"] == "repower"
    assert "start_time" in measurement_dict
    # Start time should be in ISO format
    assert "T" in measurement_dict["start_time"]

    # Check that required columns are present in time series
    required_cols = {
        "Time [s]",
        "Voltage [V]",
        "Current [A]",
        "Step count",
    }
    assert required_cols.issubset(set(df.columns))

    # Check that data was read
    assert len(df) > 0

    # Check that time starts at 0 (or close to it)
    assert df["Time [s]"][0] == pytest.approx(0.0, abs=1.0)

    # Check that time is monotonically increasing
    time_diffs = df["Time [s]"].diff()
    assert (time_diffs[1:] >= 0).all()

    # Check voltage is in reasonable range
    assert df["Voltage [V]"].min() > 0
    assert df["Voltage [V]"].max() < 5.0

    # Check that step count is present and valid
    assert df["Step count"].min() >= 0

    # Check steps DataFrame structure
    assert len(steps) > 0
    assert "Step count" in steps.columns
    assert "Duration [s]" in steps.columns

    # Verify that capacity values from file are used (not recalculated)
    # Repower has separate charge/discharge columns, so verify those match directly
    # Verify that we have capacity columns
    assert "Charge capacity [A.h]" in df.columns
    assert "Discharge capacity [A.h]" in df.columns

    # Get charge and discharge capacities from processed data
    charge_cap = df["Charge capacity [A.h]"].to_numpy()
    discharge_cap = df["Discharge capacity [A.h]"].to_numpy()

    # Get charge and discharge capacities from raw file
    raw_charge_cap = raw_df["Charge Capacity(Ah)"].values
    raw_discharge_cap = raw_df["Discharge Capacity(Ah)"].values

    # Sample some data points and verify that charge/discharge capacities
    # match the original file values directly
    # Skip initial zeros and check a range with non-zero values
    start_idx = 100  # Skip first 100 rows which may be zeros after step resets
    end_idx = min(start_idx + 200, len(df), len(raw_df))
    sample_indices = range(start_idx, end_idx)

    # Get capacities for sample
    sample_charge = charge_cap[sample_indices]
    sample_discharge = discharge_cap[sample_indices]
    sample_raw_charge = raw_charge_cap[sample_indices]
    sample_raw_discharge = raw_discharge_cap[sample_indices]

    # Find points where both have non-zero values
    non_zero_mask = ((sample_charge > 1e-8) | (sample_discharge > 1e-8)) & (
        (sample_raw_charge > 1e-8) | (sample_raw_discharge > 1e-8)
    )
    non_zero_count = non_zero_mask.sum()

    # Only verify if we have non-zero values in the raw file
    if non_zero_count > 10:
        assert sample_charge[non_zero_mask] == pytest.approx(
            sample_raw_charge[non_zero_mask], abs=1e-6
        ), "Charge capacity values from file don't match"
        assert sample_discharge[non_zero_mask] == pytest.approx(
            sample_raw_discharge[non_zero_mask], abs=1e-6
        ), "Discharge capacity values from file don't match"
    else:
        # If raw file doesn't have capacity values, just verify columns exist
        assert "Charge capacity [A.h]" in df.columns
        assert "Discharge capacity [A.h]" in df.columns

    # Test with custom timezone
    measurement_tz = {"name": "00"}
    measurement_details_tz = iwdata.read.measurement_details(
        p,
        measurement_tz,
        "repower",
        options={
            "timezone": "US/Pacific",
            "cell_metadata": {
                "Lower voltage cut-off [V]": 2.5,
                "Upper voltage cut-off [V]": 4.2,
                "Nominal cell capacity [A.h]": 5.0,
            },
        },
    )
    assert measurement_details_tz["measurement"]["start_time"] is not None
    # The start_time should be in ISO format
    assert "T" in measurement_details_tz["measurement"]["start_time"]
