import pandas as pd
import pytest
import ionworksdata as iwdata
from datetime import datetime, timezone


def test_neware_excel_single_sheet(tmp_path):
    """Test reading a single sheet from Excel file"""
    # Create test data
    test_data = pd.DataFrame(
        {
            "DateTime": ["2021-01-01 00:00:00", "2021-01-01 00:00:01"],
            "Current(A)": [2.0, 3.0],  # Test Current(A) mapping
            "Voltage (V)": [4.0, 3.0],
            "Step ID": [0, 1],
            "Cycle ID": [0, 0],
            "Status": ["D", "C"],
        }
    )

    # Save as Excel file
    excel_path = tmp_path / "test.xlsx"
    test_data.to_excel(excel_path, sheet_name="Data", index=False)

    # Read with Neware reader
    df_read = iwdata.read.time_series(excel_path, "neware")

    # Verify output
    assert "Time [s]" in df_read.columns
    assert "Timestamp" not in df_read.columns  # Should be converted to Time [s]
    assert df_read["Time [s]"].to_list() == [0.0, 1.0]
    assert "Sheet" not in df_read.columns  # Single sheet, no Sheet column


def test_neware_excel_specific_sheet_names(tmp_path):
    """Test reading specific sheets by name"""
    # Create test data for multiple sheets
    test_data1 = pd.DataFrame(
        {
            "DateTime": ["2021-01-01 00:00:00", "2021-01-01 00:00:01"],
            "Current (A)": [2.0, 3.0],
            "Voltage (V)": [4.0, 3.0],
            "Step ID": [0, 1],
        }
    )

    test_data2 = pd.DataFrame(
        {
            "DateTime": ["2021-01-01 01:00:00", "2021-01-01 01:00:01"],
            "Current (A)": [5.0, 6.0],
            "Voltage (V)": [3.5, 2.5],
            "Step ID": [2, 3],
        }
    )

    # Save as Excel file with multiple sheets
    excel_path = tmp_path / "test.xlsx"
    with pd.ExcelWriter(excel_path) as writer:
        test_data1.to_excel(writer, sheet_name="Test1", index=False)
        test_data2.to_excel(writer, sheet_name="Test2", index=False)
        test_data1.to_excel(writer, sheet_name="Other", index=False)

    # Read specific sheets using options
    df_read = iwdata.read.time_series(
        excel_path,
        "neware",
        options={"sheets": {"type": "name", "value": ["Test1", "Test2"]}},
    )

    # Verify output
    assert "Sheet" not in df_read.columns  # Sheet column is filtered out in processing
    assert len(df_read) == 4  # 2 rows from each sheet
    assert "Time [s]" in df_read.columns


def test_neware_excel_sheet_pattern(tmp_path):
    """Test reading sheets by regex pattern"""
    # Create test data for multiple sheets with different timestamps
    test_data1 = pd.DataFrame(
        {
            "DateTime": ["2021-01-01 00:00:00", "2021-01-01 00:00:01"],
            "Current (A)": [2.0, 3.0],
            "Voltage (V)": [4.0, 3.0],
            "Step ID": [0, 1],
        }
    )

    test_data2 = pd.DataFrame(
        {
            "DateTime": ["2021-01-01 00:00:02", "2021-01-01 00:00:03"],
            "Current (A)": [5.0, 6.0],
            "Voltage (V)": [3.5, 2.5],
            "Step ID": [2, 3],
        }
    )

    # Save as Excel file with multiple sheets
    excel_path = tmp_path / "test.xlsx"
    with pd.ExcelWriter(excel_path) as writer:
        test_data1.to_excel(writer, sheet_name="Data_001", index=False)
        test_data2.to_excel(writer, sheet_name="Data_002", index=False)
        test_data1.to_excel(writer, sheet_name="Other", index=False)

    # Read sheets matching pattern
    df_read = iwdata.read.time_series(
        excel_path,
        "neware",
        options={"sheets": {"type": "pattern", "value": r"Data_\d+"}},
    )

    # Verify output
    assert "Sheet" not in df_read.columns  # Sheet column is filtered out in processing
    assert len(df_read) == 4  # 2 rows from each matched sheet


def test_neware_excel_combined_sheet_selection(tmp_path):
    """Test pattern matching multiple different sheets"""
    # Create test data with different timestamps to avoid conflicts
    test_data1 = pd.DataFrame(
        {
            "DateTime": ["2021-01-01 00:00:00"],
            "Current (A)": [2.0],
            "Voltage (V)": [4.0],
            "Step ID": [0],
        }
    )

    test_data2 = pd.DataFrame(
        {
            "DateTime": ["2021-01-01 00:00:01"],
            "Current (A)": [3.0],
            "Voltage (V)": [3.5],
            "Step ID": [1],
        }
    )

    test_data3 = pd.DataFrame(
        {
            "DateTime": ["2021-01-01 00:00:02"],
            "Current (A)": [4.0],
            "Voltage (V)": [3.0],
            "Step ID": [2],
        }
    )

    # Save as Excel file with multiple sheets
    excel_path = tmp_path / "test.xlsx"
    with pd.ExcelWriter(excel_path) as writer:
        test_data1.to_excel(writer, sheet_name="Specific", index=False)
        test_data2.to_excel(writer, sheet_name="Test_001", index=False)
        test_data3.to_excel(writer, sheet_name="Test_002", index=False)
        test_data1.to_excel(writer, sheet_name="Other", index=False)

    # Test reading by pattern that matches multiple sheets
    df_read = iwdata.read.time_series(
        excel_path,
        "neware",
        options={"sheets": {"type": "pattern", "value": r"(Specific|Test_\d+)"}},
    )

    # Verify output
    assert "Sheet" not in df_read.columns  # Sheet column is filtered out in processing
    assert len(df_read) == 3  # 1 row from each sheet


def test_neware_excel_sheet_not_found(tmp_path):
    """Test error when specified sheet doesn't exist"""
    test_data = pd.DataFrame(
        {
            "DateTime": ["2021-01-01 00:00:00"],
            "Current (A)": [2.0],
        }
    )

    excel_path = tmp_path / "test.xlsx"
    test_data.to_excel(excel_path, sheet_name="Data", index=False)

    # Try to read non-existent sheet
    with pytest.raises(ValueError, match="Sheet 'NonExistent' not found"):
        iwdata.read.time_series(
            excel_path,
            "neware",
            options={"sheets": {"type": "name", "value": ["NonExistent"]}},
        )


def test_neware_excel_no_matching_sheets(tmp_path):
    """Test error when pattern matches no sheets"""
    test_data = pd.DataFrame(
        {
            "DateTime": ["2021-01-01 00:00:00"],
            "Current (A)": [2.0],
        }
    )

    excel_path = tmp_path / "test.xlsx"
    test_data.to_excel(excel_path, sheet_name="Data", index=False)

    # Try pattern that matches nothing
    with pytest.raises(ValueError, match="No sheets found matching pattern"):
        iwdata.read.time_series(
            excel_path,
            "neware",
            options={"sheets": {"type": "pattern", "value": r"NonExistent_\d+"}},
        )


def test_neware_csv_with_sheet_params(tmp_path):
    """Test error when using sheet parameters with CSV"""
    test_data = pd.DataFrame(
        {
            "DateTime": ["2021-01-01 00:00:00"],
            "Current (A)": [2.0],
        }
    )

    csv_path = tmp_path / "test.csv"
    test_data.to_csv(csv_path, index=False)

    # Try to use sheet params with CSV
    with pytest.raises(
        ValueError, match="Sheet selection is only supported for Excel files"
    ):
        iwdata.read.time_series(
            csv_path, "neware", options={"sheets": {"type": "name", "value": ["Test"]}}
        )


def test_neware_excel_read_start_time(tmp_path):
    """Test reading start time from Excel file"""
    test_data = pd.DataFrame(
        {
            "Date(h:min:s.ms)": [
                "2021-01-01 00:00:00",
                "2021-01-01 00:00:01",
            ],  # Test Date(h:min:s.ms) mapping
        }
    )

    excel_path = tmp_path / "test.xlsx"
    test_data.to_excel(excel_path, sheet_name="Data", index=False)

    reader = iwdata.read.Neware()
    start_time = reader.read_start_time(excel_path)

    assert start_time == datetime(2021, 1, 1, tzinfo=timezone.utc)


def test_neware_excel_all_sheets(tmp_path):
    """Test reading all sheets"""
    test_data1 = pd.DataFrame(
        {
            "DateTime": ["2021-01-01 00:00:00"],
            "Current (A)": [2.0],
            "Voltage (V)": [4.0],
            "Step ID": [0],
        }
    )

    test_data2 = pd.DataFrame(
        {
            "DateTime": ["2021-01-01 00:00:01"],
            "Current (A)": [3.0],
            "Voltage (V)": [3.5],
            "Step ID": [1],
        }
    )

    test_data3 = pd.DataFrame(
        {
            "DateTime": ["2021-01-01 00:00:02"],
            "Current (A)": [4.0],
            "Voltage (V)": [3.0],
            "Step ID": [2],
        }
    )

    # Save as Excel file with multiple sheets
    excel_path = tmp_path / "test.xlsx"
    with pd.ExcelWriter(excel_path) as writer:
        test_data1.to_excel(writer, sheet_name="Sheet1", index=False)
        test_data2.to_excel(writer, sheet_name="Sheet2", index=False)
        test_data3.to_excel(writer, sheet_name="Sheet3", index=False)

    # Read all sheets
    df_read = iwdata.read.time_series(
        excel_path, "neware", options={"sheets": {"type": "all"}}
    )

    # Verify output
    assert "Sheet" not in df_read.columns  # Sheet column is filtered out in processing
    assert len(df_read) == 3  # 1 row from each sheet


def test_neware_excel_invalid_sheet_spec(tmp_path):
    """Test error handling for invalid sheet specifications"""
    test_data = pd.DataFrame(
        {
            "DateTime": ["2021-01-01 00:00:00"],
            "Current (A)": [2.0],
        }
    )

    excel_path = tmp_path / "test.xlsx"
    test_data.to_excel(excel_path, sheet_name="Data", index=False)

    # Test invalid structure
    with pytest.raises(ValueError, match="'sheets' must be a dictionary"):
        iwdata.read.time_series(excel_path, "neware", options={"sheets": "invalid"})

    # Test missing required value for name type
    with pytest.raises(
        ValueError, match="For 'name' type, 'value' must be a sheet name or list"
    ):
        iwdata.read.time_series(
            excel_path, "neware", options={"sheets": {"type": "name"}}
        )

    # Test invalid type
    with pytest.raises(ValueError, match="Unsupported sheet type 'invalid'"):
        iwdata.read.time_series(
            excel_path,
            "neware",
            options={"sheets": {"type": "invalid", "value": []}},
        )


def test_neware_excel_single_sheet_name_as_string(tmp_path):
    """Test reading a single sheet specified as a string (not list)"""
    test_data = pd.DataFrame(
        {
            "DateTime": ["2021-01-01 00:00:00", "2021-01-01 00:00:01"],
            "Current (A)": [2.0, 3.0],
            "Voltage (V)": [4.0, 3.0],
            "Step ID": [0, 1],
        }
    )

    excel_path = tmp_path / "test.xlsx"
    test_data.to_excel(excel_path, sheet_name="Data", index=False)

    # Read using string value instead of list
    df_read = iwdata.read.time_series(
        excel_path, "neware", options={"sheets": {"type": "name", "value": "Data"}}
    )

    # Verify output
    assert "Time [s]" in df_read.columns
    assert len(df_read) == 2
    assert df_read["Time [s]"].to_list() == [0.0, 1.0]


def test_neware_excel_filter_1970_timestamps(tmp_path):
    """Test filtering out January 1970 timestamps when first valid timestamp is after 1970"""
    test_data = pd.DataFrame(
        {
            "DateTime": [
                "1970-01-01 00:00:00",  # Should be filtered out
                "1970-01-15 12:00:00",  # Should be filtered out
                "2021-01-01 00:00:00",  # Valid timestamp
                "2021-01-01 00:00:01",  # Valid timestamp
            ],
            "Current (A)": [1.0, 2.0, 3.0, 4.0],
            "Voltage (V)": [3.0, 3.5, 4.0, 3.8],
            "Step ID": [0, 1, 2, 3],
        }
    )

    excel_path = tmp_path / "test.xlsx"
    test_data.to_excel(excel_path, sheet_name="Data", index=False)

    # Read with Neware reader
    df_read = iwdata.read.time_series(excel_path, "neware")

    # Verify 1970 timestamps were filtered out
    assert len(df_read) == 2  # Only 2021 timestamps should remain
    assert df_read["Time [s]"].to_list() == [0.0, 1.0]

    # Test start time reading
    reader = iwdata.read.Neware()
    start_time = reader.read_start_time(excel_path)
    assert start_time == datetime(2021, 1, 1, tzinfo=timezone.utc)


def test_neware_excel_keep_1970_timestamps_if_all_data_from_1970(tmp_path):
    """Test keeping 1970 timestamps if all data is from 1970 (legitimate old data)"""
    test_data = pd.DataFrame(
        {
            "DateTime": [
                "1970-06-01 00:00:00",  # Valid 1970 data (after January)
                "1970-06-01 00:00:01",  # Valid 1970 data
            ],
            "Current (A)": [2.0, 3.0],
            "Voltage (V)": [4.0, 3.5],
            "Step ID": [0, 1],
        }
    )

    excel_path = tmp_path / "test.xlsx"
    test_data.to_excel(excel_path, sheet_name="Data", index=False)

    # Read with Neware reader
    df_read = iwdata.read.time_series(excel_path, "neware")

    # Verify all data is kept (legitimate 1970 data)
    assert len(df_read) == 2
    assert df_read["Time [s]"].to_list() == [0.0, 1.0]


def test_neware_excel_read_start_time_with_sheets(tmp_path):
    """Test reading start time from specific Excel sheet"""
    test_data1 = pd.DataFrame(
        {
            "DateTime": ["2021-01-01 02:00:00"],  # Later time
        }
    )
    test_data2 = pd.DataFrame(
        {
            "DateTime": ["2021-01-01 01:00:00"],  # Earlier time
        }
    )

    excel_path = tmp_path / "test.xlsx"
    with pd.ExcelWriter(excel_path) as writer:
        test_data1.to_excel(writer, sheet_name="Sheet1", index=False)
        test_data2.to_excel(writer, sheet_name="Sheet2", index=False)

    reader = iwdata.read.Neware()
    # Read from Sheet2 only (earlier time)
    start_time = reader.read_start_time(
        excel_path,
        extra_column_mappings=None,
        options={"sheets": {"type": "name", "value": ["Sheet2"]}},
    )

    assert start_time == datetime(2021, 1, 1, 1, 0, 0, tzinfo=timezone.utc)


def test_neware_real_data():
    """Test reading a real Neware Excel file using measurement_details"""
    from pathlib import Path

    p = Path("tests/test_data/neware.xlsx")

    # Test reading with measurement_details using pattern matching for Detail sheets
    measurement = {"name": "00"}
    measurement_details = iwdata.read.measurement_details(
        p,
        measurement,
        "neware",
        options={
            "cell_metadata": {"Nominal cell capacity [A.h]": 1.0},
            "sheets": {"type": "pattern", "value": "Detail_*"},
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
    assert measurement_dict["cycler"] == "neware"
    assert "start_time" in measurement_dict
    # Start time should be in ISO format
    assert "T" in measurement_dict["start_time"]

    # Check that required columns are present in time series
    # measurement_details with keep_only_required_columns=True filters to:
    # Time [s], Current [A], Voltage [V], Temperature [degC], Step count
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

    # Note: Neware reader doesn't support timezone option directly
    # The start_time is read from the DateTime column in the file


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
