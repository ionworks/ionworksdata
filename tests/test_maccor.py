# pyright: reportMissingTypeStubs=false
from datetime import datetime, timezone

import pandas as pd  # type: ignore[reportMissingTypeStubs]
import polars as pl
import pytest
from pathlib import Path

import ionworksdata as iwdata


@pytest.mark.parametrize("time_column", ["Test (Sec)", "Test Time (Hr)"])
def test_maccor(tmp_path, time_column):
    if time_column == "Test (Sec)":
        times = [0.0, 1.0]
    elif time_column == "Test Time (Hr)":
        times = [0.0, 1.0 / 3600]
    test_data = pl.DataFrame(
        {
            time_column: times,
            "Current (A)": [2.0, 3.0],
            "Voltage (V)": [4.0, 3.0],
            "Cycle": [0, 0],
            "Step": [0, 1],
            "LogTemp001": [20.0, 20.0],
            "Status": ["D", "C"],
        }
    )
    test_data.write_csv(tmp_path / "test.txt", separator="\t")
    expected_data_pl = pl.DataFrame(
        {
            "Time [s]": [0.0, 1.0],
            "Current [A]": [2.0, -3.0],
            "Voltage [V]": [4.0, 3.0],
            "Step from cycler": [0, 1],
            "Cycle from cycler": [0, 0],
            "Temperature [degC]": [20.0, 20.0],
            "Step count": [0, 1],
        }
    )
    expected_data_pl = iwdata.transform.set_cycle_count(expected_data_pl)
    expected_data_pl = iwdata.transform.set_capacity(expected_data_pl, options=None)
    expected_data_pl = iwdata.transform.set_energy(expected_data_pl, options=None)
    # Keep explicit reader for this test (file format is complex)
    df_read = iwdata.read.time_series(tmp_path / "test.txt", "maccor")
    pd.testing.assert_frame_equal(
        expected_data_pl.to_pandas().sort_index(axis=1),
        df_read.to_pandas().sort_index(axis=1),
    )


def test_maccor_csv(tmp_path):
    test_data = pl.DataFrame(
        {
            "Prog Time": [0.0, 1.0],
            "Current": [2.0, 3.0],
            "Voltage": [4.0, 3.0],
            "Cycle": [0, 0],
            "Step": [0, 1],
            "LogTemp001": [20.0, 20.0],
            "Status": ["D", "C"],
        }
    )
    # Write CSV with extra header row (units row) above column names
    with open(tmp_path / "test.csv", "w") as f:
        # Write header row
        f.write(",".join(test_data.columns) + "\n")
        # Write units row (empty for all columns)
        f.write(",".join([""] * len(test_data.columns)) + "\n")
        # Write data
        test_data.write_csv(f, separator=",", include_header=False)
    expected_data_pl = pl.DataFrame(
        {
            "Time [s]": [0.0, 1.0],
            "Current [A]": [2.0, -3.0],
            "Voltage [V]": [4.0, 3.0],
            "Step from cycler": [0, 1],
            "Cycle from cycler": [0, 0],
            "Temperature [degC]": [20.0, 20.0],
            "Step count": [0, 1],
        }
    )
    expected_data_pl = iwdata.transform.set_cycle_count(expected_data_pl)
    expected_data_pl = iwdata.transform.set_capacity(expected_data_pl, options=None)
    expected_data_pl = iwdata.transform.set_energy(expected_data_pl, options=None)
    df_read = iwdata.read.time_series(tmp_path / "test.csv", "maccor")
    pd.testing.assert_frame_equal(
        expected_data_pl.to_pandas().sort_index(axis=1),
        df_read.to_pandas().sort_index(axis=1),
    )


def test_maccor_msg(tmp_path):
    test_data = pl.DataFrame(
        {
            "Test (Sec)": [0.0, 1.0, 2.0],
            "Current (A)": [None, 3.0, 4.0],
            "Voltage (V)": [None, 4.1, 4.2],
            "Cycle": [0, 0, 0],
            "Step": [0, 1, 1],
            "LogTemp001": [None, 20.0, 20.0],
            "Status": ["MSG", "CHA", "CHA"],
        }
    )
    test_data.write_csv(tmp_path / "test.txt", separator="\t")
    expected_data_pl = pl.DataFrame(
        {
            "Time [s]": [0.0, 1.0],
            "Current [A]": [-3.0, -4.0],
            "Voltage [V]": [4.1, 4.2],
            "Step from cycler": [1, 1],
            "Cycle from cycler": [0, 0],
            "Temperature [degC]": [20.0, 20.0],
            "Step count": [0, 0],
        }
    )
    expected_data_pl = iwdata.transform.set_cycle_count(expected_data_pl)
    expected_data_pl = iwdata.transform.set_capacity(expected_data_pl, options=None)
    expected_data_pl = iwdata.transform.set_energy(expected_data_pl, options=None)
    df_read = iwdata.read.time_series(tmp_path / "test.txt", "maccor")
    pd.testing.assert_frame_equal(
        expected_data_pl.to_pandas().sort_index(axis=1),
        df_read.to_pandas().sort_index(axis=1),
    )


def test_maccor_unsigned_current(tmp_path):
    test_data = pl.DataFrame(
        {
            "Test (Sec)": [0.0, 1.0, 2.0, 3.0, 4.0],
            "Current (A)": [0.0, 2.0, 3.0, 1.0, 2.0],
            "Voltage (V)": [4.0, 3.0, 2.0, 3.0, 3.5],
            "Cycle": [0, 0, 0, 0, 0],
            "Step": [0, 0, 0, 1, 1],
            "LogTemp001": [20.0, 20.0, 20.0, 20.0, 20.0],
            "Status": ["D", "D", "D", "C", "C"],
        }
    )
    test_data.write_csv(tmp_path / "test.txt", separator="\t")
    expected_data_pl = pl.DataFrame(
        {
            "Time [s]": [0.0, 1.0, 2.0, 3.0, 4.0],
            "Current [A]": [0.0, 2.0, 3.0, -1.0, -2.0],
            "Voltage [V]": [4.0, 3.0, 2.0, 3.0, 3.5],
            "Step from cycler": [0, 0, 0, 1, 1],
            "Cycle from cycler": [0, 0, 0, 0, 0],
            "Temperature [degC]": [20.0, 20.0, 20.0, 20.0, 20.0],
            "Step count": [0, 0, 0, 1, 1],
        }
    )
    expected_data_pl = iwdata.transform.set_cycle_count(expected_data_pl)
    expected_data_pl = iwdata.transform.set_capacity(expected_data_pl, options=None)
    expected_data_pl = iwdata.transform.set_energy(expected_data_pl, options=None)
    df_read = iwdata.read.time_series(tmp_path / "test.txt", "maccor")
    pd.testing.assert_frame_equal(
        expected_data_pl.to_pandas().sort_index(axis=1),
        df_read.to_pandas().sort_index(axis=1),
    )


def test_maccor_bad_extension(tmp_path):
    test_data = pl.DataFrame(
        {
            "Test (Sec)": [0.0, 1.0],
        }
    )
    test_data.write_csv(tmp_path / "test.bad", separator="\t")
    with pytest.raises(ValueError, match="Unsupported file extension: .bad"):
        iwdata.read.time_series(tmp_path / "test.bad", "maccor")


def test_maccor_no_header(tmp_path):
    test_data = pl.DataFrame(
        {
            "Test (Sec)": [0.0, 1.0],
        }
    )
    test_data.write_csv(tmp_path / "test.txt", separator="\t")
    with pytest.raises(ValueError, match="Could not find header row in Maccor file"):
        iwdata.read.time_series(tmp_path / "test.txt", "maccor")


def test_maccor_read_header_single_line(tmp_path):
    """Test read_header method with single line header (CSV format)"""
    # Create a CSV file with single line header
    with open(tmp_path / "test.csv", "w") as f:
        f.write("Step,Test (Sec),Current (A),Voltage (V)\n")
        f.write("0,0.0,2.0,4.0\n")
        f.write("1,1.0,3.0,3.0\n")

    reader = iwdata.read.Maccor()
    header = reader.read_header(tmp_path / "test.csv")
    assert "Step" in header


def test_maccor_read_header_multi_line(tmp_path):
    """Test read_header method with multi-line header (TXT format)"""
    # Create a TXT file with multi-line header
    with open(tmp_path / "test.txt", "w") as f:
        f.write("Header line 1\n")
        f.write("Header line 2\n")
        f.write("Step\tTest (Sec)\tCurrent (A)\tVoltage (V)\n")
        f.write("0\t0.0\t2.0\t4.0\n")
        f.write("1\t1.0\t3.0\t3.0\n")

    reader = iwdata.read.Maccor()
    header = reader.read_header(tmp_path / "test.txt")
    assert "Header line 1" in header
    assert "Header line 2" in header


def test_maccor_read_start_time_with_date_format(tmp_path):
    """Test read_start_time with different date formats"""
    # Test with "%d %B %Y, %I:%M:%S %p" format
    with open(tmp_path / "test.txt", "w") as f:
        f.write("Date of Test:\t23 May 2025, 10:00:00 AM\tFilename:\ttest.txt\n")
        f.write("Step\n")

    start_time = iwdata.read.Maccor().read_start_time(tmp_path / "test.txt")
    assert start_time == datetime(2025, 5, 23, 10, 0, 0, tzinfo=timezone.utc)


def test_maccor_read_start_time_with_date_only(tmp_path):
    """Test read_start_time with date only format"""
    # Test with "%m/%d/%Y" format (date only)
    with open(tmp_path / "test.txt", "w") as f:
        f.write("Date of Test:\t05/23/2025\tFilename:\ttest.txt\n")
        f.write("Step\n")

    start_time = iwdata.read.Maccor().read_start_time(tmp_path / "test.txt")
    assert start_time == datetime(2025, 5, 23, 0, 0, 0, tzinfo=timezone.utc)


def test_maccor_read_start_time_with_custom_timezone(tmp_path):
    """Test read_start_time with custom timezone"""
    with open(tmp_path / "test.txt", "w") as f:
        f.write("Date of Test:\t23 May 2025, 10:00:00 AM\tFilename:\ttest.txt\n")
        f.write("Step\n")

    start_time = iwdata.read.Maccor().read_start_time(
        tmp_path / "test.txt",
        extra_column_mappings=None,
        options={"timezone": "America/New_York"},
    )
    # Should convert to UTC
    assert start_time.tzinfo == timezone.utc


def test_maccor_read_start_time_invalid_timezone(tmp_path):
    """Test read_start_time with invalid timezone"""
    with open(tmp_path / "test.txt", "w") as f:
        f.write("Date of Test:\t23 May 2025, 10:00:00 AM\tFilename:\ttest.txt\n")
        f.write("Step\n")

    with pytest.raises(ValueError, match="Invalid timezone"):
        iwdata.read.Maccor().read_start_time(
            tmp_path / "test.txt",
            extra_column_mappings=None,
            options={"timezone": 123},  # Invalid timezone type
        )


def test_maccor_read_start_time_no_date(tmp_path):
    """Test read_start_time when no date is found"""
    with open(tmp_path / "test.txt", "w") as f:
        f.write("No date information here\n")
        f.write("Step\n")

    start_time = iwdata.read.Maccor().read_start_time(tmp_path / "test.txt")
    assert start_time is None


def test_maccor_excel_2_row_header():
    """Test reading a Maccor file with 2 row header"""
    p = Path("tests/test_data/maccor_2_row_header.xlsx")
    df_read, steps_read = iwdata.read.time_series_and_steps(p)
    assert df_read is not None
    assert steps_read is not None
    assert len(steps_read) == 2
    assert "Time [s]" in df_read.columns
    assert "Current [A]" in df_read.columns
    assert "Voltage [V]" in df_read.columns


def test_maccor_columns_keep_logic(tmp_path):
    """Test that Time [h] and Status columns are properly excluded from kept columns"""
    test_data = pl.DataFrame(
        {
            "Test Time (Hr)": [
                0.0,
                1.0 / 3600,
            ],  # This should become "Time [h]" then "Time [s]"
            "Current (A)": [2.0, 3.0],
            "Voltage (V)": [4.0, 3.0],
            "Cycle": [0, 0],
            "Step": [0, 1],
            "Status": ["D", "C"],  # This should be dropped
            "LogTemp001": [20.0, 20.0],
        }
    )
    test_data.write_csv(tmp_path / "test.txt", separator="\t")

    df_read = iwdata.read.time_series(tmp_path / "test.txt", "maccor")

    # Check that Status column is not in the output
    assert "Status" not in df_read.columns
    # Check that Time [h] is not in the output (should be converted to Time [s])
    assert "Time [h]" not in df_read.columns
    # Check that other expected columns are present
    expected_columns = [
        "Time [s]",
        "Current [A]",
        "Voltage [V]",
        "Step from cycler",
        "Cycle from cycler",
        "Temperature [degC]",
        "Step count",
        "Discharge capacity [A.h]",
        "Charge capacity [A.h]",
        "Discharge energy [W.h]",
        "Charge energy [W.h]",
    ]
    for col in expected_columns:
        assert col in df_read.columns


def test_maccor_real_data():
    """Test reading a real Maccor file using measurement_details"""
    from pathlib import Path

    p = Path("tests/test_data/maccor.txt")

    # Test reading with measurement_details
    measurement = {"name": "00"}
    measurement_details = iwdata.read.measurement_details(
        p,
        measurement,
        "maccor",
        options={
            "cell_metadata": {"Nominal cell capacity [A.h]": 1.0},
            "file_encoding": "ISO-8859-1",
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
    assert measurement_dict["cycler"] == "maccor"
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

    # Check that data was read (file has many lines)
    assert len(df) > 1000

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

    # Test with custom timezone
    measurement_tz = {"name": "00"}
    measurement_details_tz = iwdata.read.measurement_details(
        p,
        measurement_tz,
        "maccor",
        options={
            "timezone": "US/Pacific",
            "cell_metadata": {"Nominal cell capacity [A.h]": 1.0},
            "file_encoding": "ISO-8859-1",
        },
    )
    assert measurement_details_tz["measurement"]["start_time"] is not None
    # The start_time should be in ISO format
    assert "T" in measurement_details_tz["measurement"]["start_time"]


def test_maccor_123_extension(tmp_path):
    """Test reading Maccor file with .123 extension"""
    test_data = pl.DataFrame(
        {
            "Test (Sec)": [0.0, 1.0],
            "Current (A)": [2.0, 3.0],
            "Voltage (V)": [4.0, 3.0],
            "Cycle": [0, 0],
            "Step": [0, 1],
            "LogTemp001": [20.0, 20.0],
            "Status": ["D", "C"],
        }
    )
    test_data.write_csv(tmp_path / "test.123", separator="\t")
    expected_data_pl = pl.DataFrame(
        {
            "Time [s]": [0.0, 1.0],
            "Current [A]": [2.0, -3.0],
            "Voltage [V]": [4.0, 3.0],
            "Step from cycler": [0, 1],
            "Cycle from cycler": [0, 0],
            "Temperature [degC]": [20.0, 20.0],
            "Step count": [0, 1],
        }
    )
    expected_data_pl = iwdata.transform.set_cycle_count(expected_data_pl)
    expected_data_pl = iwdata.transform.set_capacity(expected_data_pl, options=None)
    expected_data_pl = iwdata.transform.set_energy(expected_data_pl, options=None)
    df_read = iwdata.read.time_series(tmp_path / "test.123", "maccor")
    pd.testing.assert_frame_equal(
        expected_data_pl.to_pandas().sort_index(axis=1),
        df_read.to_pandas().sort_index(axis=1),
    )


def test_detect_maccor_123_extension(tmp_path):
    """Test detecting Maccor file with .123 extension"""
    txt_file = tmp_path / "maccor.123"
    with open(txt_file, "w") as f:
        f.write("Date of Test:\t04/30/2024\n")
        f.write("Step\tTest (Sec)\tVoltage\n")
        f.write("1\t0.0\t3.0\n")

    detected = iwdata.read.detect_reader(txt_file)
    assert detected == "maccor"


def test_detect_maccor_csv(tmp_path):
    """Test detecting Maccor CSV file"""
    csv_file = tmp_path / "maccor.csv"
    with open(csv_file, "w") as f:
        f.write("Date of Test: 04/30/2024\n")
        f.write("Step,Prog Time,Current,Voltage\n")
        f.write("units row\n")
        f.write("1,0.0,2.0,4.0\n")

    detected = iwdata.read.detect_reader(csv_file)
    assert detected == "maccor"


def test_detect_maccor_xlsx(tmp_path):
    """Test detecting Maccor Excel file"""
    excel_file = tmp_path / "maccor.xlsx"
    # Create Excel file with Maccor signature (header in first row)
    data = pl.DataFrame(
        {
            "Step": [0, 1, 2],
            "Test (Sec)": [0.0, 1.0, 2.0],
            "Voltage (V)": [4.0, 3.0, 3.5],
        }
    )
    data.to_pandas().to_excel(excel_file, index=False)

    detected = iwdata.read.detect_reader(excel_file)
    assert detected == "maccor"


def test_detect_maccor_xlsx_with_header_in_first_row(tmp_path):
    """Test detecting Maccor Excel file where header row is in first row"""
    excel_file = tmp_path / "maccor_header_row.xlsx"
    # Create Excel file where header is in first row (as per new assumption)
    data = pl.DataFrame(
        {
            "Step": [0, 1, 2],
            "Test Time (sec)": [0.0, 1.0, 2.0],
            "Current (A)": [2.0, 3.0, 4.0],
            "Voltage (V)": [4.0, 3.0, 3.5],
            "Cycle": [0, 0, 0],
            "Status": ["D", "D", "C"],
        }
    )
    data.to_pandas().to_excel(excel_file, index=False)

    detected = iwdata.read.detect_reader(excel_file)
    assert detected == "maccor"


def test_detect_maccor_xlsx_column_names_only(tmp_path):
    """Test detecting Maccor Excel file based on column names only"""
    excel_file = tmp_path / "maccor_columns.xlsx"
    # Create Excel file with Maccor column names in first row
    data = pl.DataFrame(
        {
            "Step": [0, 1, 2],
            "Test (Sec)": [0.0, 1.0, 2.0],
            "Current (A)": [2.0, 3.0, 4.0],
            "Voltage (V)": [4.0, 3.0, 3.5],
            "Cycle": [0, 0, 0],
            "Status": ["D", "D", "C"],
        }
    )
    data.to_pandas().to_excel(excel_file, index=False)

    detected = iwdata.read.detect_reader(excel_file)
    assert detected == "maccor"


def test_detect_maccor_xlsx_prog_time(tmp_path):
    """Test detecting Maccor Excel file with Prog Time column"""
    excel_file = tmp_path / "maccor_prog_time.xlsx"
    # Create Excel file with Prog Time instead of Test (Sec)
    data = pl.DataFrame(
        {
            "Step": [0, 1],
            "Prog Time": [0.0, 1.0],
            "Current": [2.0, 3.0],
            "Voltage": [4.0, 3.0],
        }
    )
    data.to_pandas().to_excel(excel_file, index=False)

    detected = iwdata.read.detect_reader(excel_file)
    assert detected == "maccor"


def test_maccor_test_time_timestamp(tmp_path):
    """Test Maccor file with Test Time column containing timestamps"""
    # Create test data with Test Time as datetime strings
    test_data = pl.DataFrame(
        {
            "Step": [0, 0, 1, 1],
            "Test Time": [
                "04/30/2024 11:25:26",
                "04/30/2024 11:25:31",
                "04/30/2024 11:25:36",
                "04/30/2024 11:25:41",
            ],
            "Current (A)": [0.0, 2.0, 3.0, 1.0],
            "Voltage (V)": [4.0, 3.9, 3.8, 3.7],
            "Cycle": [0, 0, 0, 0],
            "LogTemp001": [20.0, 20.0, 20.0, 20.0],
            "Status": ["R", "D", "D", "C"],
        }
    )
    test_data.write_csv(tmp_path / "test.txt", separator="\t")

    # Expected data - Time should be computed from timestamps
    expected_data_pl = pl.DataFrame(
        {
            "Time [s]": [0.0, 5.0, 10.0, 15.0],
            "Current [A]": [0.0, 2.0, 3.0, -1.0],
            "Voltage [V]": [4.0, 3.9, 3.8, 3.7],
            "Step from cycler": [0, 0, 1, 1],
            "Cycle from cycler": [0, 0, 0, 0],
            "Temperature [degC]": [20.0, 20.0, 20.0, 20.0],
            "Step count": [0, 0, 1, 1],
        }
    )
    expected_data_pl = iwdata.transform.set_cycle_count(expected_data_pl)
    expected_data_pl = iwdata.transform.set_capacity(expected_data_pl, options=None)
    expected_data_pl = iwdata.transform.set_energy(expected_data_pl, options=None)

    df_read = iwdata.read.time_series(tmp_path / "test.txt", "maccor")

    # Check that Time [s] was computed correctly from timestamps
    assert "Time [s]" in df_read.columns
    pd.testing.assert_frame_equal(
        expected_data_pl.to_pandas().sort_index(axis=1),
        df_read.to_pandas().sort_index(axis=1),
    )


def test_maccor_dpt_timestamp(tmp_path):
    """Test Maccor file with DPT column containing timestamps"""
    # Create test data with DPT column (datetime point)
    test_data = pl.DataFrame(
        {
            "Step": [0, 0, 1, 1],
            "Test Time (sec)": [0.0, 5.0, 10.0, 15.0],  # Numeric time also present
            "DPT": [
                "04/30/2024 11:25:26",
                "04/30/2024 11:25:31",
                "04/30/2024 11:25:36",
                "04/30/2024 11:25:41",
            ],
            "Current (A)": [0.0, 2.0, 3.0, 1.0],
            "Voltage (V)": [4.0, 3.9, 3.8, 3.7],
            "Cycle": [0, 0, 0, 0],
            "Status": ["R", "D", "D", "C"],
        }
    )
    test_data.write_csv(tmp_path / "test.txt", separator="\t")

    # Expected data - should use numeric Time [s] since it's available
    expected_data_pl = pl.DataFrame(
        {
            "Time [s]": [0.0, 5.0, 10.0, 15.0],
            "Current [A]": [0.0, 2.0, 3.0, -1.0],
            "Voltage [V]": [4.0, 3.9, 3.8, 3.7],
            "Step from cycler": [0, 0, 1, 1],
            "Cycle from cycler": [0, 0, 0, 0],
            "Step count": [0, 0, 1, 1],
        }
    )
    expected_data_pl = iwdata.transform.set_cycle_count(expected_data_pl)
    expected_data_pl = iwdata.transform.set_capacity(expected_data_pl, options=None)
    expected_data_pl = iwdata.transform.set_energy(expected_data_pl, options=None)

    df_read = iwdata.read.time_series(tmp_path / "test.txt", "maccor")

    # Check that Time [s] exists and Timestamp was removed
    assert "Time [s]" in df_read.columns
    assert "Timestamp" not in df_read.columns
    pd.testing.assert_frame_equal(
        expected_data_pl.to_pandas().sort_index(axis=1),
        df_read.to_pandas().sort_index(axis=1),
    )


def test_maccor_xlsx_excel_time_format():
    """Test reading real Maccor Excel file with Excel time duration format"""
    from pathlib import Path

    # Read the actual test file
    df = iwdata.read.time_series(
        Path("tests/test_data/maccor.xlsx"),
        "maccor",
    )

    # Basic validation
    assert "Time [s]" in df.columns
    assert "Voltage [V]" in df.columns
    assert "Current [A]" in df.columns
    assert "Step from cycler" in df.columns
    assert "Discharge capacity [A.h]" in df.columns
    assert "Charge capacity [A.h]" in df.columns
    assert "Discharge energy [W.h]" in df.columns
    assert "Charge energy [W.h]" in df.columns
    # Single capacity/energy columns should be dropped
    assert "Capacity [A.h]" not in df.columns
    assert "Energy [W.h]" not in df.columns

    # Check that we have data
    assert df.shape[0] > 100  # Should have at least 100 rows

    # Check that Time [s] starts at 0 and increases
    time_values = df["Time [s]"].to_list()
    assert time_values[0] == 0.0
    assert time_values[1] == 1.0
    assert time_values[2] == 2.0

    # Check that we have capacity and energy columns from Excel file
    assert "Discharge capacity [A.h]" in df.columns
    assert "Charge capacity [A.h]" in df.columns
    assert "Discharge energy [W.h]" in df.columns
    assert "Charge energy [W.h]" in df.columns


def test_maccor_time_strictly_increasing(tmp_path):
    """Test that time is validated to be strictly increasing"""
    # Create test data with time that goes backwards
    test_data = pl.DataFrame(
        {
            "Step": [0, 0, 0, 0],
            "Test (Sec)": [0.0, 1.0, 0.5, 2.0],  # Time goes backwards at index 2
            "Current (A)": [0.0, 1.0, 2.0, 3.0],
            "Voltage (V)": [4.0, 3.9, 3.8, 3.7],
        }
    )
    test_data.write_csv(tmp_path / "test.txt", separator="\t")

    # Should raise ValueError about time not being strictly increasing (default behavior)
    with pytest.raises(
        ValueError,
        match=r"Time \[s\] must be strictly increasing",
    ):
        iwdata.read.time_series(tmp_path / "test.txt", "maccor")


def test_maccor_time_offset_fix(tmp_path):
    """Test that time_offset_fix option can automatically fix non-increasing time"""
    # Create test data with time that goes backwards
    test_data = pl.DataFrame(
        {
            "Step": [0, 0, 0, 0],
            "Test (Sec)": [0.0, 1.0, 0.5, 2.0],  # Time goes backwards at index 2
            "Current (A)": [0.0, 1.0, 2.0, 3.0],
            "Voltage (V)": [4.0, 3.9, 3.8, 3.7],
        }
    )
    test_data.write_csv(tmp_path / "test.txt", separator="\t")

    # Read with time_offset_fix=1.0 to automatically fix
    df = iwdata.read.time_series(
        tmp_path / "test.txt",
        "maccor",
        options={"time_offset_fix": 1.0},
    )

    # Check that time was fixed
    time_values = df["Time [s]"].to_list()
    assert len(time_values) == 4

    # Algorithm: diff = [1.0, -0.5, 1.5], fixed_diff = max(diff, 1.0) = [1.0, 1.0, 1.5]
    # Result: [0.0, 0.0+1.0, 0.0+1.0+1.0, 0.0+1.0+1.0+1.5] = [0.0, 1.0, 2.0, 3.5]
    assert time_values[0] == 0.0
    assert time_values[1] == 1.0
    assert time_values[2] == 2.0
    assert time_values[3] == 3.5

    # Verify all values are strictly increasing
    for i in range(1, len(time_values)):
        assert time_values[i] > time_values[i - 1]
