# pyright: reportMissingTypeStubs=false
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd  # type: ignore[reportMissingTypeStubs]
import polars as pl
import pytest

import ionworksdata as iwdata


def test_base_reader_not_implemented():
    with pytest.raises(NotImplementedError):
        iwdata.read.BaseReader().run(None)

    with pytest.raises(NotImplementedError):
        iwdata.read.BaseReader().read_start_time(None)


def test_could_not_identify_current_units():
    df = pl.DataFrame(
        {
            "Time [s]": [0.0, 1.0, 2.0],
        }
    )
    with pytest.raises(RuntimeError, match="Could not identify current units"):
        iwdata.read.BaseReader().standard_data_processing(df)


def test_numeric_string_conversion():
    """Test that numeric columns with string values are converted to numeric."""
    df = pl.DataFrame(
        {
            "Time [s]": [0.0, 1.0, 2.0],
            "Current [A]": [1.0, 2.0, 3.0],
            "Voltage [V]": ["4", "5", "6"],
        }
    )
    result = iwdata.read.BaseReader().standard_data_processing(df).to_pandas()
    assert pd.api.types.is_numeric_dtype(result["Voltage [V]"])
    assert result["Voltage [V]"].tolist() == [4.0, 5.0, 6.0]


def test_read_csv():
    df = pd.DataFrame(
        {
            "Voltage (V)": [1, 2, 3],
            "Current (mA)": [4, 5, 6],
            "Time (s)": [7, 8, 9],
            "Step number": [0, 0, 0],
        }
    )
    df_read = iwdata.read.time_series(
        df, "csv", extra_column_mappings={"Step number": "Step number"}
    )
    assert set(df_read.columns) == {
        "Time [s]",
        "Voltage [V]",
        "Current [A]",
        "Step count",
        "Cycle count",
        "Discharge capacity [A.h]",
        "Charge capacity [A.h]",
        "Discharge energy [W.h]",
        "Charge energy [W.h]",
    }
    assert df_read["Current [A]"].to_list() == [-0.004, -0.005, -0.006]


def test_extra_columns():
    df = pd.DataFrame(
        {
            "Voltage (V)": [1, 2, 3],
            "Current (mA)": [4, 5, 6],
            "Time (s)": [7, 8, 9],
            "ID": ["a", "b", "c"],
        }
    )
    df_read = iwdata.read.time_series(df, "csv")
    assert "ID" not in df_read.columns

    df_read = iwdata.read.time_series(df, "csv", extra_column_mappings={"ID": "ID"})
    assert df_read["ID"].to_list() == ["a", "b", "c"]


def test_duplicate_columns():
    d = pl.DataFrame({"A": [1, 1], "B": [2, 2]})
    new_names = {
        "A": "C",
        "B": "C",
    }
    with pytest.warns(UserWarning, match="Duplicate columns for"):
        iwdata.util.check_for_duplicates(new_names, d)


def test_time_series():
    df = pd.DataFrame(
        {
            "Time [s]": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            "Current [A]": [4.0, 4.0, -1.0, -1.0, 4.0, 4.0, -1.0, -1.0],
            "Voltage [V]": [4.0, 2.0, 3.0, 4.0, 3.0, 2.0, 3.0, 4.0],
            "Temperature [degC]": [20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0],
            "Step number": [0, 0, 1, 1, 0, 0, 1, 1],
            "Cycle number": [0, 0, 0, 0, 1, 1, 1, 1],
        }
    )
    df_expected = pl.DataFrame(
        {
            "Time [s]": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            "Current [A]": [4.0, 4.0, -1.0, -1.0, 4.0, 4.0, -1.0, -1.0],
            "Voltage [V]": [4.0, 2.0, 3.0, 4.0, 3.0, 2.0, 3.0, 4.0],
            "Temperature [degC]": [20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0],
            "Step from cycler": [0, 0, 1, 1, 0, 0, 1, 1],
            "Cycle from cycler": [0, 0, 0, 0, 1, 1, 1, 1],
        }
    )
    options = {"step column": "Step from cycler"}
    df_expected = iwdata.transform.set_step_count(df_expected, options=options)
    df_expected = iwdata.transform.set_cycle_count(df_expected)
    df_expected = iwdata.transform.set_cycle_count(df_expected)
    df_expected = iwdata.transform.set_capacity(df_expected, options=None)
    df_expected = iwdata.transform.set_energy(df_expected, options=None)
    df_read = iwdata.read.time_series(
        df,
        "csv",
        extra_column_mappings={
            "Step number": "Step from cycler",
            "Cycle number": "Cycle from cycler",
        },
    )
    pd.testing.assert_frame_equal(
        df_expected.to_pandas().sort_index(axis=1),
        df_read.to_pandas().sort_index(axis=1),
    )


def test_time_series_and_steps(tmp_path):
    df = pd.DataFrame(
        {
            "Time [s]": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            "Current [A]": [4.0, 4.0, -1.0, -1.0, 4.0, 4.0, -1.0, -1.0],
            "Voltage [V]": [4.0, 2.0, 3.0, 4.0, 3.0, 2.0, 3.0, 4.0],
            "Temperature [degC]": [20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0],
            "Step number": [0, 0, 1, 1, 0, 0, 1, 1],
            "Cycle number": [0, 0, 0, 0, 1, 1, 1, 1],
            "Extra column": [0, 0, 0, 0, 0, 0, 0, 0],
        }
    )
    # We expect step and cycle number as they are passed as extra_column_mappings
    df_expected = pl.DataFrame(
        {
            "Time [s]": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            "Current [A]": [4.0, 4.0, -1.0, -1.0, 4.0, 4.0, -1.0, -1.0],
            "Voltage [V]": [4.0, 2.0, 3.0, 4.0, 3.0, 2.0, 3.0, 4.0],
            "Temperature [degC]": [20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0],
            "Step from cycler": [0, 0, 1, 1, 0, 0, 1, 1],
            "Cycle from cycler": [0, 0, 0, 0, 1, 1, 1, 1],
        }
    )
    options = {"step column": "Step from cycler"}
    df_expected = iwdata.transform.set_step_count(df_expected, options=options)
    df_expected = iwdata.transform.set_cycle_count(df_expected)
    df_expected = iwdata.transform.set_capacity(df_expected, options=None)
    df_expected = iwdata.transform.set_energy(df_expected, options=None)
    df_read, steps_read = iwdata.read.time_series_and_steps(
        df,
        "csv",
        extra_column_mappings={
            "Step number": "Step from cycler",
            "Cycle number": "Cycle from cycler",
        },
        save_dir=tmp_path,
    )
    pd.testing.assert_frame_equal(
        df_read.to_pandas().sort_index(axis=1),
        df_expected.to_pandas().sort_index(axis=1),
    )
    assert steps_read["Start index"].to_list() == [0, 2, 4, 6]
    assert steps_read["End index"].to_list() == [1, 3, 5, 7]
    assert steps_read["Step type"].to_list() == [
        "Constant current discharge",
        "Constant current charge",
        "Constant current discharge",
        "Constant current charge",
    ]
    assert steps_read["Step from cycler"].to_list() == [0, 1, 0, 1]
    assert steps_read["Cycle from cycler"].to_list() == [0, 0, 1, 1]

    assert steps_read["Step count"].to_list() == [0, 1, 2, 3]
    assert steps_read["Cycle count"].to_list() == [0, 0, 1, 1]

    assert (tmp_path / "time_series.csv").exists()
    assert (tmp_path / "steps.csv").exists()


def test_keep_required_columns():
    df = pl.DataFrame(
        {
            "Time [s]": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            "Extra column": [0, 0, 0, 0, 0, 0, 0, 0],
            "Column to drop": [0, 0, 0, 0, 0, 0, 0, 0],
        }
    )
    df_expected = df.select(["Time [s]", "Extra column"]).to_pandas()
    df_read = iwdata.read.keep_required_columns(df, ["Extra column"]).to_pandas()
    pd.testing.assert_frame_equal(df_expected, df_read)


@pytest.mark.parametrize(
    "reader_name,expected_class",
    [
        ("csv", iwdata.read.CSV),
        ("maccor", iwdata.read.Maccor),
        ("neware", iwdata.read.Neware),
        ("repower", iwdata.read.Repower),
    ],
)
def test_get_reader_object(reader_name, expected_class):
    reader_object = iwdata.read.BaseReader.get_reader_object(reader_name)
    assert isinstance(reader_object, expected_class)


def test_get_reader_object_error():
    with pytest.raises(ValueError):
        iwdata.read.BaseReader.get_reader_object("nonexistent")


def test_readers_return_polars():
    """Test that readers return Polars DataFrames."""
    import polars as pl

    # Test CSV reader
    csv_data = iwdata.read.time_series(
        Path("tests/test_data/constant_current_synthetic.csv"),
        "csv",
        extra_column_mappings={
            "time": "Time [s]",
            "voltage": "Voltage [V]",
            "current": "Current [A]",
        },
    )
    assert isinstance(csv_data, pl.DataFrame), (
        "CSV reader should return Polars DataFrame"
    )

    # Verify it has the expected columns
    assert "Time [s]" in csv_data.columns
    assert "Voltage [V]" in csv_data.columns
    assert "Current [A]" in csv_data.columns


def test_start_time(tmp_path):
    # CSV files do not have a start time
    start_time = iwdata.read.start_time(filename=None, reader="csv")
    assert start_time is None

    # Maccor files have a start time in the header (test auto-detection)
    with open(tmp_path / "test.txt", "w") as f:
        f.write("Today's Date:\t23 May 2025\tDate of Test:\t23 May 2025, 10:00:00 AM\n")
        f.write("Step\tTest (Sec)\n")  # Need Step and Test Time for detection

    start_time = iwdata.read.start_time(tmp_path / "test.txt")
    assert start_time == datetime(2025, 5, 23, 10, 0, 0, tzinfo=timezone.utc)

    # Novonix start time tested in tests/test_novonix.py

    # Neware files have a datetime column (test auto-detection)
    test_data = pd.DataFrame(
        {
            "DateTime": ["2021-01-01 00:00:00"],
        }
    )
    test_data.to_csv(tmp_path / "test.csv")

    start_time = iwdata.read.start_time(tmp_path / "test.csv")
    assert start_time == datetime(2021, 1, 1, tzinfo=timezone.utc)

    # Repower has a system time column (test auto-detection)
    # Need Repower signature columns for detection
    test_data = pd.DataFrame(
        {
            "Cycle ID": [1],
            "Step ID": [1],
            "Record ID": [1],
            "System Time": [" 2025-05-23 10:00:00"],
        }
    )
    test_data.to_csv(tmp_path / "test.csv")

    start_time = iwdata.read.start_time(
        tmp_path / "test.csv",
        extra_column_mappings=None,
        options={"cell_metadata": {"Nominal cell capacity [A.h]": 1.0}},
    )
    assert start_time == datetime(2025, 5, 23, 10, 0, 0, tzinfo=timezone.utc)


def test_mpt(tmp_path):
    test_data = pd.DataFrame(
        {
            "mode": [0, 0],
            "time/s": [0.0, 1.0],
            "I/mA": [2.0 * 1000, 3.0 * 1000],
            "Ecell/V": [4.0, 3.0],
            "Ns": [0, 1],
            "Cycle number": [0, 0],
        }
    )
    test_data.to_csv(tmp_path / "test.mpt", sep="\t")

    expected_data_pl = pl.DataFrame(
        {
            "Time [s]": [0.0, 1.0],
            "Current [A]": [2.0, 3.0],
            "Voltage [V]": [4.0, 3.0],
            "Step from cycler": [0, 1],
            "Cycle from cycler": [0, 0],
            "Step count": [0, 1],
        }
    )
    expected_data_pl = iwdata.transform.set_cycle_count(expected_data_pl)
    expected_data_pl = iwdata.transform.set_capacity(expected_data_pl, options=None)
    expected_data_pl = iwdata.transform.set_energy(expected_data_pl, options=None)
    df_read = iwdata.read.time_series(tmp_path / "test.mpt", "biologic mpt")
    pd.testing.assert_frame_equal(
        expected_data_pl.to_pandas().sort_index(axis=1),
        df_read.to_pandas().sort_index(axis=1),
    )


def test_biologic_txt(tmp_path):
    test_data = pd.DataFrame(
        {
            "mode": [0, 0],
            "time/s": [0.0, 1.0],
            "I/mA": [2.0 * 1000, 3.0 * 1000],
            "Ecell/V": [4.0, 3.0],
            "Ns": [0, 1],
            "Cycle number": [0, 0],
        }
    )
    test_data.to_csv(tmp_path / "test.txt", sep=",")
    expected_data_pl = pl.DataFrame(
        {
            "Time [s]": [0.0, 1.0],
            "Current [A]": [2.0, 3.0],
            "Voltage [V]": [4.0, 3.0],
            "Step from cycler": [0, 1],
            "Cycle from cycler": [0, 0],
            "Step count": [0, 1],
        }
    )
    expected_data_pl = iwdata.transform.set_cycle_count(expected_data_pl)
    expected_data_pl = iwdata.transform.set_capacity(expected_data_pl, options=None)
    expected_data_pl = iwdata.transform.set_energy(expected_data_pl, options=None)
    df_read = iwdata.read.time_series(tmp_path / "test.txt", "biologic")
    pd.testing.assert_frame_equal(
        expected_data_pl.to_pandas().sort_index(axis=1),
        df_read.to_pandas().sort_index(axis=1),
    )

    # Novonix reader parsing tested in tests/test_novonix.py


def test_neware(tmp_path):
    test_data = pd.DataFrame(
        {
            "DateTime": ["2021-01-01 00:00:00", "2021-01-01 00:00:01"],
            "Current (A)": [2.0, 3.0],
            "Voltage (V)": [4.0, 3.0],
            "Step ID": [0, 1],
            "Cycle ID": [0, 0],
            "Status": ["D", "C"],
        }
    )
    test_data.to_csv(tmp_path / "test.csv")
    expected_data_pl = pl.DataFrame(
        {
            "Time [s]": [0.0, 1.0],
            "Current [A]": [2.0, 3.0],
            "Voltage [V]": [4.0, 3.0],
            "Step from cycler": [0, 1],
            "Cycle from cycler": [0, 0],
            "Step count": [0, 1],
        }
    )
    expected_data_pl = iwdata.transform.set_cycle_count(expected_data_pl)
    expected_data_pl = iwdata.transform.set_capacity(expected_data_pl, options=None)
    expected_data_pl = iwdata.transform.set_energy(expected_data_pl, options=None)
    # Test auto-detection (Neware CSV has DateTime column)
    df_read = iwdata.read.time_series(tmp_path / "test.csv")
    pd.testing.assert_frame_equal(
        expected_data_pl.to_pandas().sort_index(axis=1),
        df_read.to_pandas().sort_index(axis=1),
    )


def test_time_series_wrong_argument_order(tmp_path):
    """Test that wrong argument order raises a clear error message."""
    test_file = tmp_path / "test.csv"
    test_data = pd.DataFrame(
        {
            "Time [s]": [0.0, 1.0],
            "Current [A]": [1.0, 2.0],
            "Voltage [V]": [3.0, 4.0],
        }
    )
    test_data.to_csv(test_file)

    # Test with wrong order: reader first, filename second
    with pytest.raises(
        ValueError,
        match="Arguments appear to be in the wrong order for time_series",
    ) as exc_info:
        iwdata.read.time_series("csv", str(test_file))

    error_msg = str(exc_info.value)
    assert "Expected: time_series(filename, reader" in error_msg
    assert "but got: time_series(reader='csv'" in error_msg
    assert f"Please use: time_series('{test_file}', 'csv'" in error_msg


def test_time_series_and_steps_wrong_argument_order(tmp_path):
    """Test that wrong argument order raises a clear error message."""
    test_file = tmp_path / "test.csv"
    test_data = pd.DataFrame(
        {
            "Time [s]": [0.0, 1.0],
            "Current [A]": [1.0, 2.0],
            "Voltage [V]": [3.0, 4.0],
        }
    )
    test_data.to_csv(test_file)

    # Test with wrong order: reader first, filename second
    with pytest.raises(
        ValueError,
        match="Arguments appear to be in the wrong order for time_series_and_steps",
    ) as exc_info:
        iwdata.read.time_series_and_steps("csv", str(test_file))

    error_msg = str(exc_info.value)
    assert "Expected: time_series_and_steps(filename, reader" in error_msg
    assert "but got: time_series_and_steps(reader='csv'" in error_msg
    assert f"Please use: time_series_and_steps('{test_file}', 'csv'" in error_msg


def test_start_time_wrong_argument_order(tmp_path):
    """Test that wrong argument order raises a clear error message."""
    test_file = tmp_path / "test.txt"
    with open(test_file, "w") as f:
        f.write("Today's Date:\t23 May 2025\tDate of Test:\t23 May 2025, 10:00:00 AM\n")
        f.write("Step\n")

    # Test with wrong order: reader first, filename second
    with pytest.raises(
        ValueError,
        match="Arguments appear to be in the wrong order for start_time",
    ) as exc_info:
        iwdata.read.start_time("maccor", str(test_file))

    error_msg = str(exc_info.value)
    assert "Expected: start_time(filename, reader" in error_msg
    assert "but got: start_time(reader='maccor'" in error_msg
    assert f"Please use: start_time('{test_file}', 'maccor'" in error_msg


def test_measurement_details_wrong_argument_order(tmp_path):
    """Test that wrong argument order raises a clear error message."""
    test_file = tmp_path / "test.csv"
    test_data = pd.DataFrame(
        {
            "Time [s]": [0.0, 1.0],
            "Current [A]": [1.0, 2.0],
            "Voltage [V]": [3.0, 4.0],
        }
    )
    test_data.to_csv(test_file)

    measurement = {"name": "test"}

    # Test with wrong order: reader first, filename second
    with pytest.raises(
        ValueError,
        match="Arguments appear to be in the wrong order for measurement_details",
    ) as exc_info:
        iwdata.read.measurement_details(measurement, "csv", str(test_file))

    error_msg = str(exc_info.value)
    assert (
        "Arguments appear to be in the wrong order for measurement_details" in error_msg
    )
    assert "Expected: measurement_details(filename, measurement, reader" in error_msg
    # The error message should indicate the arguments are swapped
    assert (
        "measurement='csv'" in error_msg
        or "reader='csv'" in error_msg
        or "file path" in error_msg
    )


def test_wrong_order_with_path_object(tmp_path):
    """Test that wrong order detection works with Path objects."""
    test_file = tmp_path / "test.csv"
    test_data = pd.DataFrame(
        {
            "Time [s]": [0.0, 1.0],
            "Current [A]": [1.0, 2.0],
            "Voltage [V]": [3.0, 4.0],
        }
    )
    test_data.to_csv(test_file)

    # Test with Path object in wrong position
    with pytest.raises(
        ValueError,
        match="Arguments appear to be in the wrong order",
    ):
        iwdata.read.time_series("csv", test_file)


def test_correct_order_does_not_raise_error(tmp_path):
    """Test that correct argument order does not raise an error."""
    test_file = tmp_path / "test.csv"
    test_data = pd.DataFrame(
        {
            "Time [s]": [0.0, 1.0],
            "Current [A]": [1.0, 2.0],
            "Voltage [V]": [3.0, 4.0],
        }
    )
    test_data.to_csv(test_file)

    # These should not raise errors
    iwdata.read.time_series(test_file, "csv")  # Explicit reader
    iwdata.read.time_series_and_steps(test_file, "csv")
    iwdata.read.start_time(test_file, "csv")
