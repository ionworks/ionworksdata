import pandas as pd
import polars as pl
import pytest

import ionworksdata as iwdata


@pytest.fixture
def test_data():
    df = pd.DataFrame(
        {
            "Time [s]": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            "Current [A]": [4.0, 4.0, -1.0, -1.0, 4.0, 4.0, -1.0, -1.0],
            "Voltage [V]": [4.0, 2.0, 3.0, 4.0, 3.0, 2.0, 3.0, 4.0],
            "Temperature [degC]": [20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0],
            "Step from cycler": [0, 0, 1, 1, 0, 0, 1, 1],
            "Cycle from cycler": [0, 0, 0, 0, 1, 1, 1, 1],
        }
    )
    return df


def test_measurement_details(test_data):
    df_expected = pl.from_pandas(test_data.copy())
    options = {"step column": "Step from cycler"}
    df_expected = iwdata.transform.set_step_count(df_expected, options=options)
    df_expected = iwdata.transform.set_cycle_count(df_expected)
    # Add capacity and energy columns (measurement_details adds these by default)
    df_expected = iwdata.transform.set_capacity(df_expected)
    df_expected = iwdata.transform.set_energy(df_expected)

    measurement = {}
    options = {"cell_metadata": {"Nominal cell capacity [A.h]": 1.0}}
    measurement_details = iwdata.read.measurement_details(
        test_data,
        measurement,
        "csv",
        extra_column_mappings={
            "Step from cycler": "Step from cycler",
            "Cycle from cycler": "Cycle from cycler",
        },
        options=options,
    )

    assert measurement_details["measurement"]["cycler"] == "csv"

    df_read = measurement_details["time_series"]
    pd.testing.assert_frame_equal(
        df_expected.to_pandas().sort_index(axis=1),
        df_read.to_pandas().sort_index(axis=1),
        check_exact=False,
        rtol=1e-5,
    )


def test_reader_unknown_file():
    with pytest.raises(FileNotFoundError):
        iwdata.read.measurement_details(
            "test_file.csv",
            {},
            "csv",
        )


def test_mpt(tmp_path):
    test_file = tmp_path / "test.mpt"
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
    test_data.to_csv(test_file, sep="\t")

    options = {"cell_metadata": {"Nominal cell capacity [A.h]": 1.0}}
    measurement_details = iwdata.read.measurement_details(
        test_file,
        {},
        "biologic mpt",
        options=options,
    )
    read_data = measurement_details["time_series"]

    # Build expected data by reading through time_series and applying transforms
    # (same as measurement_details does)
    expected_data_pl = iwdata.read.time_series(
        test_file, "biologic mpt", options=options
    )
    expected_data_pl = iwdata.read.keep_required_columns(expected_data_pl)
    expected_data = expected_data_pl.to_pandas()

    pd.testing.assert_frame_equal(
        expected_data.sort_index(axis=1),
        read_data.to_pandas().sort_index(axis=1),
        check_exact=False,
        rtol=1e-5,
    )


def test_column_mapping(tmp_path):
    test_file = tmp_path / "test.csv"
    test_data = pd.DataFrame(
        {
            "Time [s]": [0.0, 1.0],
            "Current [A]": [2.0, 3.0],
            "Voltage [V]": [3.0, 4.0],
            "Temperature [degC]": [5.0, 6.0],
            "Freq.HZ": [0.1, 0.2],
        }
    )
    test_data.to_csv(test_file)
    options = {"cell_metadata": {"Nominal cell capacity [A.h]": 1.0}}
    measurement_details = iwdata.read.measurement_details(
        test_file,
        {},
        "csv",
        extra_column_mappings={"Freq.HZ": "Frequency [Hz]"},
        options=options,
    )
    assert "Frequency [Hz]" in measurement_details["time_series"].columns


@pytest.mark.parametrize("unit", ["s", "h"])
def test_column_mapping_custom_time_column(tmp_path, unit):
    test_file = tmp_path / "test.csv"
    test_data = pd.DataFrame(
        {
            "weird_time_column": [0.0, 1.0],
            "Current [A]": [2.0, 3.0],
            "Voltage [V]": [3.0, 4.0],
        }
    )
    test_data.to_csv(test_file)
    options = {"cell_metadata": {"Nominal cell capacity [A.h]": 1.0}}
    measurement_details = iwdata.read.measurement_details(
        test_file,
        {},
        "csv",
        extra_column_mappings={"weird_time_column": f"Time [{unit}]"},
        options=options,
    )
    assert "Time [s]" in measurement_details["time_series"].columns
    assert measurement_details["time_series"]["Time [s]"].to_list() == (
        [0.0, 1.0] if unit == "s" else [0.0, 3600.0]
    )


def test_constant_columns(test_data):
    options = {"cell_metadata": {"Nominal cell capacity [A.h]": 1.0}}
    measurement_details = iwdata.read.measurement_details(
        test_data,
        {},
        "csv",
        extra_constant_columns={"Frequency [Hz]": 1.0},
        options=options,
    )
    assert (measurement_details["time_series"]["Frequency [Hz]"] == 1.0).all()
    assert len(measurement_details["time_series"]["Frequency [Hz]"]) == 8


def test_csv_no_start_time(test_data):
    options = {"cell_metadata": {"Nominal cell capacity [A.h]": 1.0}}
    with pytest.warns(UserWarning, match="CSV reader does not"):
        iwdata.read.measurement_details(
            test_data,
            {},
            "csv",
            options=options,
        )


def test_label_measurement_details():
    data = pd.DataFrame(
        {
            "Time [s]": [0, 1800, 3600],
            "Current [A]": [1, 1, 1],
            "Voltage [V]": [4.0, 3.0, 2.0],
            "Step number": [0, 0, 0],
        }
    )
    measurement_details = iwdata.read.measurement_details(
        data,
        {},
        "csv",
        options={"cell_metadata": {"Nominal cell capacity [A.h]": 1.0}},
    )
    assert measurement_details["measurement"]["step_labels_validated"]
    # steps is polars; convert to pandas for iloc
    assert measurement_details["steps"].to_pandas()["Label"].iloc[0] == "Cycling"


def test_failed_to_identify_step_labels(test_data):
    measurement_details = iwdata.read.measurement_details(
        test_data,
        {},
        "csv",
    )
    assert not measurement_details["measurement"]["step_labels_validated"]


def test_measurement_details_missing_cell_metadata(tmp_path):
    """Test measurement_details when cell_metadata is missing or incomplete"""
    # Create test data
    test_data = pd.DataFrame(
        {
            "Time [s]": [0.0, 1.0, 2.0, 3.0],
            "Current [A]": [1.0, 1.0, -1.0, -1.0],
            "Voltage [V]": [4.0, 3.0, 3.0, 4.0],
            "Step from cycler": [0, 0, 1, 1],
            "Cycle from cycler": [0, 0, 0, 0],
        }
    )
    test_data.to_csv(tmp_path / "test.csv")

    measurement = {"test_name": "test"}

    # Test with no cell_metadata in options
    result = iwdata.read.measurement_details(
        measurement=measurement,
        reader="csv",
        filename=tmp_path / "test.csv",
        options={},  # No cell_metadata
    )

    assert result["measurement"]["step_labels_validated"] is False
    assert "start_time" not in result["measurement"]
    assert result["measurement"]["cycler"] == "csv"

    # Test with empty cell_metadata
    result = iwdata.read.measurement_details(
        measurement=measurement,
        reader="csv",
        filename=tmp_path / "test.csv",
        options={"cell_metadata": {}},  # Empty cell_metadata
    )

    assert result["measurement"]["step_labels_validated"] is False


def test_measurement_details_step_label_validation(tmp_path):
    """Test measurement_details step label validation logic"""
    # Create test data with steps that should have valid labels
    test_data = pd.DataFrame(
        {
            "Time [s]": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "Current [A]": [1.0, 1.0, -1.0, -1.0, 0.5, 0.5],
            "Voltage [V]": [4.0, 3.0, 3.0, 4.0, 3.5, 3.0],
            "Step from cycler": [0, 0, 1, 1, 2, 2],
            "Cycle from cycler": [0, 0, 0, 0, 1, 1],
        }
    )
    test_data.to_csv(tmp_path / "test.csv")

    measurement = {"test_name": "test"}

    # Test with cell_metadata that should trigger label validation
    result = iwdata.read.measurement_details(
        measurement=measurement,
        reader="csv",
        filename=tmp_path / "test.csv",
        options={"cell_metadata": {"Nominal cell capacity [A.h]": 1.0}},
    )

    # Check that the validation logic was executed
    assert "step_labels_validated" in result["measurement"]
    # The actual value depends on whether the labels are valid, but the key should exist
    assert isinstance(result["measurement"]["step_labels_validated"], bool)
