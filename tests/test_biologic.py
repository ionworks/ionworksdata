# pyright: reportMissingTypeStubs=false
from datetime import datetime, timezone

import pandas as pd  # type: ignore[reportMissingTypeStubs]
import polars as pl
import pytest

import ionworksdata as iwdata


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
    df_read = iwdata.read.time_series(tmp_path / "test.mpt", "biologic mpt").to_pandas()
    pd.testing.assert_frame_equal(
        expected_data_pl.to_pandas().sort_index(axis=1),
        df_read.sort_index(axis=1),
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
    df_read = iwdata.read.time_series(tmp_path / "test.txt", "biologic").to_pandas()
    pd.testing.assert_frame_equal(
        expected_data_pl.to_pandas().sort_index(axis=1),
        df_read.sort_index(axis=1),
    )


def test_biologic_start_time(tmp_path):
    # Biologic MPT files have a start time from the Date column
    test_data = pd.DataFrame(
        {
            "mode": [0],
            "Date": ["2021-01-01 00:00:00"],
        }
    )
    test_data.to_csv(tmp_path / "test.mpt", sep="\t")

    start_time = iwdata.read.start_time(tmp_path / "test.mpt", "biologic mpt")
    assert start_time == datetime(2021, 1, 1, tzinfo=timezone.utc)


def test_get_biologic_reader_object():
    reader_object = iwdata.read.BaseReader.get_reader_object("Biologic MPT")
    assert isinstance(reader_object, iwdata.read.BiologicMPT)


@pytest.mark.parametrize("has_nb_header_lines", [True, False])
def test_biologic_skiprows(tmp_path, has_nb_header_lines):
    """Test Biologic file reading with and without 'Nb header lines' specification"""
    if has_nb_header_lines:
        # Create a mock Biologic ASCII file with 'Nb header lines' specification
        test_file_content = """BT-Lab ASCII FILE
Nb header lines : 9

Modulo Bat

Run on channel :
User :

mode,time/s,I/mA,Ecell/V,Ns,Cycle number
0,0.0,2000.0,4.0,0,0
0,1.0,3000.0,3.0,1,0
"""
        test_file = tmp_path / "test_biologic_with_nb.txt"
    else:
        # Create a mock Biologic file without 'Nb header lines' specification
        test_file_content = """BT-Lab ASCII FILE

Modulo Bat

Run on channel :
User :

mode,time/s,I/mA,Ecell/V,Ns,Cycle number
0,0.0,2000.0,4.0,0,0
0,1.0,3000.0,3.0,1,0
"""
        test_file = tmp_path / "test_biologic_without_nb.txt"

    # Write the test file
    with open(test_file, "w", encoding="utf-8") as f:
        f.write(test_file_content)

    # Test that the file can be read correctly with skiprows
    df_read = iwdata.read.time_series(test_file, "biologic").to_pandas()

    # Verify that the data was read correctly (should have 2 rows of data)
    assert len(df_read) == 2
    assert "Time [s]" in df_read.columns
    assert "Current [A]" in df_read.columns
    assert "Voltage [V]" in df_read.columns

    # Verify the time values are correctly parsed
    expected_time_1 = 0.0
    expected_time_2 = 1.0
    assert df_read["Time [s]"].iloc[0] == expected_time_1
    assert df_read["Time [s]"].iloc[1] == expected_time_2

    # Verify current values (converted from mA to A)
    expected_current_1 = 2.0
    expected_current_2 = 3.0
    assert df_read["Current [A]"].iloc[0] == expected_current_1
    assert df_read["Current [A]"].iloc[1] == expected_current_2

    # Verify voltage values
    expected_voltage_1 = 4.0
    expected_voltage_2 = 3.0
    assert df_read["Voltage [V]"].iloc[0] == expected_voltage_1
    assert df_read["Voltage [V]"].iloc[1] == expected_voltage_2


def test_biologic_eis_file(tmp_path):
    """Test reading a Biologic EIS file with freq/Hz header format"""
    # Create a mock Biologic EIS file with freq/Hz header
    headers = (
        "freq/Hz\tRe(Z)/Ohm\t-Im(Z)/Ohm\t|Z|/Ohm\tPhase(Z)/deg\t"
        "time/s\t<Ewe>/V\t<I>/mA\tCs/µF\tCp/µF\tcycle number\t"
        "|Ewe|/V\t|I|/A\tI Range\t(Q-Qo)/mA.h\tRe(Y)/Ohm-1\t"
        "Im(Y)/Ohm-1\t|Y|/Ohm-1\tPhase(Y)/deg\tdq/mA.h"
    )
    # Dummy data rows
    data_row1 = (
        "1.0000E+005\t4.1476E+000\t6.3813E-001\t4.1964E+000\t-8.7467E+000\t"
        "0.0\t1.1306E+000\t5.6441E+000\t2.4940E+000\t5.7672E-002\t1\t"
        "9.8753E-003\t2.3533E-003\t10\t-5.0445E-006\t2.3553E-001\t"
        "3.6238E-002\t2.3830E-001\t8.7467E+000\t-5.0445E-006"
    )
    data_row2 = (
        "5.0000E+004\t4.2000E+000\t7.0000E-001\t4.2500E+000\t-9.0000E+000\t"
        "1.0\t1.1400E+000\t6.0000E+000\t2.5000E+000\t6.0000E-002\t1\t"
        "1.0000E-002\t2.4000E-003\t10\t-5.0000E-006\t2.4000E-001\t"
        "4.0000E-002\t2.4500E-001\t9.0000E+000\t-5.0000E-006"
    )

    test_file_content = f"{headers}\n{data_row1}\n{data_row2}\n"
    test_file = tmp_path / "test_eis.txt"

    with open(test_file, "w", encoding="utf-8") as f:
        f.write(test_file_content)

    # Read the file
    df_read = iwdata.read.time_series(test_file, "biologic")

    # Verify basic structure
    assert len(df_read) == 2
    assert "Time [s]" in df_read.columns
    assert "Voltage [V]" in df_read.columns
    assert "Current [A]" in df_read.columns
    assert "Cycle from cycler" in df_read.columns

    # Verify time values (starts at 0, second row at 1.0)
    assert df_read["Time [s]"][0] == 0.0
    assert df_read["Time [s]"][1] == 1.0
    # Verify voltage from <Ewe>/V column
    assert abs(df_read["Voltage [V]"][0] - 1.1306) < 0.001
    # Current converted from mA to A (5.6441 mA -> 5.6441e-3 A)
    assert abs(abs(df_read["Current [A]"][0]) - 5.6441e-3) < 1e-6
    assert df_read["Cycle from cycler"][0] == 1
