# pyright: reportMissingTypeStubs=false
from pathlib import Path
from unittest.mock import patch

import pytest

import ionworksdata as iwdata


def test_detect_reader_novonix():
    """Test detecting Novonix reader from file"""
    p = Path("tests/test_data/novonix.csv")
    detected = iwdata.read.detect_reader(p)
    assert detected == "novonix"


def test_detect_reader_maccor_excel_2_row_header():
    p = Path("tests/test_data/maccor_2_row_header.xlsx")
    detected = iwdata.read.detect_reader(p)
    assert detected == "maccor"


def test_detect_reader_neware():
    """Test detecting Neware reader from Excel file"""
    p = Path("tests/test_data/neware.xlsx")
    detected = iwdata.read.detect_reader(p)
    assert detected == "neware"


def test_detect_reader_maccor():
    """Test detecting Maccor reader from file"""
    p = Path("tests/test_data/maccor.txt")
    detected = iwdata.read.detect_reader(p)
    assert detected == "maccor"


def test_detect_reader_repower():
    """Test detecting Repower reader from file"""
    p = Path("tests/test_data/repower.csv")
    detected = iwdata.read.detect_reader(p)
    assert detected == "repower"


def test_detect_reader_unknown(tmp_path):
    """Test that unknown file raises ValueError"""
    # Create a file that doesn't match any known pattern
    unknown_file = tmp_path / "unknown.txt"
    with open(unknown_file, "w") as f:
        f.write("Some random content\n")
        f.write("No recognizable pattern here\n")

    with pytest.raises(ValueError, match="Could not automatically detect"):
        iwdata.read.detect_reader(unknown_file)


def test_detect_reader_and_read():
    """Test that detected reader can be used to read files"""
    # Test with Novonix
    p = Path("tests/test_data/novonix.csv")
    detected = iwdata.read.detect_reader(p)
    assert detected == "novonix"

    # Use detected reader to read the file
    df = iwdata.read.time_series(p, detected)
    assert len(df) > 0
    assert "Time [s]" in df.columns

    # Test with Repower
    p = Path("tests/test_data/repower.csv")
    detected = iwdata.read.detect_reader(p)
    assert detected == "repower"

    # Use detected reader to read the file
    df = iwdata.read.time_series(
        p,
        detected,
        options={
            "cell_metadata": {
                "Lower voltage cut-off [V]": 2.5,
                "Upper voltage cut-off [V]": 4.2,
                "Nominal cell capacity [A.h]": 5.0,
            }
        },
    )
    assert len(df) > 0
    assert "Time [s]" in df.columns


def test_detect_reader_with_measurement_details():
    """Test using detect_reader with measurement_details"""
    # Test with Novonix
    p = Path("tests/test_data/novonix.csv")
    detected = iwdata.read.detect_reader(p)

    measurement = {"name": "00"}
    measurement_details = iwdata.read.measurement_details(
        p,
        measurement,
        detected,
        options={"cell_metadata": {"Nominal cell capacity [A.h]": 1.0}},
    )

    assert measurement_details["measurement"]["cycler"] == detected
    assert "time_series" in measurement_details
    assert len(measurement_details["time_series"]) > 0


def test_detect_reader_neware_csv(tmp_path):
    """Test detecting Neware reader from CSV file"""
    # Create a CSV file with Neware signature columns
    csv_file = tmp_path / "neware.csv"
    with open(csv_file, "w") as f:
        f.write("DateTime,Current (A),Voltage (V)\n")
        f.write("2021-01-01 00:00:00,2.0,4.0\n")

    detected = iwdata.read.detect_reader(csv_file)
    assert detected == "neware"


def test_detect_reader_neware_csv_alternative_format(tmp_path):
    """Test detecting Neware CSV with Date(h:min:s.ms) column"""
    csv_file = tmp_path / "neware.csv"
    with open(csv_file, "w") as f:
        f.write("Date(h:min:s.ms),Current (A),Voltage (V)\n")
        f.write("2021-01-01 00:00:00,2.0,4.0\n")

    detected = iwdata.read.detect_reader(csv_file)
    assert detected == "neware"


def test_detect_reader_maccor_txt_test_sec(tmp_path):
    """Test detecting Maccor reader from .txt file with Test (Sec)"""
    txt_file = tmp_path / "maccor.txt"
    with open(txt_file, "w") as f:
        # First line needs to have tab for detection to work
        f.write("Date of Test:\t04/30/2024\n")
        f.write("Step\tTest (Sec)\tVoltage\n")
        f.write("1\t0.0\t3.0\n")

    detected = iwdata.read.detect_reader(txt_file)
    assert detected == "maccor"


def test_detect_reader_unicode_decode_error(tmp_path):
    """Test handling UnicodeDecodeError with latin1 fallback"""
    # Create a file with latin1 encoding that would fail with utf-8
    latin1_file = tmp_path / "latin1.txt"
    # Write bytes that are valid latin1 but not utf-8
    with open(latin1_file, "wb") as f:
        f.write(b"Date of Test: 04/30/2024\n")
        f.write(b"Step\tTest Time\tVoltage\n")
        # Add some latin1-specific bytes (e.g., é in latin1)
        f.write(b"1\t0.0\t3.0\xe9\n")

    detected = iwdata.read.detect_reader(latin1_file)
    assert detected == "maccor"


def test_detect_reader_excel_not_neware(tmp_path):
    """Test that Excel file that's not Neware continues to other checks"""
    import pandas as pd

    # Create an Excel file that doesn't have Neware columns
    excel_file = tmp_path / "not_neware.xlsx"
    df = pd.DataFrame({"SomeColumn": [1, 2, 3], "OtherColumn": [4, 5, 6]})
    df.to_excel(excel_file, index=False)

    # Should raise ValueError since it's not recognized
    with pytest.raises(ValueError, match="Could not automatically detect"):
        iwdata.read.detect_reader(excel_file)


def test_detect_reader_excel_empty_sheet(tmp_path):
    """Test handling Excel file with empty sheet"""
    import pandas as pd

    # Create an Excel file with an empty sheet
    excel_file = tmp_path / "empty.xlsx"
    df = pd.DataFrame()
    df.to_excel(excel_file, index=False)

    # Should return False for _is_neware_excel and then fail detection
    with pytest.raises(ValueError, match="Could not automatically detect"):
        iwdata.read.detect_reader(excel_file)


def test_detect_reader_excel_invalid_file(tmp_path):
    """Test handling invalid Excel file"""
    # Create a file that looks like Excel but isn't valid
    invalid_file = tmp_path / "invalid.xlsx"
    with open(invalid_file, "w") as f:
        f.write("This is not a valid Excel file\n")

    # Should return False for _is_neware_excel and then fail detection
    with pytest.raises(ValueError, match="Could not automatically detect"):
        iwdata.read.detect_reader(invalid_file)


def test_detect_reader_empty_file(tmp_path):
    """Test handling empty file"""
    empty_file = tmp_path / "empty.txt"
    empty_file.touch()

    with pytest.raises(ValueError, match="Could not automatically detect"):
        iwdata.read.detect_reader(empty_file)


def test_detect_reader_excel_empty_dataframe(tmp_path):
    """Test handling Excel file with empty dataframe"""
    import polars as pl

    # Create a valid Excel file
    import pandas as pd

    excel_file = tmp_path / "test.xlsx"
    df = pd.DataFrame({"DateTime": ["2021-01-01"], "Current (A)": [1.0]})
    df.to_excel(excel_file, index=False)

    # Mock polars.read_excel to return empty dataframe for one sheet
    with patch("ionworksdata.read.detect.pl.read_excel") as mock_read:
        # First call returns empty dataframe (simulating empty sheet)
        mock_read.return_value = pl.DataFrame()
        # This should skip the empty sheet and continue
        with pytest.raises(ValueError, match="Could not automatically detect"):
            iwdata.read.detect_reader(excel_file)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
