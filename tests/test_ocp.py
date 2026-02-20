"""Tests for the OCP (open-circuit potential) data path."""

import numpy as np
import pandas as pd
import pytest

import ionworksdata as iwdata


# ---------------------------------------------------------------------------
# measurement_details with data_type="ocp"
# ---------------------------------------------------------------------------


class TestMeasurementDetailsOCP:
    """Tests for ionworksdata.read.measurement_details with data_type='ocp'."""

    @pytest.fixture
    def ocp_csv(self, tmp_path):
        """Write a minimal OCP CSV and return the path."""
        df = pd.DataFrame(
            {
                "Voltage [V]": np.linspace(4.2, 3.0, 100),
                "Capacity [A.h]": np.linspace(0.0, 1.0, 100),
            }
        )
        path = tmp_path / "ocp_data.csv"
        df.to_csv(path, index=False)
        return path

    def test_returns_expected_keys(self, ocp_csv):
        result = iwdata.read.measurement_details(
            ocp_csv, {"name": "test"}, data_type="ocp"
        )
        assert "time_series" in result
        assert "measurement" in result
        assert "steps" not in result

    def test_time_series_columns(self, ocp_csv):
        result = iwdata.read.measurement_details(
            ocp_csv, {"name": "test"}, data_type="ocp"
        )
        ts = result["time_series"]
        assert "Voltage [V]" in ts.columns
        assert "Capacity [A.h]" in ts.columns
        assert "Step count" in ts.columns
        assert "Cycle count" in ts.columns
        # Should NOT have cycler-specific columns
        assert "Time [s]" not in ts.columns
        assert "Current [A]" not in ts.columns

    def test_time_series_shape(self, ocp_csv):
        result = iwdata.read.measurement_details(
            ocp_csv, {"name": "test"}, data_type="ocp"
        )
        assert result["time_series"].shape == (100, 4)

    def test_no_steps_returned(self, ocp_csv):
        """OCP path does not produce steps — the cloud handles it."""
        result = iwdata.read.measurement_details(
            ocp_csv, {"name": "test"}, data_type="ocp"
        )
        assert "steps" not in result

    def test_measurement_metadata(self, ocp_csv):
        result = iwdata.read.measurement_details(
            ocp_csv, {"name": "test_ocp"}, data_type="ocp"
        )
        m = result["measurement"]
        assert m["name"] == "test_ocp"
        assert m["data_type"] == "ocp"
        assert m["step_labels_validated"] is False

    def test_extra_column_mappings(self, tmp_path):
        df = pd.DataFrame(
            {
                "OCV": np.linspace(4.2, 3.0, 50),
                "SOC": np.linspace(0.0, 1.0, 50),
            }
        )
        path = tmp_path / "custom.csv"
        df.to_csv(path, index=False)

        result = iwdata.read.measurement_details(
            path,
            {"name": "mapped"},
            data_type="ocp",
            extra_column_mappings={"OCV": "Voltage [V]", "SOC": "Capacity [A.h]"},
        )
        ts = result["time_series"]
        assert "Voltage [V]" in ts.columns
        assert "Capacity [A.h]" in ts.columns

    def test_extra_constant_columns(self, ocp_csv):
        result = iwdata.read.measurement_details(
            ocp_csv,
            {"name": "const"},
            data_type="ocp",
            extra_constant_columns={"Temperature [degC]": 25.0},
        )
        ts = result["time_series"]
        assert "Temperature [degC]" in ts.columns
        assert ts["Temperature [degC]"][0] == pytest.approx(25.0)

    def test_missing_voltage_raises(self, tmp_path):
        df = pd.DataFrame({"Capacity [A.h]": [0.0, 0.5, 1.0]})
        path = tmp_path / "no_voltage.csv"
        df.to_csv(path, index=False)

        with pytest.raises(ValueError, match="Voltage"):
            iwdata.read.measurement_details(path, {"name": "bad"}, data_type="ocp")

    def test_voltage_only(self, tmp_path):
        """Capacity [A.h] is optional — only Voltage [V] is required."""
        df = pd.DataFrame({"Voltage [V]": [4.2, 4.0, 3.8]})
        path = tmp_path / "voltage_only.csv"
        df.to_csv(path, index=False)

        result = iwdata.read.measurement_details(
            path, {"name": "vonly"}, data_type="ocp"
        )
        ts = result["time_series"]
        assert "Voltage [V]" in ts.columns
        assert "Capacity [A.h]" not in ts.columns

    def test_validation_passes(self, ocp_csv):
        """OCP path should pass validation without raising."""
        result = iwdata.read.measurement_details(
            ocp_csv, {"name": "valid"}, data_type="ocp"
        )
        assert result["time_series"].shape[0] > 0

    def test_validation_can_be_disabled(self, ocp_csv):
        result = iwdata.read.measurement_details(
            ocp_csv,
            {"name": "noval"},
            data_type="ocp",
            options={"validate": False},
        )
        assert result["time_series"].shape[0] > 0

    def test_standard_path_not_affected(self):
        """Calling without data_type should still use the standard path."""
        df = pd.DataFrame(
            {
                "Time [s]": [0.0, 1.0, 2.0, 3.0],
                "Current [A]": [1.0, 1.0, -1.0, -1.0],
                "Voltage [V]": [4.0, 3.0, 3.5, 4.0],
                "Step from cycler": [0, 0, 1, 1],
                "Cycle from cycler": [0, 0, 0, 0],
            }
        )
        result = iwdata.read.measurement_details(
            df,
            {"name": "cycler"},
            "csv",
            extra_column_mappings={
                "Step from cycler": "Step from cycler",
                "Cycle from cycler": "Cycle from cycler",
            },
            options={"cell_metadata": {"Nominal cell capacity [A.h]": 1.0}},
        )
        assert "Time [s]" in result["time_series"].columns
        assert "Current [A]" in result["time_series"].columns
