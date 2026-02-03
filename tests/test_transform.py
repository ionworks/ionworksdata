import polars as pl
import pytest

import ionworksdata as iwdata


@pytest.mark.parametrize("method", ["status", "current sign", "step column"])
def test_set_cumulative_step_number(method):
    # Create a sample DataFrame for testing
    data = pl.DataFrame(
        {
            "Status": ["A", "A", "B", "B", "C", "C"],
            "Current [A]": [1.0, 2.0, -1.0, -2.0, 0.5, 0.5],
            "Step number": [0, 0, 1, 1, 0, 0],
        }
    )

    output = iwdata.transform.set_cumulative_step_number(
        data, options={"method": method}
    )
    step_number = output["Step number"].to_list()

    # All options should add or overwrite the step number column
    expected_step_number = [0, 0, 1, 1, 2, 2]
    assert step_number == expected_step_number


def test_set_cumulative_step_number_eis():
    data = pl.DataFrame(
        {
            "Frequency [Hz]": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            "Current [A]": [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 5.0, 5.0, -3.0],
        }
    )
    # expect step number to increase by 1 each time the current sign changes
    step_number = iwdata.transform.get_cumulative_step_number(
        data, options={"method": "current sign", "group EIS steps": False}
    )
    assert step_number.to_list() == [0, 1, 2, 3, 4, 5, 5, 5, 6]

    # expect step number to be constant during EIS steps
    step_number = iwdata.transform.get_cumulative_step_number(
        data, options={"method": "current sign", "group EIS steps": True}
    )
    assert step_number.to_list() == [0, 0, 0, 0, 0, 0, 1, 1, 2]


def test_set_step_count():
    data = pl.DataFrame(
        {
            "Step from cycler": [0, 0, 1, 1, 2, 2, 0, 0],
        }
    )
    output = iwdata.transform.set_step_count(data)
    assert output["Step count"].to_list() == [0, 0, 1, 1, 2, 2, 3, 3]


@pytest.mark.parametrize("cycle_column", ["Cycle number", "Cycle"])
def test_set_cumulative_cycle_number(cycle_column):
    data = pl.DataFrame(
        {
            cycle_column: [0, 0, 1, 1, 0, 0],
        }
    )
    output = iwdata.transform.set_cumulative_cycle_number(
        data, options={"cycle column": cycle_column}
    )
    expected_cycle_number = [0, 0, 1, 1, 2, 2]
    assert output["Cycle number"].to_list() == expected_cycle_number


def test_set_cycle_count():
    data = pl.DataFrame(
        {
            "Cycle from cycler": [0, 0, 1, 1, 0, 0],
        }
    )
    output = iwdata.transform.set_cycle_count(data)
    assert output["Cycle count"].to_list() == [0, 0, 1, 1, 2, 2]


def test_reset_time():
    data = pl.DataFrame({"Time [s]": [1.0, 2.0, 3.0]})
    out = iwdata.transform.reset_time(data)
    assert out["Time [s]"].to_list() == [0.0, 1.0, 2.0]


def test_set_capacity():
    data = pl.DataFrame(
        {"Time [s]": [0.0, 1.0, 2.0, 3.0], "Current [A]": [1.0, 1.0, 1.0, 1.0]}
    )
    out = iwdata.transform.set_capacity(data)
    # Positive current = discharge
    assert out["Discharge capacity [A.h]"].to_list() == [
        0.0,
        1.0 / 3600,
        2.0 / 3600,
        3.0 / 3600,
    ]
    # No charge with positive current
    assert out["Charge capacity [A.h]"].to_list() == [0.0, 0.0, 0.0, 0.0]


def test_set_energy():
    data = pl.DataFrame(
        {
            "Time [s]": [0.0, 1.0, 2.0, 3.0],
            "Voltage [V]": [4.0, 4.0, 4.0, 4.0],
            "Current [A]": [1.0, 1.0, 1.0, 1.0],
        }
    )
    # Add power column
    data = data.with_columns(
        (pl.col("Voltage [V]") * pl.col("Current [A]")).alias("Power [W]")
    )
    out = iwdata.transform.set_energy(data)
    # Positive power = discharge (4W * time in hours)
    assert out["Discharge energy [W.h]"].to_list() == [
        0.0,
        4.0 / 3600,
        8.0 / 3600,
        12.0 / 3600,
    ]
    # No charge with positive power
    assert out["Charge energy [W.h]"].to_list() == [0.0, 0.0, 0.0, 0.0]


def test_set_capacity_from_existing_column():
    # Test splitting existing Capacity column based on current direction
    data = pl.DataFrame(
        {
            "Time [s]": [0.0, 1.0, 2.0, 3.0, 4.0],
            "Current [A]": [1.0, 1.0, -0.5, -0.5, 1.0],
            "Capacity [A.h]": [0.0, 0.003, 0.005, 0.0036, 0.002],
        }
    )
    out = iwdata.transform.set_capacity(data)
    # Check that discharge capacity increases with positive current
    assert out["Discharge capacity [A.h]"][0] == 0.0
    assert out["Discharge capacity [A.h]"][1] == 0.003
    assert out["Discharge capacity [A.h]"][2] == 0.003  # No change during charge
    # Check that charge capacity increases with negative current
    assert out["Charge capacity [A.h]"][0] == 0.0
    assert out["Charge capacity [A.h]"][1] == 0.0  # No change during discharge
    assert out["Charge capacity [A.h]"][2] > 0.0  # Increases during charge


def test_set_energy_from_existing_column():
    # Test splitting existing Energy column based on power direction
    data = pl.DataFrame(
        {
            "Time [s]": [0.0, 1.0, 2.0, 3.0, 4.0],
            "Power [W]": [4.0, 4.0, -2.0, -2.0, 3.0],
            "Energy [W.h]": [0.0, 0.011, 0.016, 0.011, 0.008],
        }
    )
    out = iwdata.transform.set_energy(data)
    # Check that discharge energy increases with positive power
    assert out["Discharge energy [W.h]"][0] == 0.0
    assert out["Discharge energy [W.h]"][1] == 0.011
    assert out["Discharge energy [W.h]"][2] == 0.011  # No change during charge
    # Check that charge energy increases with negative power
    assert out["Charge energy [W.h]"][0] == 0.0
    assert out["Charge energy [W.h]"][1] == 0.0  # No change during discharge
    assert out["Charge energy [W.h]"][2] > 0.0  # Increases during charge


def test_set_net_capacity():
    """Test that set_net_capacity calculates net capacity (discharge - charge)."""
    # Test case 1: Basic discharge and charge
    data1 = pl.DataFrame(
        {
            "Discharge capacity [A.h]": [0, 1, 2, 3, 4, 4, 4],
            "Charge capacity [A.h]": [0, 0, 0, 0, 0, 1, 2],
        }
    )
    out1 = iwdata.transform.set_net_capacity(data1)
    # Net capacity = discharge - charge
    assert out1["Capacity [A.h]"].to_list() == [0, 1, 2, 3, 4, 3, 2]

    # Test case 2: Raises error if column already exists
    data2 = pl.DataFrame(
        {
            "Discharge capacity [A.h]": [0, 1, 2],
            "Charge capacity [A.h]": [0, 0, 0],
            "Capacity [A.h]": [0, 1, 2],
        }
    )
    with pytest.raises(ValueError, match="Column 'Capacity \\[A.h\\]' already exists"):
        iwdata.transform.set_net_capacity(data2)

    # Test case 3: Calculates capacity from current if columns don't exist
    data3 = pl.DataFrame(
        {
            "Time [s]": [0.0, 1.0, 2.0, 3.0],
            "Current [A]": [1.0, 1.0, -1.0, -1.0],
        }
    )
    out3 = iwdata.transform.set_net_capacity(data3)
    assert "Capacity [A.h]" in out3.columns
    # First two points: discharge, last two: charge
    # Net should be positive initially then decrease
    assert out3["Capacity [A.h]"][0] == 0.0
    assert out3["Capacity [A.h]"][1] > 0.0  # Net discharge
    assert out3["Capacity [A.h]"][3] < out3["Capacity [A.h]"][1]  # Charging reduces net

    # Test case 4: With density units
    data4 = pl.DataFrame(
        {
            "Discharge capacity [mA.h.cm-2]": [0.0, 1.0, 2.0],
            "Charge capacity [mA.h.cm-2]": [0.0, 0.0, 0.5],
        }
    )
    out4 = iwdata.transform.set_net_capacity(
        data4, options={"current units": "density"}
    )
    assert "Capacity [mA.h.cm-2]" in out4.columns
    assert out4["Capacity [mA.h.cm-2]"].to_list() == [0.0, 1.0, 1.5]


def test_set_nominal_soc():
    # Net capacity = discharge - charge
    # During discharge: discharge increases, charge stays 0
    # During charge: discharge stays at peak, charge increases
    data = pl.DataFrame(
        {
            "Discharge capacity [A.h]": [0, 1, 2, 3, 4, 4, 4],
            "Charge capacity [A.h]": [0, 0, 0, 0, 0, 1, 2],
        }
    )
    metadata = {"Nominal cell capacity [A.h]": 4}
    out = iwdata.transform.set_nominal_soc(data, metadata)
    # Net capacity: [0, 1, 2, 3, 4, 3, 2]
    # SOC = 1 - net/Q = 1 - net/4
    assert [round(x, 8) for x in out["Nominal SOC"].to_list()] == [
        1.0,  # 1 - 0/4
        0.75,  # 1 - 1/4
        0.5,  # 1 - 2/4
        0.25,  # 1 - 3/4
        0.0,  # 1 - 4/4
        0.25,  # 1 - 3/4
        0.5,  # 1 - 2/4
    ]


def test_offset_duplicate_times():
    data = pl.DataFrame({"Time [s]": [1.0, 1.0, 2.0, 2.0]})
    out = iwdata.transform.offset_duplicate_times(data)
    assert out["Time [s]"].to_list() == [1.0, 1.000001, 2.0, 2.000001]


def test_convert_current_density_to_total_current():
    data = pl.DataFrame({"Current [mA.cm-2]": [1.0, 2.0, 3.0]})
    metadata = {"Electrode area [cm2]": 10.0}
    out = iwdata.transform.convert_current_density_to_total_current(data, metadata)
    assert out.columns == ["Current [A]"]
    assert out["Current [A]"].to_list() == [0.01, 0.02, 0.03]


def test_convert_total_current_to_current_density():
    data = pl.DataFrame({"Current [A]": [0.01, 0.02, 0.03]})
    metadata = {"Electrode area [cm2]": 10.0}
    out = iwdata.transform.convert_total_current_to_current_density(data, metadata)
    assert out.columns == ["Current [mA.cm-2]"]
    assert out["Current [mA.cm-2]"].to_list() == [1.0, 2.0, 3.0]


def test_set_positive_current_for_discharge():
    # Test case 1: Positive current is charging
    data1 = pl.DataFrame(
        {
            "Time [s]": [0, 1, 2, 3],
            "Current [A]": [1, 1, -1, -1],
            "Voltage [V]": [3.0, 3.1, 3.0, 2.9],
            "Step count": [0, 0, 1, 1],
        }
    )
    expected1 = pl.DataFrame(
        {
            "Time [s]": [0, 1, 2, 3],
            "Current [A]": [-1, -1, 1, 1],
            "Voltage [V]": [3.0, 3.1, 3.0, 2.9],
            "Step count": [0, 0, 1, 1],
        }
    )
    out1 = iwdata.transform.set_positive_current_for_discharge(data1)
    assert out1.to_dicts() == expected1.to_dicts()

    # Test case 2: Positive current is already discharging
    data2 = pl.DataFrame(
        {
            "Time [s]": [0, 1, 2, 3],
            "Current [A]": [1, 1, -1, -1],
            "Voltage [V]": [3.1, 3.0, 3.0, 3.1],
            "Step count": [0, 0, 1, 1],
        }
    )
    expected2 = data2.clone()
    out2 = iwdata.transform.set_positive_current_for_discharge(data2)
    assert out2.to_dicts() == expected2.to_dicts()

    # Test case 3: Positive current is charging and step count isn't set based on current sign
    data3 = pl.DataFrame(
        {
            "Time [s]": [0, 1, 2, 3],
            "Current [A]": [1, 1, -1, -1],
            "Voltage [V]": [3.0, 3.1, 3.0, 2.9],
            "Step count": [0, 1, 2, 3],
        }
    )
    expected3 = pl.DataFrame(
        {
            "Time [s]": [0, 1, 2, 3],
            "Current [A]": [-1, -1, 1, 1],
            "Voltage [V]": [3.0, 3.1, 3.0, 2.9],
            "Step count": [0, 1, 2, 3],
        }
    )
    out3 = iwdata.transform.set_positive_current_for_discharge(data3)
    assert out3.to_dicts() == expected3.to_dicts()


def test_remove_outliers():
    # Create a sample DataFrame
    data = pl.DataFrame(
        {"Step number": [1, 1, 1, 2, 2, 2], "Value": [10, 1, 2, 3, 4, 20]}
    )

    # Test case 1: Remove outliers without data_range
    result = iwdata.transform.remove_outliers(data, "Value", z_threshold=1)
    assert result.height == 4, "Should remove two outliers"
    assert 10 not in result["Value"].to_list(), "Should remove the outlier value 10"
    assert 20 not in result["Value"].to_list(), "Should remove the outlier value 20"

    # Test case 2: Remove outliers with data_range
    result = iwdata.transform.remove_outliers(
        data, "Value", z_threshold=1, data_range=slice(0, 1)
    )
    assert result.height == 5, "Should remove one outlier from the second step"
    assert 10 not in result["Value"].to_list(), "Should remove the outlier value 10"
    assert 20 in result["Value"].to_list(), (
        "Should not remove the value 20 as it's outside the range"
    )

    # Test case 3: No outliers removed with high z_threshold
    result = iwdata.transform.remove_outliers(data, "Value", z_threshold=3)
    assert result.height == 6, "Should not remove any outliers"

    print("All tests passed!")


def test_capacity_always_positive():
    """Test that discharge and charge capacity are always >= 0."""
    # Test case 1: Pure discharge
    data1 = pl.DataFrame(
        {
            "Time [s]": [0.0, 1.0, 2.0, 3.0],
            "Current [A]": [1.0, 1.0, 1.0, 1.0],
        }
    )
    out1 = iwdata.transform.set_capacity(data1)
    assert all(out1["Discharge capacity [A.h]"] >= 0)
    assert all(out1["Charge capacity [A.h]"] >= 0)

    # Test case 2: Pure charge
    data2 = pl.DataFrame(
        {
            "Time [s]": [0.0, 1.0, 2.0, 3.0],
            "Current [A]": [-1.0, -1.0, -1.0, -1.0],
        }
    )
    out2 = iwdata.transform.set_capacity(data2)
    assert all(out2["Discharge capacity [A.h]"] >= 0)
    assert all(out2["Charge capacity [A.h]"] >= 0)

    # Test case 3: Mixed discharge and charge
    data3 = pl.DataFrame(
        {
            "Time [s]": [0.0, 1.0, 2.0, 3.0, 4.0],
            "Current [A]": [1.0, 1.0, -1.0, -1.0, 1.0],
        }
    )
    out3 = iwdata.transform.set_capacity(data3)
    assert all(out3["Discharge capacity [A.h]"] >= 0)
    assert all(out3["Charge capacity [A.h]"] >= 0)

    # Test case 4: With step resets
    data4 = pl.DataFrame(
        {
            "Time [s]": [0.0, 1.0, 2.0, 3.0, 4.0],
            "Current [A]": [1.0, 1.0, -1.0, -1.0, 1.0],
            "Step count": [0, 0, 1, 1, 2],
        }
    )
    out4 = iwdata.transform.set_capacity(data4)
    assert all(out4["Discharge capacity [A.h]"] >= 0)
    assert all(out4["Charge capacity [A.h]"] >= 0)

    # Test case 5: From existing capacity column
    data5 = pl.DataFrame(
        {
            "Time [s]": [0.0, 1.0, 2.0, 3.0, 4.0],
            "Current [A]": [1.0, 1.0, -0.5, -0.5, 1.0],
            "Capacity [A.h]": [0.0, 0.003, 0.005, 0.0036, 0.002],
        }
    )
    out5 = iwdata.transform.set_capacity(data5)
    assert all(out5["Discharge capacity [A.h]"] >= 0)
    assert all(out5["Charge capacity [A.h]"] >= 0)


def test_energy_always_positive():
    """Test that discharge and charge energy are always >= 0."""
    # Test case 1: Pure discharge
    data1 = pl.DataFrame(
        {
            "Time [s]": [0.0, 1.0, 2.0, 3.0],
            "Voltage [V]": [4.0, 4.0, 4.0, 4.0],
            "Current [A]": [1.0, 1.0, 1.0, 1.0],
        }
    )
    data1 = data1.with_columns(
        (pl.col("Voltage [V]") * pl.col("Current [A]")).alias("Power [W]")
    )
    out1 = iwdata.transform.set_energy(data1)
    assert all(out1["Discharge energy [W.h]"] >= 0)
    assert all(out1["Charge energy [W.h]"] >= 0)

    # Test case 2: Pure charge
    data2 = pl.DataFrame(
        {
            "Time [s]": [0.0, 1.0, 2.0, 3.0],
            "Voltage [V]": [4.0, 4.0, 4.0, 4.0],
            "Current [A]": [-1.0, -1.0, -1.0, -1.0],
        }
    )
    data2 = data2.with_columns(
        (pl.col("Voltage [V]") * pl.col("Current [A]")).alias("Power [W]")
    )
    out2 = iwdata.transform.set_energy(data2)
    assert all(out2["Discharge energy [W.h]"] >= 0)
    assert all(out2["Charge energy [W.h]"] >= 0)

    # Test case 3: Mixed discharge and charge
    data3 = pl.DataFrame(
        {
            "Time [s]": [0.0, 1.0, 2.0, 3.0, 4.0],
            "Voltage [V]": [4.0, 4.0, 4.0, 4.0, 4.0],
            "Current [A]": [1.0, 1.0, -1.0, -1.0, 1.0],
        }
    )
    data3 = data3.with_columns(
        (pl.col("Voltage [V]") * pl.col("Current [A]")).alias("Power [W]")
    )
    out3 = iwdata.transform.set_energy(data3)
    assert all(out3["Discharge energy [W.h]"] >= 0)
    assert all(out3["Charge energy [W.h]"] >= 0)

    # Test case 4: With step resets
    data4 = pl.DataFrame(
        {
            "Time [s]": [0.0, 1.0, 2.0, 3.0, 4.0],
            "Voltage [V]": [4.0, 4.0, 4.0, 4.0, 4.0],
            "Current [A]": [1.0, 1.0, -1.0, -1.0, 1.0],
            "Step count": [0, 0, 1, 1, 2],
        }
    )
    data4 = data4.with_columns(
        (pl.col("Voltage [V]") * pl.col("Current [A]")).alias("Power [W]")
    )
    out4 = iwdata.transform.set_energy(data4)
    assert all(out4["Discharge energy [W.h]"] >= 0)
    assert all(out4["Charge energy [W.h]"] >= 0)

    # Test case 5: From existing energy column
    data5 = pl.DataFrame(
        {
            "Time [s]": [0.0, 1.0, 2.0, 3.0, 4.0],
            "Power [W]": [4.0, 4.0, -2.0, -2.0, 3.0],
            "Energy [W.h]": [0.0, 0.011, 0.016, 0.011, 0.008],
        }
    )
    out5 = iwdata.transform.set_energy(data5)
    assert all(out5["Discharge energy [W.h]"] >= 0)
    assert all(out5["Charge energy [W.h]"] >= 0)


def test_capacity_starts_at_zero_each_step():
    """Test that discharge and charge capacity start at 0 for each step."""
    # Test case 1: Multiple steps with discharge
    data1 = pl.DataFrame(
        {
            "Time [s]": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "Current [A]": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "Step count": [0, 0, 1, 1, 2, 2],
        }
    )
    out1 = iwdata.transform.set_capacity(data1)
    # Check that first point of each step has capacity = 0
    assert out1["Discharge capacity [A.h]"][0] == 0.0  # Step 1 start
    assert out1["Discharge capacity [A.h]"][2] == 0.0  # Step 2 start
    assert out1["Discharge capacity [A.h]"][4] == 0.0  # Step 3 start
    assert out1["Charge capacity [A.h]"][0] == 0.0
    assert out1["Charge capacity [A.h]"][2] == 0.0
    assert out1["Charge capacity [A.h]"][4] == 0.0

    # Test case 2: Multiple steps with charge
    data2 = pl.DataFrame(
        {
            "Time [s]": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "Current [A]": [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
            "Step count": [0, 0, 1, 1, 2, 2],
        }
    )
    out2 = iwdata.transform.set_capacity(data2)
    # Check that first point of each step has capacity = 0
    assert out2["Discharge capacity [A.h]"][0] == 0.0
    assert out2["Discharge capacity [A.h]"][2] == 0.0
    assert out2["Discharge capacity [A.h]"][4] == 0.0
    assert out2["Charge capacity [A.h]"][0] == 0.0
    assert out2["Charge capacity [A.h]"][2] == 0.0
    assert out2["Charge capacity [A.h]"][4] == 0.0

    # Test case 3: Mixed discharge and charge steps
    data3 = pl.DataFrame(
        {
            "Time [s]": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "Current [A]": [1.0, 1.0, -1.0, -1.0, 1.0, 1.0],
            "Step count": [0, 0, 1, 1, 2, 2],
        }
    )
    out3 = iwdata.transform.set_capacity(data3)
    # Check that first point of each step has capacity = 0
    assert out3["Discharge capacity [A.h]"][0] == 0.0
    assert out3["Discharge capacity [A.h]"][2] == 0.0
    assert out3["Discharge capacity [A.h]"][4] == 0.0
    assert out3["Charge capacity [A.h]"][0] == 0.0
    assert out3["Charge capacity [A.h]"][2] == 0.0
    assert out3["Charge capacity [A.h]"][4] == 0.0

    # Test case 4: From existing capacity column with steps
    data4 = pl.DataFrame(
        {
            "Time [s]": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "Current [A]": [1.0, 1.0, -1.0, -1.0, 1.0, 1.0],
            "Capacity [A.h]": [0.0, 0.003, 0.005, 0.0036, 0.002, 0.004],
            "Step count": [0, 0, 1, 1, 2, 2],
        }
    )
    out4 = iwdata.transform.set_capacity(data4)
    # Check that first point of each step has capacity = 0
    assert out4["Discharge capacity [A.h]"][0] == 0.0
    assert out4["Discharge capacity [A.h]"][2] == 0.0
    assert out4["Discharge capacity [A.h]"][4] == 0.0
    assert out4["Charge capacity [A.h]"][0] == 0.0
    assert out4["Charge capacity [A.h]"][2] == 0.0
    assert out4["Charge capacity [A.h]"][4] == 0.0


def test_energy_starts_at_zero_each_step():
    """Test that discharge and charge energy start at 0 for each step."""
    # Test case 1: Multiple steps with discharge
    data1 = pl.DataFrame(
        {
            "Time [s]": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "Voltage [V]": [4.0, 4.0, 4.0, 4.0, 4.0, 4.0],
            "Current [A]": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "Step count": [0, 0, 1, 1, 2, 2],
        }
    )
    data1 = data1.with_columns(
        (pl.col("Voltage [V]") * pl.col("Current [A]")).alias("Power [W]")
    )
    out1 = iwdata.transform.set_energy(data1)
    # Check that first point of each step has energy = 0
    assert out1["Discharge energy [W.h]"][0] == 0.0  # Step 1 start
    assert out1["Discharge energy [W.h]"][2] == 0.0  # Step 2 start
    assert out1["Discharge energy [W.h]"][4] == 0.0  # Step 3 start
    assert out1["Charge energy [W.h]"][0] == 0.0
    assert out1["Charge energy [W.h]"][2] == 0.0
    assert out1["Charge energy [W.h]"][4] == 0.0

    # Test case 2: Multiple steps with charge
    data2 = pl.DataFrame(
        {
            "Time [s]": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "Voltage [V]": [4.0, 4.0, 4.0, 4.0, 4.0, 4.0],
            "Current [A]": [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
            "Step count": [0, 0, 1, 1, 2, 2],
        }
    )
    data2 = data2.with_columns(
        (pl.col("Voltage [V]") * pl.col("Current [A]")).alias("Power [W]")
    )
    out2 = iwdata.transform.set_energy(data2)
    # Check that first point of each step has energy = 0
    assert out2["Discharge energy [W.h]"][0] == 0.0
    assert out2["Discharge energy [W.h]"][2] == 0.0
    assert out2["Discharge energy [W.h]"][4] == 0.0
    assert out2["Charge energy [W.h]"][0] == 0.0
    assert out2["Charge energy [W.h]"][2] == 0.0
    assert out2["Charge energy [W.h]"][4] == 0.0

    # Test case 3: Mixed discharge and charge steps
    data3 = pl.DataFrame(
        {
            "Time [s]": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "Voltage [V]": [4.0, 4.0, 4.0, 4.0, 4.0, 4.0],
            "Current [A]": [1.0, 1.0, -1.0, -1.0, 1.0, 1.0],
            "Step count": [0, 0, 1, 1, 2, 2],
        }
    )
    data3 = data3.with_columns(
        (pl.col("Voltage [V]") * pl.col("Current [A]")).alias("Power [W]")
    )
    out3 = iwdata.transform.set_energy(data3)
    # Check that first point of each step has energy = 0
    assert out3["Discharge energy [W.h]"][0] == 0.0
    assert out3["Discharge energy [W.h]"][2] == 0.0
    assert out3["Discharge energy [W.h]"][4] == 0.0
    assert out3["Charge energy [W.h]"][0] == 0.0
    assert out3["Charge energy [W.h]"][2] == 0.0
    assert out3["Charge energy [W.h]"][4] == 0.0

    # Test case 4: From existing energy column with steps
    data4 = pl.DataFrame(
        {
            "Time [s]": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "Power [W]": [4.0, 4.0, -2.0, -2.0, 3.0, 3.0],
            "Energy [W.h]": [0.0, 0.011, 0.016, 0.011, 0.008, 0.012],
            "Step count": [0, 0, 1, 1, 2, 2],
        }
    )
    out4 = iwdata.transform.set_energy(data4)
    # Check that first point of each step has energy = 0
    assert out4["Discharge energy [W.h]"][0] == 0.0
    assert out4["Discharge energy [W.h]"][2] == 0.0
    assert out4["Discharge energy [W.h]"][4] == 0.0
    assert out4["Charge energy [W.h]"][0] == 0.0
    assert out4["Charge energy [W.h]"][2] == 0.0
    assert out4["Charge energy [W.h]"][4] == 0.0


def test_capacity_from_existing_single_column_with_steps():
    """Test that capacity calculated from existing single column with steps
    resets to 0 at each step and is always positive."""
    # Test case 1: Single capacity column with steps, non-zero at step boundaries
    data1 = pl.DataFrame(
        {
            "Time [s]": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "Current [A]": [1.0, 1.0, -1.0, -1.0, 1.0, 1.0],
            "Capacity [A.h]": [0.0, 0.003, 0.005, 0.0036, 0.002, 0.004],
            "Step count": [0, 0, 1, 1, 2, 2],
        }
    )
    out1 = iwdata.transform.set_capacity(data1)
    # Verify reset to 0 at each step start
    assert out1["Discharge capacity [A.h]"][0] == 0.0
    assert out1["Discharge capacity [A.h]"][2] == 0.0
    assert out1["Discharge capacity [A.h]"][4] == 0.0
    assert out1["Charge capacity [A.h]"][0] == 0.0
    assert out1["Charge capacity [A.h]"][2] == 0.0
    assert out1["Charge capacity [A.h]"][4] == 0.0
    # Verify always positive
    assert all(out1["Discharge capacity [A.h]"] >= 0)
    assert all(out1["Charge capacity [A.h]"] >= 0)

    # Test case 2: Single capacity column with steps, increasing values
    data2 = pl.DataFrame(
        {
            "Time [s]": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "Current [A]": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "Capacity [A.h]": [0.0, 0.001, 0.002, 0.003, 0.004, 0.005],
            "Step count": [0, 0, 1, 1, 2, 2],
        }
    )
    out2 = iwdata.transform.set_capacity(data2)
    # Verify reset to 0 at each step start
    assert out2["Discharge capacity [A.h]"][0] == 0.0
    assert out2["Discharge capacity [A.h]"][2] == 0.0
    assert out2["Discharge capacity [A.h]"][4] == 0.0
    # Verify always positive
    assert all(out2["Discharge capacity [A.h]"] >= 0)
    assert all(out2["Charge capacity [A.h]"] >= 0)

    # Test case 3: Single capacity column with steps, decreasing values
    # (capacity decreases during charge)
    data3 = pl.DataFrame(
        {
            "Time [s]": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "Current [A]": [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
            "Capacity [A.h]": [0.005, 0.004, 0.003, 0.002, 0.001, 0.0],
            "Step count": [0, 0, 1, 1, 2, 2],
        }
    )
    out3 = iwdata.transform.set_capacity(data3)
    # Verify reset to 0 at each step start
    assert out3["Discharge capacity [A.h]"][0] == 0.0
    assert out3["Discharge capacity [A.h]"][2] == 0.0
    assert out3["Discharge capacity [A.h]"][4] == 0.0
    assert out3["Charge capacity [A.h]"][0] == 0.0
    assert out3["Charge capacity [A.h]"][2] == 0.0
    assert out3["Charge capacity [A.h]"][4] == 0.0
    # Verify always positive
    assert all(out3["Discharge capacity [A.h]"] >= 0)
    assert all(out3["Charge capacity [A.h]"] >= 0)


def test_energy_from_existing_single_column_with_steps():
    """Test that energy calculated from existing single column with steps
    resets to 0 at each step and is always positive."""
    # Test case 1: Single energy column with steps, non-zero at step boundaries
    data1 = pl.DataFrame(
        {
            "Time [s]": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "Power [W]": [4.0, 4.0, -2.0, -2.0, 3.0, 3.0],
            "Energy [W.h]": [0.0, 0.011, 0.016, 0.011, 0.008, 0.012],
            "Step count": [0, 0, 1, 1, 2, 2],
        }
    )
    out1 = iwdata.transform.set_energy(data1)
    # Verify reset to 0 at each step start
    assert out1["Discharge energy [W.h]"][0] == 0.0
    assert out1["Discharge energy [W.h]"][2] == 0.0
    assert out1["Discharge energy [W.h]"][4] == 0.0
    assert out1["Charge energy [W.h]"][0] == 0.0
    assert out1["Charge energy [W.h]"][2] == 0.0
    assert out1["Charge energy [W.h]"][4] == 0.0
    # Verify always positive
    assert all(out1["Discharge energy [W.h]"] >= 0)
    assert all(out1["Charge energy [W.h]"] >= 0)

    # Test case 2: Single energy column with steps, increasing values
    data2 = pl.DataFrame(
        {
            "Time [s]": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "Power [W]": [4.0, 4.0, 4.0, 4.0, 4.0, 4.0],
            "Energy [W.h]": [0.0, 0.001, 0.002, 0.003, 0.004, 0.005],
            "Step count": [0, 0, 1, 1, 2, 2],
        }
    )
    out2 = iwdata.transform.set_energy(data2)
    # Verify reset to 0 at each step start
    assert out2["Discharge energy [W.h]"][0] == 0.0
    assert out2["Discharge energy [W.h]"][2] == 0.0
    assert out2["Discharge energy [W.h]"][4] == 0.0
    # Verify always positive
    assert all(out2["Discharge energy [W.h]"] >= 0)
    assert all(out2["Charge energy [W.h]"] >= 0)

    # Test case 3: Single energy column with steps, decreasing values
    # (energy decreases during charge)
    data3 = pl.DataFrame(
        {
            "Time [s]": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "Power [W]": [-2.0, -2.0, -2.0, -2.0, -2.0, -2.0],
            "Energy [W.h]": [0.005, 0.004, 0.003, 0.002, 0.001, 0.0],
            "Step count": [0, 0, 1, 1, 2, 2],
        }
    )
    out3 = iwdata.transform.set_energy(data3)
    # Verify reset to 0 at each step start
    assert out3["Discharge energy [W.h]"][0] == 0.0
    assert out3["Discharge energy [W.h]"][2] == 0.0
    assert out3["Discharge energy [W.h]"][4] == 0.0
    assert out3["Charge energy [W.h]"][0] == 0.0
    assert out3["Charge energy [W.h]"][2] == 0.0
    assert out3["Charge energy [W.h]"][4] == 0.0
    # Verify always positive
    assert all(out3["Discharge energy [W.h]"] >= 0)
    assert all(out3["Charge energy [W.h]"] >= 0)


def test_capacity_from_existing_split_columns_with_steps():
    """Test that when already-split capacity columns exist, set_capacity
    uses them if valid, or transforms them (abs + reset) if invalid."""
    # Test case 1: Valid split columns - should be used as-is
    data1 = pl.DataFrame(
        {
            "Time [s]": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "Current [A]": [1.0, 1.0, -1.0, -1.0, 1.0, 1.0],
            "Discharge capacity [A.h]": [0.0, 0.001, 0.0, 0.0, 0.0, 0.001],
            "Charge capacity [A.h]": [0.0, 0.0, 0.0, 0.001, 0.0, 0.0],
            "Step count": [0, 0, 1, 1, 2, 2],
        }
    )
    out1 = iwdata.transform.set_capacity(data1)
    # Should use existing columns (they're valid)
    assert out1["Discharge capacity [A.h]"][0] == 0.0
    assert out1["Discharge capacity [A.h]"][1] == 0.001
    assert out1["Discharge capacity [A.h]"][2] == 0.0
    assert out1["Charge capacity [A.h]"][2] == 0.0
    assert out1["Charge capacity [A.h]"][3] == 0.001

    # Test case 2: Invalid split columns with negative values - should transform
    data2 = pl.DataFrame(
        {
            "Time [s]": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "Current [A]": [1.0, 1.0, -1.0, -1.0, 1.0, 1.0],
            "Discharge capacity [A.h]": [0.0, 0.001, -0.001, 0.0, 0.0, 0.001],
            "Charge capacity [A.h]": [0.0, 0.0, 0.0, 0.001, 0.0, 0.0],
            "Step count": [0, 0, 1, 1, 2, 2],
        }
    )
    out2 = iwdata.transform.set_capacity(data2)
    # Should transform (abs applied, step resets fixed)
    assert all(out2["Discharge capacity [A.h]"] >= 0)
    assert all(out2["Charge capacity [A.h]"] >= 0)
    assert out2["Discharge capacity [A.h]"][0] == 0.0
    assert out2["Discharge capacity [A.h]"][2] == 0.0
    assert out2["Discharge capacity [A.h]"][4] == 0.0
    # The negative value should be made positive
    assert out2["Discharge capacity [A.h]"][2] == 0.0  # After reset

    # Test case 3: Invalid split columns with incorrect step resets - should transform
    data3 = pl.DataFrame(
        {
            "Time [s]": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "Current [A]": [1.0, 1.0, -1.0, -1.0, 1.0, 1.0],
            "Discharge capacity [A.h]": [0.0, 0.001, 0.002, 0.001, 0.0, 0.001],
            "Charge capacity [A.h]": [0.0, 0.0, 0.001, 0.002, 0.001, 0.0],
            "Step count": [0, 0, 1, 1, 2, 2],
        }
    )
    out3 = iwdata.transform.set_capacity(data3)
    # Should transform (step resets fixed)
    assert out3["Discharge capacity [A.h]"][0] == 0.0
    assert out3["Discharge capacity [A.h]"][2] == 0.0
    assert out3["Discharge capacity [A.h]"][4] == 0.0
    assert all(out3["Discharge capacity [A.h]"] >= 0)
    assert all(out3["Charge capacity [A.h]"] >= 0)


def test_energy_from_existing_split_columns_with_steps():
    """Test that when already-split energy columns exist, set_energy
    uses them if valid, or transforms them (abs + reset) if invalid."""
    # Test case 1: Valid split columns - should be used as-is
    data1 = pl.DataFrame(
        {
            "Time [s]": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "Power [W]": [4.0, 4.0, -2.0, -2.0, 3.0, 3.0],
            "Discharge energy [W.h]": [0.0, 0.001, 0.0, 0.0, 0.0, 0.001],
            "Charge energy [W.h]": [0.0, 0.0, 0.0, 0.001, 0.0, 0.0],
            "Step count": [0, 0, 1, 1, 2, 2],
        }
    )
    out1 = iwdata.transform.set_energy(data1)
    # Should use existing columns (they're valid)
    assert out1["Discharge energy [W.h]"][0] == 0.0
    assert out1["Discharge energy [W.h]"][1] == 0.001
    assert out1["Discharge energy [W.h]"][2] == 0.0
    assert out1["Charge energy [W.h]"][2] == 0.0
    assert out1["Charge energy [W.h]"][3] == 0.001

    # Test case 2: Invalid split columns with negative values - should transform
    data2 = pl.DataFrame(
        {
            "Time [s]": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "Power [W]": [4.0, 4.0, -2.0, -2.0, 3.0, 3.0],
            "Discharge energy [W.h]": [0.0, 0.001, -0.001, 0.0, 0.0, 0.001],
            "Charge energy [W.h]": [0.0, 0.0, 0.0, 0.001, 0.0, 0.0],
            "Step count": [0, 0, 1, 1, 2, 2],
        }
    )
    out2 = iwdata.transform.set_energy(data2)
    # Should transform (abs applied, step resets fixed)
    assert all(out2["Discharge energy [W.h]"] >= 0)
    assert all(out2["Charge energy [W.h]"] >= 0)
    assert out2["Discharge energy [W.h]"][0] == 0.0
    assert out2["Discharge energy [W.h]"][2] == 0.0
    assert out2["Discharge energy [W.h]"][4] == 0.0

    # Test case 3: Invalid split columns with incorrect step resets - should transform
    data3 = pl.DataFrame(
        {
            "Time [s]": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "Power [W]": [4.0, 4.0, -2.0, -2.0, 3.0, 3.0],
            "Discharge energy [W.h]": [0.0, 0.001, 0.002, 0.001, 0.0, 0.001],
            "Charge energy [W.h]": [0.0, 0.0, 0.001, 0.002, 0.001, 0.0],
            "Step count": [0, 0, 1, 1, 2, 2],
        }
    )
    out3 = iwdata.transform.set_energy(data3)
    # Should transform (step resets fixed)
    assert out3["Discharge energy [W.h]"][0] == 0.0
    assert out3["Discharge energy [W.h]"][2] == 0.0
    assert out3["Discharge energy [W.h]"][4] == 0.0
    assert all(out3["Discharge energy [W.h]"] >= 0)
    assert all(out3["Charge energy [W.h]"] >= 0)


def test_capacity_with_zero_current():
    """Test capacity calculation with zero current values."""
    # Test case 1: All zero current
    data1 = pl.DataFrame(
        {
            "Time [s]": [0.0, 1.0, 2.0, 3.0],
            "Current [A]": [0.0, 0.0, 0.0, 0.0],
        }
    )
    out1 = iwdata.transform.set_capacity(data1)
    assert all(out1["Discharge capacity [A.h]"] == 0.0)
    assert all(out1["Charge capacity [A.h]"] == 0.0)

    # Test case 2: Mixed zero and non-zero current
    data2 = pl.DataFrame(
        {
            "Time [s]": [0.0, 1.0, 2.0, 3.0, 4.0],
            "Current [A]": [1.0, 0.0, 0.0, -1.0, 0.0],
        }
    )
    out2 = iwdata.transform.set_capacity(data2)
    # Capacity should not change during zero current periods (pure zero)
    assert out2["Discharge capacity [A.h]"][0] == 0.0
    assert out2["Discharge capacity [A.h]"][1] > 0.0
    # Discharge capacity doesn't change during zero current (indices 1-3)
    assert out2["Discharge capacity [A.h]"][1] == out2["Discharge capacity [A.h]"][2]
    assert out2["Discharge capacity [A.h]"][2] == out2["Discharge capacity [A.h]"][3]
    # Charge capacity increases during negative current (index 3)
    # Note: trapezoidal integration includes the transition, so index 4 may
    # have slightly more charge capacity than index 3
    assert out2["Charge capacity [A.h]"][3] > out2["Charge capacity [A.h]"][2]
    assert out2["Charge capacity [A.h]"][4] >= out2["Charge capacity [A.h]"][3]


def test_capacity_with_decreasing_single_column():
    """Test capacity calculation when single capacity column decreases
    (e.g., Novonix style where capacity decreases during discharge)."""
    # Test case 1: Capacity decreases during discharge (Novonix style)
    data1 = pl.DataFrame(
        {
            "Time [s]": [0.0, 1.0, 2.0, 3.0, 4.0],
            "Current [A]": [1.0, 1.0, 1.0, 1.0, 1.0],
            "Capacity [A.h]": [5.0, 4.0, 3.0, 2.0, 1.0],  # Decreasing
        }
    )
    out1 = iwdata.transform.set_capacity(data1)
    # Should handle decreasing capacity correctly
    assert all(out1["Discharge capacity [A.h]"] >= 0)
    assert all(out1["Charge capacity [A.h]"] >= 0)
    # When capacity decreases, the deltas are negative, but we take abs
    # The initial value (5.0) is included in the first delta
    # So discharge capacity starts at 5.0 and decreases to 1.0
    # After abs, it should be positive and decreasing
    assert out1["Discharge capacity [A.h]"][0] >= 0.0
    assert out1["Discharge capacity [A.h]"][4] >= 0.0
    # The capacity should reflect the decreasing pattern
    assert out1["Discharge capacity [A.h]"][0] > out1["Discharge capacity [A.h]"][4]

    # Test case 2: Capacity decreases during discharge with steps
    data2 = pl.DataFrame(
        {
            "Time [s]": [0.0, 1.0, 2.0, 3.0, 4.0],
            "Current [A]": [1.0, 1.0, 1.0, 1.0, 1.0],
            "Capacity [A.h]": [5.0, 4.0, 3.0, 2.0, 1.0],
            "Step count": [0, 0, 1, 1, 2],
        }
    )
    out2 = iwdata.transform.set_capacity(data2)
    assert all(out2["Discharge capacity [A.h]"] >= 0)
    assert out2["Discharge capacity [A.h]"][0] == 0.0
    assert out2["Discharge capacity [A.h]"][2] == 0.0  # Reset at step boundary
    assert out2["Discharge capacity [A.h]"][4] == 0.0  # Reset at step boundary


def test_capacity_with_varying_time_steps():
    """Test capacity calculation with non-uniform time steps."""
    # Test case 1: Large time gaps
    data1 = pl.DataFrame(
        {
            "Time [s]": [0.0, 100.0, 200.0, 500.0, 1000.0],
            "Current [A]": [1.0, 1.0, 1.0, 1.0, 1.0],
        }
    )
    out1 = iwdata.transform.set_capacity(data1)
    # Capacity should accumulate correctly despite varying time steps
    assert out1["Discharge capacity [A.h]"][0] == 0.0
    assert out1["Discharge capacity [A.h]"][1] > 0.0
    assert out1["Discharge capacity [A.h]"][4] > out1["Discharge capacity [A.h]"][1]

    # Test case 2: Very small time steps
    data2 = pl.DataFrame(
        {
            "Time [s]": [0.0, 0.001, 0.002, 0.003, 0.004],
            "Current [A]": [1.0, 1.0, 1.0, 1.0, 1.0],
        }
    )
    out2 = iwdata.transform.set_capacity(data2)
    assert out2["Discharge capacity [A.h]"][0] == 0.0
    assert out2["Discharge capacity [A.h]"][4] > 0.0


def test_capacity_with_current_sign_changes_in_step():
    """Test capacity calculation when current changes sign within a step."""
    # Test case 1: Current changes from discharge to charge within step
    data1 = pl.DataFrame(
        {
            "Time [s]": [0.0, 1.0, 2.0, 3.0, 4.0],
            "Current [A]": [1.0, 1.0, -1.0, -1.0, 1.0],
            "Step count": [0, 0, 0, 0, 0],  # All same step
        }
    )
    out1 = iwdata.transform.set_capacity(data1)
    assert all(out1["Discharge capacity [A.h]"] >= 0)
    assert all(out1["Charge capacity [A.h]"] >= 0)
    # Discharge should accumulate during positive current
    assert out1["Discharge capacity [A.h]"][1] > out1["Discharge capacity [A.h]"][0]
    # Trapezoidal integration includes transitions, so discharge may increase
    # slightly at index 2 (transition from +1 to -1)
    assert out1["Discharge capacity [A.h]"][2] >= out1["Discharge capacity [A.h]"][1]
    # Charge should accumulate during negative current
    assert out1["Charge capacity [A.h]"][3] > out1["Charge capacity [A.h]"][2]
    # Trapezoidal integration includes transitions, so charge may increase
    # slightly at index 4 (transition from -1 to +1)
    assert out1["Charge capacity [A.h]"][4] >= out1["Charge capacity [A.h]"][3]


def test_capacity_with_different_units():
    """Test capacity calculation with different current and capacity units."""
    # Test case 1: Standard units (A and A.h)
    data1 = pl.DataFrame(
        {
            "Time [s]": [0.0, 1.0, 2.0, 3.0],
            "Current [A]": [1.0, 1.0, 1.0, 1.0],
        }
    )
    out1 = iwdata.transform.set_capacity(data1, options={"current units": "total"})
    assert "Discharge capacity [A.h]" in out1.columns
    assert "Charge capacity [A.h]" in out1.columns
    assert out1["Discharge capacity [A.h]"][0] == 0.0
    assert out1["Discharge capacity [A.h]"][3] > 0.0

    # Test case 2: Density units (mA.cm-2 and mA.h.cm-2)
    data2 = pl.DataFrame(
        {
            "Time [s]": [0.0, 1.0, 2.0, 3.0],
            "Current [mA.cm-2]": [1.0, 1.0, 1.0, 1.0],
        }
    )
    out2 = iwdata.transform.set_capacity(data2, options={"current units": "density"})
    assert "Discharge capacity [mA.h.cm-2]" in out2.columns
    assert "Charge capacity [mA.h.cm-2]" in out2.columns
    assert out2["Discharge capacity [mA.h.cm-2]"][0] == 0.0
    assert out2["Discharge capacity [mA.h.cm-2]"][3] > 0.0


def test_capacity_with_single_row():
    """Test capacity calculation with single row dataframe."""
    data = pl.DataFrame(
        {
            "Time [s]": [0.0],
            "Current [A]": [1.0],
        }
    )
    out = iwdata.transform.set_capacity(data)
    assert out["Discharge capacity [A.h]"][0] == 0.0
    assert out["Charge capacity [A.h]"][0] == 0.0


def test_capacity_with_very_small_values():
    """Test capacity calculation with very small current and time values."""
    data = pl.DataFrame(
        {
            "Time [s]": [0.0, 1e-6, 2e-6, 3e-6],
            "Current [A]": [1e-6, 1e-6, 1e-6, 1e-6],
        }
    )
    out = iwdata.transform.set_capacity(data)
    assert all(out["Discharge capacity [A.h]"] >= 0)
    assert all(out["Charge capacity [A.h]"] >= 0)
    assert out["Discharge capacity [A.h]"][0] == 0.0


def test_capacity_split_columns_missing_one():
    """Test capacity calculation when only one split column exists."""
    # Test case 1: Only discharge column exists
    data1 = pl.DataFrame(
        {
            "Time [s]": [0.0, 1.0, 2.0, 3.0],
            "Current [A]": [1.0, 1.0, 1.0, 1.0],
            "Discharge capacity [A.h]": [0.0, 0.001, 0.002, 0.003],
        }
    )
    # Should calculate charge from current
    out1 = iwdata.transform.set_capacity(data1)
    assert "Discharge capacity [A.h]" in out1.columns
    assert "Charge capacity [A.h]" in out1.columns
    assert all(out1["Discharge capacity [A.h]"] >= 0)
    assert all(out1["Charge capacity [A.h]"] >= 0)

    # Test case 2: Only charge column exists
    data2 = pl.DataFrame(
        {
            "Time [s]": [0.0, 1.0, 2.0, 3.0],
            "Current [A]": [-1.0, -1.0, -1.0, -1.0],
            "Charge capacity [A.h]": [0.0, 0.001, 0.002, 0.003],
        }
    )
    # Should calculate discharge from current
    out2 = iwdata.transform.set_capacity(data2)
    assert "Discharge capacity [A.h]" in out2.columns
    assert "Charge capacity [A.h]" in out2.columns
    assert all(out2["Discharge capacity [A.h]"] >= 0)
    assert all(out2["Charge capacity [A.h]"] >= 0)


def test_capacity_with_negative_split_columns():
    """Test capacity calculation with negative values in split columns."""
    # Test case 1: Negative discharge capacity (should be made positive)
    data1 = pl.DataFrame(
        {
            "Time [s]": [0.0, 1.0, 2.0, 3.0],
            "Current [A]": [1.0, 1.0, 1.0, 1.0],
            "Discharge capacity [A.h]": [0.0, -0.001, -0.002, -0.003],
            "Charge capacity [A.h]": [0.0, 0.0, 0.0, 0.0],
        }
    )
    out1 = iwdata.transform.set_capacity(data1)
    assert all(out1["Discharge capacity [A.h]"] >= 0)
    assert all(out1["Charge capacity [A.h]"] >= 0)

    # Test case 2: Negative charge capacity (should be made positive)
    data2 = pl.DataFrame(
        {
            "Time [s]": [0.0, 1.0, 2.0, 3.0],
            "Current [A]": [-1.0, -1.0, -1.0, -1.0],
            "Discharge capacity [A.h]": [0.0, 0.0, 0.0, 0.0],
            "Charge capacity [A.h]": [0.0, -0.001, -0.002, -0.003],
        }
    )
    out2 = iwdata.transform.set_capacity(data2)
    assert all(out2["Discharge capacity [A.h]"] >= 0)
    assert all(out2["Charge capacity [A.h]"] >= 0)


def test_capacity_with_single_column_increasing_charge():
    """Test capacity calculation when single capacity column increases
    during charge (normal case)."""
    data = pl.DataFrame(
        {
            "Time [s]": [0.0, 1.0, 2.0, 3.0, 4.0],
            "Current [A]": [-1.0, -1.0, -1.0, -1.0, -1.0],
            "Capacity [A.h]": [0.0, 0.001, 0.002, 0.003, 0.004],  # Increasing
        }
    )
    out = iwdata.transform.set_capacity(data)
    assert all(out["Discharge capacity [A.h]"] >= 0)
    assert all(out["Charge capacity [A.h]"] >= 0)
    assert out["Charge capacity [A.h]"][0] == 0.0
    assert out["Charge capacity [A.h]"][4] > out["Charge capacity [A.h]"][1]


def test_capacity_with_single_column_mixed_directions():
    """Test capacity calculation when single capacity column has
    mixed increases and decreases."""
    data = pl.DataFrame(
        {
            "Time [s]": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "Current [A]": [1.0, 1.0, -1.0, -1.0, 1.0, 1.0],
            "Capacity [A.h]": [5.0, 4.0, 3.0, 4.0, 3.0, 2.0],  # Mixed
        }
    )
    out = iwdata.transform.set_capacity(data)
    assert all(out["Discharge capacity [A.h]"] >= 0)
    assert all(out["Charge capacity [A.h]"] >= 0)
    # Capacity column: [5.0, 4.0, 3.0, 4.0, 3.0, 2.0]
    # Deltas: [5.0, -1.0, -1.0, 1.0, -1.0, -1.0]
    # During discharge (positive current at indices 0-1, 4-5):
    #   Discharge deltas: [5.0, -1.0, 0, 0, -1.0, -1.0]
    #   Discharge cumulative: [5.0, 4.0, 4.0, 4.0, 3.0, 2.0]
    # During charge (negative current at indices 2-3):
    #   Charge deltas: [0, 0, 1.0, 1.0, 0, 0]
    #   Charge cumulative: [0, 0, 1.0, 2.0, 2.0, 2.0]
    # After abs: discharge starts at 5.0 and decreases, charge increases
    assert out["Discharge capacity [A.h]"][0] >= 0.0
    assert out["Discharge capacity [A.h]"][5] < out["Discharge capacity [A.h]"][0]
    assert out["Charge capacity [A.h]"][3] > out["Charge capacity [A.h]"][2]


def test_capacity_step_reset_with_split_columns():
    """Test that step resets work correctly with split capacity columns."""
    # Test case 1: Split columns with incorrect step boundaries
    data1 = pl.DataFrame(
        {
            "Time [s]": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "Current [A]": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "Discharge capacity [A.h]": [0.0, 0.001, 0.002, 0.003, 0.004, 0.005],
            "Charge capacity [A.h]": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "Step count": [0, 0, 1, 1, 2, 2],
        }
    )
    out1 = iwdata.transform.set_capacity(data1)
    # Should reset at step boundaries
    assert out1["Discharge capacity [A.h]"][0] == 0.0
    assert out1["Discharge capacity [A.h]"][2] == 0.0
    assert out1["Discharge capacity [A.h]"][4] == 0.0

    # Test case 2: Split columns already reset correctly
    data2 = pl.DataFrame(
        {
            "Time [s]": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "Current [A]": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "Discharge capacity [A.h]": [0.0, 0.001, 0.0, 0.001, 0.0, 0.001],
            "Charge capacity [A.h]": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "Step count": [0, 0, 1, 1, 2, 2],
        }
    )
    out2 = iwdata.transform.set_capacity(data2)
    # Should preserve correct resets
    assert out2["Discharge capacity [A.h]"][0] == 0.0
    assert out2["Discharge capacity [A.h]"][2] == 0.0
    assert out2["Discharge capacity [A.h]"][4] == 0.0
