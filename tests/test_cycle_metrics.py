import polars as pl
import pytest

import ionworksdata as iwdata


def test_get_cycle_metrics_basic():
    """Test basic cycle metrics calculation."""
    # Create a steps DataFrame with 3 cycles
    steps = pl.DataFrame(
        {
            "Cycle count": [0, 0, 1, 1, 2, 2],
            "Discharge capacity [A.h]": [0.5, 0.5, 0.48, 0.48, 0.46, 0.46],
            "Charge capacity [A.h]": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            "Discharge energy [W.h]": [1.8, 1.8, 1.72, 1.72, 1.65, 1.65],
            "Charge energy [W.h]": [2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
        }
    )
    result = iwdata.cycle_metrics.get_cycle_metrics(steps)

    # Should have 3 rows (one per cycle)
    assert result.height == 3

    # Check column names
    assert "Cycle number" in result.columns
    assert "Discharge capacity [A.h]" in result.columns
    assert "Charge capacity [A.h]" in result.columns
    assert "Coulombic efficiency" in result.columns
    assert "Energy efficiency" in result.columns
    assert "Capacity retention" in result.columns
    assert "Capacity fade" in result.columns
    assert "Energy retention" in result.columns
    assert "Energy fade" in result.columns

    # Check cycle numbers
    assert result["Cycle number"].to_list() == [0, 1, 2]

    # Check summed capacities (each cycle has 2 steps)
    assert result["Discharge capacity [A.h]"].to_list() == [1.0, 0.96, 0.92]
    assert result["Charge capacity [A.h]"].to_list() == [1.0, 1.0, 1.0]

    # Check summed energies (each cycle has 2 steps)
    assert result["Discharge energy [W.h]"].to_list() == [3.6, 3.44, 3.3]
    assert result["Charge energy [W.h]"].to_list() == [4.0, 4.0, 4.0]

    # Check coulombic efficiency (discharge / charge)
    assert result["Coulombic efficiency"][0] == 1.0  # 1.0 / 1.0
    assert result["Coulombic efficiency"][1] == 0.96  # 0.96 / 1.0
    assert result["Coulombic efficiency"][2] == 0.92  # 0.92 / 1.0

    # Check energy efficiency
    assert result["Energy efficiency"][0] == 0.9  # 3.6 / 4.0
    assert abs(result["Energy efficiency"][1] - 0.86) < 0.01  # 3.44 / 4.0
    assert abs(result["Energy efficiency"][2] - 0.825) < 0.01  # 3.3 / 4.0

    # Check capacity retention (relative to first cycle)
    assert result["Capacity retention"][0] == 1.0
    assert result["Capacity retention"][1] == 0.96
    assert result["Capacity retention"][2] == 0.92

    # Check capacity fade (1 - retention)
    assert result["Capacity fade"][0] == 0.0
    assert abs(result["Capacity fade"][1] - 0.04) < 0.001
    assert abs(result["Capacity fade"][2] - 0.08) < 0.001

    # Check energy retention (relative to first cycle)
    assert result["Energy retention"][0] == 1.0
    assert abs(result["Energy retention"][1] - 3.44 / 3.6) < 0.001
    assert abs(result["Energy retention"][2] - 3.3 / 3.6) < 0.001

    # Check energy fade (1 - retention)
    assert result["Energy fade"][0] == 0.0
    assert abs(result["Energy fade"][1] - (1 - 3.44 / 3.6)) < 0.001
    assert abs(result["Energy fade"][2] - (1 - 3.3 / 3.6)) < 0.001


def test_get_cycle_metrics_no_energy():
    """Test cycle metrics when energy columns are not present."""
    steps = pl.DataFrame(
        {
            "Cycle count": [0, 1, 2],
            "Discharge capacity [A.h]": [1.0, 0.98, 0.95],
            "Charge capacity [A.h]": [1.0, 1.0, 1.0],
        }
    )
    result = iwdata.cycle_metrics.get_cycle_metrics(steps)

    # Should not have energy columns or energy metrics
    assert "Discharge energy [W.h]" not in result.columns
    assert "Charge energy [W.h]" not in result.columns
    assert "Energy efficiency" not in result.columns
    assert "Energy retention" not in result.columns
    assert "Energy fade" not in result.columns
    assert "Energy throughput [W.h]" not in result.columns

    # Should still have capacity metrics
    assert "Coulombic efficiency" in result.columns
    assert "Capacity retention" in result.columns
    assert "Capacity fade" in result.columns
    assert "Capacity throughput [A.h]" in result.columns


def test_get_cycle_metrics_zero_charge():
    """Test cycle metrics with zero charge capacity (edge case)."""
    steps = pl.DataFrame(
        {
            "Cycle count": [0, 1],
            "Discharge capacity [A.h]": [1.0, 0.5],
            "Charge capacity [A.h]": [1.0, 0.0],  # Zero charge in cycle 1
        }
    )
    result = iwdata.cycle_metrics.get_cycle_metrics(steps)

    # Coulombic efficiency should be None for cycle with zero charge
    assert result["Coulombic efficiency"][0] == 1.0
    assert result["Coulombic efficiency"][1] is None


def test_get_cycle_metrics_density_units():
    """Test cycle metrics with current density units."""
    steps = pl.DataFrame(
        {
            "Cycle count": [0, 1],
            "Discharge capacity [mA.h.cm-2]": [10.0, 9.5],
            "Charge capacity [mA.h.cm-2]": [10.0, 10.0],
        }
    )
    result = iwdata.cycle_metrics.get_cycle_metrics(
        steps, options={"current units": "density"}
    )

    # Should have density unit columns
    assert "Discharge capacity [mA.h.cm-2]" in result.columns
    assert "Charge capacity [mA.h.cm-2]" in result.columns
    assert "Capacity throughput [mA.h.cm-2]" in result.columns

    # Check metrics (use approximate comparison for floating point)
    assert result["Coulombic efficiency"][0] == 1.0
    assert abs(result["Coulombic efficiency"][1] - 0.95) < 0.001
    assert result["Capacity retention"][0] == 1.0
    assert abs(result["Capacity retention"][1] - 0.95) < 0.001


def test_get_cycle_metrics_missing_columns():
    """Test that missing required columns raise appropriate errors."""
    # Missing Cycle count
    steps1 = pl.DataFrame(
        {
            "Discharge capacity [A.h]": [1.0],
            "Charge capacity [A.h]": [1.0],
        }
    )
    with pytest.raises(ValueError, match="Cycle count"):
        iwdata.cycle_metrics.get_cycle_metrics(steps1)

    # Missing discharge capacity
    steps2 = pl.DataFrame(
        {
            "Cycle count": [0],
            "Charge capacity [A.h]": [1.0],
        }
    )
    with pytest.raises(ValueError, match="Discharge capacity"):
        iwdata.cycle_metrics.get_cycle_metrics(steps2)

    # Missing charge capacity
    steps3 = pl.DataFrame(
        {
            "Cycle count": [0],
            "Discharge capacity [A.h]": [1.0],
        }
    )
    with pytest.raises(ValueError, match="Charge capacity"):
        iwdata.cycle_metrics.get_cycle_metrics(steps3)


def test_get_cycle_metrics_single_cycle():
    """Test cycle metrics with a single cycle."""
    steps = pl.DataFrame(
        {
            "Cycle count": [0, 0, 0],
            "Discharge capacity [A.h]": [0.3, 0.3, 0.4],
            "Charge capacity [A.h]": [0.35, 0.35, 0.3],
        }
    )
    result = iwdata.cycle_metrics.get_cycle_metrics(steps)

    assert result.height == 1
    assert result["Cycle number"][0] == 0
    assert result["Discharge capacity [A.h]"][0] == 1.0
    assert result["Charge capacity [A.h]"][0] == 1.0
    assert result["Coulombic efficiency"][0] == 1.0
    assert result["Capacity retention"][0] == 1.0
    assert result["Capacity fade"][0] == 0.0


def test_get_cycle_metrics_zero_first_cycle():
    """Test cycle metrics when first cycle has zero discharge capacity."""
    steps = pl.DataFrame(
        {
            "Cycle count": [0, 1],
            "Discharge capacity [A.h]": [0.0, 1.0],
            "Charge capacity [A.h]": [1.0, 1.0],
        }
    )
    result = iwdata.cycle_metrics.get_cycle_metrics(steps)

    # Capacity retention/fade should be None when first cycle discharge is zero
    assert result["Capacity retention"][0] is None
    assert result["Capacity retention"][1] is None
    assert result["Capacity fade"][0] is None
    assert result["Capacity fade"][1] is None


def test_get_cycle_metrics_throughput():
    """Test capacity and energy throughput calculations."""
    steps = pl.DataFrame(
        {
            "Cycle count": [0, 1, 2],
            "Discharge capacity [A.h]": [1.0, 0.98, 0.95],
            "Charge capacity [A.h]": [1.0, 1.0, 1.0],
            "Discharge energy [W.h]": [3.6, 3.5, 3.4],
            "Charge energy [W.h]": [4.0, 4.0, 4.0],
        }
    )
    result = iwdata.cycle_metrics.get_cycle_metrics(steps)

    # Check capacity throughput (cumulative sum of discharge + charge)
    assert "Capacity throughput [A.h]" in result.columns
    # Cycle 0: 1.0 + 1.0 = 2.0
    # Cycle 1: 2.0 + 0.98 + 1.0 = 3.98
    # Cycle 2: 3.98 + 0.95 + 1.0 = 5.93
    assert result["Capacity throughput [A.h]"][0] == 2.0
    assert abs(result["Capacity throughput [A.h]"][1] - 3.98) < 0.001
    assert abs(result["Capacity throughput [A.h]"][2] - 5.93) < 0.001

    # Check energy throughput (cumulative sum of discharge + charge energy)
    assert "Energy throughput [W.h]" in result.columns
    # Cycle 0: 3.6 + 4.0 = 7.6
    # Cycle 1: 7.6 + 3.5 + 4.0 = 15.1
    # Cycle 2: 15.1 + 3.4 + 4.0 = 22.5
    assert result["Energy throughput [W.h]"][0] == 7.6
    assert abs(result["Energy throughput [W.h]"][1] - 15.1) < 0.001
    assert abs(result["Energy throughput [W.h]"][2] - 22.5) < 0.001


def test_get_cycle_metrics_start_time():
    """Test start time calculation."""
    steps = pl.DataFrame(
        {
            "Cycle count": [0, 0, 1, 1, 2, 2],
            "Discharge capacity [A.h]": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            "Charge capacity [A.h]": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            "Start time [s]": [0.0, 100.0, 200.0, 300.0, 400.0, 500.0],
        }
    )
    result = iwdata.cycle_metrics.get_cycle_metrics(steps)

    assert "Start time [s]" in result.columns
    # Each cycle's start time is the minimum start time of its steps
    assert result["Start time [s]"][0] == 0.0
    assert result["Start time [s]"][1] == 200.0
    assert result["Start time [s]"][2] == 400.0


def test_get_cycle_metrics_voltage():
    """Test voltage metrics."""
    steps = pl.DataFrame(
        {
            "Cycle count": [0, 0, 1, 1],
            "Discharge capacity [A.h]": [0.5, 0.5, 0.5, 0.5],
            "Charge capacity [A.h]": [0.5, 0.5, 0.5, 0.5],
            "Min voltage [V]": [3.0, 3.2, 2.9, 3.1],
            "Max voltage [V]": [4.1, 4.2, 4.0, 4.15],
        }
    )
    result = iwdata.cycle_metrics.get_cycle_metrics(steps)

    assert "Min voltage [V]" in result.columns
    assert "Max voltage [V]" in result.columns
    # Min is the minimum of all steps in the cycle
    assert result["Min voltage [V]"][0] == 3.0
    assert result["Min voltage [V]"][1] == 2.9
    # Max is the maximum of all steps in the cycle
    assert result["Max voltage [V]"][0] == 4.2
    assert result["Max voltage [V]"][1] == 4.15


def test_get_cycle_metrics_duration():
    """Test duration metrics."""
    steps = pl.DataFrame(
        {
            "Cycle count": [0, 0, 1, 1],
            "Discharge capacity [A.h]": [0.5, 0.5, 0.5, 0.5],
            "Charge capacity [A.h]": [0.5, 0.5, 0.5, 0.5],
            "Duration [s]": [100.0, 200.0, 150.0, 250.0],
        }
    )
    result = iwdata.cycle_metrics.get_cycle_metrics(steps)

    assert "Cycle duration [s]" in result.columns
    # Total duration is sum of all step durations in the cycle
    assert result["Cycle duration [s]"][0] == 300.0
    assert result["Cycle duration [s]"][1] == 400.0


def test_get_cycle_metrics_current_by_sign():
    """Test mean discharge/charge current calculations based on current sign."""
    # Positive current = discharge, negative current = charge
    steps = pl.DataFrame(
        {
            "Cycle count": [0, 0, 1, 1],
            "Discharge capacity [A.h]": [0.5, 0.0, 0.5, 0.0],
            "Charge capacity [A.h]": [0.0, 0.5, 0.0, 0.5],
            "Mean current [A]": [
                1.0,
                -0.5,
                0.8,
                -0.4,
            ],  # Positive=discharge, negative=charge
            "Duration [s]": [100.0, 200.0, 150.0, 250.0],
        }
    )
    result = iwdata.cycle_metrics.get_cycle_metrics(steps)

    assert "Mean discharge current [A]" in result.columns
    assert "Mean charge current [A]" in result.columns
    assert "Discharge duration [s]" in result.columns
    assert "Charge duration [s]" in result.columns

    # Check discharge current (duration-weighted, from positive current steps)
    assert result["Mean discharge current [A]"][0] == 1.0  # Only one discharge step
    assert result["Mean discharge current [A]"][1] == 0.8  # Only one discharge step

    # Check charge current (duration-weighted, from negative current steps)
    assert result["Mean charge current [A]"][0] == -0.5  # Only one charge step
    assert result["Mean charge current [A]"][1] == -0.4  # Only one charge step

    # Check durations
    assert result["Discharge duration [s]"][0] == 100.0
    assert result["Discharge duration [s]"][1] == 150.0
    assert result["Charge duration [s]"][0] == 200.0
    assert result["Charge duration [s]"][1] == 250.0


def test_get_cycle_metrics_temperature():
    """Test mean temperature calculation."""
    steps = pl.DataFrame(
        {
            "Cycle count": [0, 0, 1, 1],
            "Discharge capacity [A.h]": [0.5, 0.5, 0.5, 0.5],
            "Charge capacity [A.h]": [0.5, 0.5, 0.5, 0.5],
            "Mean temperature [degC]": [25.0, 30.0, 28.0, 32.0],
            "Duration [s]": [100.0, 200.0, 150.0, 250.0],
        }
    )
    result = iwdata.cycle_metrics.get_cycle_metrics(steps)

    assert "Mean temperature [degC]" in result.columns
    # Duration-weighted mean: (25*100 + 30*200) / 300 = 8500/300 = 28.33...
    assert abs(result["Mean temperature [degC]"][0] - 28.333) < 0.01
    # Duration-weighted mean: (28*150 + 32*250) / 400 = 12200/400 = 30.5
    assert abs(result["Mean temperature [degC]"][1] - 30.5) < 0.01
