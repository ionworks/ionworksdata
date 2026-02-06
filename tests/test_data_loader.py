from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pybamm
import pytest
from scipy.signal import savgol_filter

import ionworksdata as iwdata
from ionworksdata.load import first_step_from_cycle, last_step_from_cycle


def test_get_item():
    data_loader = iwdata.DataLoader.from_local(
        Path("tests/test_data/cccv-synthetic-with-steps")
    )
    assert data_loader["Voltage [V]"].equals(data_loader.data["Voltage [V]"])


def test_ocp_data_loader():
    data = pd.read_csv(Path("tests/test_data/ocv_synthetic.csv"))
    data_loader = iwdata.OCPDataLoader(data)
    assert isinstance(data_loader, iwdata.OCPDataLoader)
    assert "Voltage [V]" in data_loader.data.columns
    assert "Capacity [A.h]" in data_loader.data.columns
    assert data_loader["Voltage [V]"].equals(data["Voltage [V]"])
    assert data_loader["Capacity [A.h]"].equals(data["Capacity [A.h]"])


def test_load_data():
    data_loader = iwdata.DataLoader.from_local(
        Path("tests/test_data/cccv-synthetic-with-steps"),
        {"first_step": 0, "last_step": 9},
    )
    true_full_data = pd.read_csv(
        Path("tests/test_data/cccv-synthetic-with-steps/time_series.csv")
    )
    assert data_loader.data.columns.tolist() == true_full_data.columns.tolist()
    assert true_full_data.equals(data_loader.data)
    assert data_loader.initial_voltage == true_full_data["Voltage [V]"].iloc[0]
    assert isinstance(data_loader, iwdata.DataLoader)

    data_loader_steps_2_3 = iwdata.DataLoader.from_local(
        data_path=Path("tests/test_data/cccv-synthetic-with-steps"),
        options={
            "first_step": 2,
            "last_step": 3,
        },
    )
    np.testing.assert_allclose(
        data_loader_steps_2_3.data["Time [s]"].iloc[-1]
        - data_loader_steps_2_3.data["Time [s]"].iloc[0],
        data_loader_steps_2_3.steps["Duration [s]"].sum(),
    )
    all_steps = pd.read_csv(Path("tests/test_data/cccv-synthetic-with-steps/steps.csv"))
    assert data_loader_steps_2_3.initial_voltage == all_steps["End voltage [V]"].iloc[1]


def test_plot_data():
    data_loader = iwdata.DataLoader.from_local(
        Path("tests/test_data/cccv-synthetic-with-steps"),
        {"first_step": 0, "last_step": 9},
    )
    fig, ax = data_loader.plot_data()
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax[0], plt.Axes)
    assert isinstance(ax[1], plt.Axes)
    assert isinstance(ax[2], plt.Axes)


def test_generate_experiment():
    data_loader = iwdata.DataLoader.from_local(
        Path("tests/test_data/cccv-synthetic-with-steps"),
        {"first_step": 0, "last_step": 8},
    )
    experiment = data_loader.generate_experiment(use_cv=False)
    assert isinstance(experiment, pybamm.Experiment)
    assert len(experiment.steps) == 9
    assert all(isinstance(step, pybamm.step.Current) for step in experiment.steps)
    experiment = data_loader.generate_experiment(use_cv=True)
    assert isinstance(experiment.steps[4], pybamm.step.Voltage)

    data_loader_steps_2_5 = iwdata.DataLoader.from_local(
        data_path=Path("tests/test_data/cccv-synthetic-with-steps"),
        options={
            "first_step": 2,
            "last_step": 5,
        },
    )
    experiment = data_loader_steps_2_5.generate_experiment(use_cv=False)
    assert isinstance(experiment, pybamm.Experiment)
    assert len(experiment.steps) == 4
    assert all(isinstance(step, pybamm.step.Current) for step in experiment.steps)


def test_generate_interpolant():
    data_loader = iwdata.DataLoader.from_local(
        Path("tests/test_data/cccv-synthetic-with-steps"),
        {"first_step": 0, "last_step": 8},
    )
    interpolant = data_loader.generate_interpolant()
    # Make sure we can discretise and that the interpolant is correct
    model = pybamm.lithium_ion.SPM()
    param = pybamm.ParameterValues("Chen2020")
    param.update({"Current function [A]": interpolant})
    sim = pybamm.Simulation(model, parameter_values=param)
    sol = sim.solve(
        initial_soc=f"{data_loader.initial_voltage} V", t_eval=[0, interpolant.x[0][-1]]
    )
    assert sol["Current [A]"].entries[0] == 0.0
    assert sol["Current [A]"](t=3600 + 1e-9) == 5.0
    # Make sure we are doing something productive (reducing tpts)
    assert len(interpolant.x[0]) == 114
    assert len(data_loader.data["Time [s]"]) == 563


def test_filter_data():
    data_loader_filtered = iwdata.DataLoader.from_local(
        Path("tests/test_data/cccv-synthetic-with-steps"),
        {
            "first_step": 2,
            "last_step": 4,
            "filters": {
                "Voltage [V]": {
                    "filter_type": "savgol",
                    "parameters": {"window_length": 5, "polyorder": 2},
                }
            },
        },
    )
    data_loader_unfiltered = iwdata.DataLoader.from_local(
        Path("tests/test_data/cccv-synthetic-with-steps"),
        {"first_step": 2, "last_step": 4},
    )
    filtered_voltage_manual = savgol_filter(
        data_loader_unfiltered.data["Voltage [V]"], window_length=5, polyorder=2
    )

    # Make sure that the filtered data is filtered and that the filtered data is different from the raw data
    np.testing.assert_allclose(
        filtered_voltage_manual, data_loader_filtered.data["Voltage [V]"]
    )
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(
            data_loader_unfiltered.data["Voltage [V]"],
            data_loader_filtered.data["Voltage [V]"],
        )


def test_interpolate_data():
    data_loader = iwdata.DataLoader.from_local(
        Path("tests/test_data/cccv-synthetic-with-steps"),
        {
            "first_step": 2,
            "last_step": 4,
            "interpolate": 0.1,
        },
    )

    data_loader_raw = iwdata.DataLoader.from_local(
        Path("tests/test_data/cccv-synthetic-with-steps"),
        {"first_step": 2, "last_step": 4},
    )

    np.testing.assert_allclose(
        data_loader.data["Time [s]"],
        np.arange(
            data_loader_raw.data["Time [s]"].min(),
            data_loader_raw.data["Time [s]"].max(),
            0.1,
        ),
    )

    np.testing.assert_allclose(
        data_loader.data["Voltage [V]"],
        np.interp(
            data_loader.data["Time [s]"],
            data_loader_raw.data["Time [s]"],
            data_loader_raw.data["Voltage [V]"],
        ),
    )


def test_get_step():
    data_loader_full = iwdata.DataLoader.from_local(
        Path("tests/test_data/cccv-synthetic-with-steps"), {}
    )
    data_loader = iwdata.DataLoader.from_local(
        Path("tests/test_data/cccv-synthetic-with-steps"),
        {"first_step": 1, "last_step": 2},
    )
    full_data = data_loader_full.data
    full_data = full_data[full_data["Step count"].isin([1, 2])]
    pd.testing.assert_frame_equal(
        data_loader.data.sort_index(axis=1), full_data.sort_index(axis=1)
    )


def test_get_step_from_cycle():
    data_loader_full = iwdata.DataLoader.from_local(
        Path("tests/test_data/cccv-synthetic-with-steps"), {}
    )
    data_loader = iwdata.DataLoader.from_local(
        Path("tests/test_data/cccv-synthetic-with-steps"),
        {
            "first_step": first_step_from_cycle(1),
            "last_step": last_step_from_cycle(1),
        },
    )
    full_data = data_loader_full.data
    full_data = full_data[full_data["Cycle count"] == 1]
    pd.testing.assert_frame_equal(
        data_loader.data.sort_index(axis=1), full_data.sort_index(axis=1)
    )


def test_data_loader_to_config_basic():
    """Test basic DataLoader to_config method."""
    time_series = pd.DataFrame(
        {
            "Time [s]": [0, 1, 2, 3, 4],
            "Voltage [V]": [3.5, 3.6, 3.7, 3.8, 3.9],
            "Current [A]": [1.0, 1.0, 1.0, 1.0, 1.0],
        }
    )
    steps = pd.DataFrame(
        {
            "Start index": [0],
            "End index": [4],
            "Start voltage [V]": [3.5],
        }
    )

    data_loader = iwdata.DataLoader(time_series, steps)
    config = data_loader.to_config()

    assert "data" in config
    assert "time_series" in config["data"]
    assert "steps" in config["data"]
    # Should not have options for basic case
    assert "options" not in config


def test_data_loader_to_config_with_step_filtering():
    """Test DataLoader to_config with step filtering."""
    time_series = pd.DataFrame(
        {
            "Time [s]": [0, 1, 2, 3, 4, 5, 6],
            "Voltage [V]": [3.5, 3.6, 3.7, 3.8, 3.9, 4.0, 4.1],
            "Current [A]": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        }
    )
    steps = pd.DataFrame(
        {
            "Start index": [0, 3, 5],
            "End index": [2, 4, 6],
            "Start voltage [V]": [3.5, 3.8, 4.0],
            "End voltage [V]": [3.7, 3.9, 4.1],
            "Step count": [0, 1, 2],
        }
    )

    data_loader = iwdata.DataLoader(
        time_series,
        steps,
        first_step=1,
        last_step=2,
    )

    # With filter_data=True (default), data is already filtered so no first_step/last_step in options
    config = data_loader.to_config()
    assert "data" in config
    assert "time_series" in config["data"]
    assert "steps" in config["data"]
    # first_step/last_step should NOT be in options since data is already filtered
    assert "options" not in config or "first_step" not in config.get("options", {})

    # With filter_data=False, original data is saved with first_step/last_step in options
    config_unfiltered = data_loader.to_config(filter_data=False)
    assert "data" in config_unfiltered
    assert "options" in config_unfiltered
    assert config_unfiltered["options"]["first_step"] == 1
    assert config_unfiltered["options"]["last_step"] == 2


def test_data_loader_to_config_with_filters():
    """Test DataLoader to_config with filters."""
    time_series = pd.DataFrame(
        {
            "Time [s]": [0, 1, 2, 3, 4],
            "Voltage [V]": [3.5, 3.6, 3.7, 3.8, 3.9],
            "Current [A]": [1.0, 1.0, 1.0, 1.0, 1.0],
        }
    )
    steps = pd.DataFrame(
        {
            "Start index": [0],
            "End index": [4],
            "Start voltage [V]": [3.5],
        }
    )

    filters = {
        "Voltage [V]": {
            "filter_type": "savgol",
            "parameters": {"window_length": 3, "polyorder": 1},
        }
    }

    data_loader = iwdata.DataLoader(time_series, steps, filters=filters)
    config = data_loader.to_config()

    assert "data" in config
    assert "options" in config
    assert config["options"]["filters"] == filters


def test_data_loader_to_config_with_interpolate():
    """Test DataLoader to_config with interpolation."""
    time_series = pd.DataFrame(
        {
            "Time [s]": [0, 1, 2, 3, 4],
            "Voltage [V]": [3.5, 3.6, 3.7, 3.8, 3.9],
            "Current [A]": [1.0, 1.0, 1.0, 1.0, 1.0],
        }
    )
    steps = pd.DataFrame(
        {
            "Start index": [0],
            "End index": [4],
            "Start voltage [V]": [3.5],
        }
    )

    # Test with float interpolate
    data_loader = iwdata.DataLoader(time_series, steps, interpolate=0.5)
    config = data_loader.to_config()

    assert "data" in config
    assert "options" in config
    assert config["options"]["interpolate"] == 0.5

    # Test with numpy array interpolate
    interpolate_array = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    data_loader2 = iwdata.DataLoader(time_series, steps, interpolate=interpolate_array)
    config2 = data_loader2.to_config()

    assert "options" in config2
    assert isinstance(config2["options"]["interpolate"], list)
    assert config2["options"]["interpolate"] == interpolate_array.tolist()


def test_data_loader_roundtrip_filter_data_true():
    """Test DataLoader roundtrip with filter_data=True (default)."""
    time_series = pd.DataFrame(
        {
            "Time [s]": [0, 1, 2, 3, 4, 5],
            "Voltage [V]": [3.5, 3.6, 3.7, 3.8, 3.9, 4.0],
            "Current [A]": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        }
    )
    steps = pd.DataFrame(
        {
            "Start index": [0, 3],
            "End index": [2, 5],
            "Start voltage [V]": [3.5, 3.8],
            "End voltage [V]": [3.7, 4.0],
            "Step count": [0, 1],
        }
    )

    # Create DataLoader with step filtering
    data_loader1 = iwdata.DataLoader(
        time_series,
        steps,
        first_step=1,  # Only step 1
        last_step=1,
    )

    # Convert to config with filter_data=True (default)
    config = data_loader1.to_config(filter_data=True)

    # Config should contain filtered data, no first_step/last_step in options
    assert "data" in config
    assert "time_series" in config["data"]
    assert "steps" in config["data"]
    assert "options" not in config or "first_step" not in config.get("options", {})

    # Create new DataLoader from config
    data_loader2 = iwdata.DataLoader(
        config["data"]["time_series"],
        config["data"]["steps"],
        **config.get("options", {}),
    )

    # Check that both DataLoaders produce same data values
    # (indices will differ: original vs 0-based, so compare values only)
    pd.testing.assert_frame_equal(
        data_loader1.data.reset_index(drop=True),
        data_loader2.data.reset_index(drop=True),
    )
    # For steps, Start/End index will differ but other columns should match
    steps_cols_to_compare = [
        c for c in data_loader1.steps.columns if c not in ["Start index", "End index"]
    ]
    pd.testing.assert_frame_equal(
        data_loader1.steps[steps_cols_to_compare].reset_index(drop=True),
        data_loader2.steps[steps_cols_to_compare].reset_index(drop=True),
        check_dtype=False,
    )


def test_data_loader_roundtrip_filter_data_false():
    """Test DataLoader roundtrip with filter_data=False."""
    time_series = pd.DataFrame(
        {
            "Time [s]": [0, 1, 2, 3, 4, 5],
            "Voltage [V]": [3.5, 3.6, 3.7, 3.8, 3.9, 4.0],
            "Current [A]": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        }
    )
    steps = pd.DataFrame(
        {
            "Start index": [0, 3],
            "End index": [2, 5],
            "Start voltage [V]": [3.5, 3.8],
            "End voltage [V]": [3.7, 4.0],
            "Step count": [0, 1],
        }
    )

    # Create DataLoader with step filtering
    data_loader1 = iwdata.DataLoader(
        time_series,
        steps,
        first_step=1,  # Only step 1
        last_step=1,
    )

    # Convert to config with filter_data=False
    config = data_loader1.to_config(filter_data=False)

    # Config should contain original unfiltered data with first_step/last_step in options
    assert "data" in config
    assert "time_series" in config["data"]
    assert "steps" in config["data"]
    assert "options" in config
    assert config["options"]["first_step"] == 1
    assert config["options"]["last_step"] == 1

    # Create new DataLoader from config - should re-filter to same result
    data_loader2 = iwdata.DataLoader(
        config["data"]["time_series"],
        config["data"]["steps"],
        **config.get("options", {}),
    )

    # Check that both DataLoaders produce same data
    pd.testing.assert_frame_equal(data_loader1.data, data_loader2.data)
    pd.testing.assert_frame_equal(data_loader1.steps, data_loader2.steps)


def test_ocp_data_loader_to_config_basic():
    """Test basic OCPDataLoader to_config method."""
    ocp_data = pd.DataFrame(
        {
            "Capacity [A.h]": [0.0, 0.1, 0.2, 0.3],
            "Voltage [V]": [3.5, 3.6, 3.7, 3.8],
        }
    )

    data_loader = iwdata.OCPDataLoader(ocp_data)
    config = data_loader.to_config()

    assert "data" in config
    assert "options" not in config


def test_ocp_data_loader_to_config_with_filters():
    """Test OCPDataLoader to_config with filters."""
    ocp_data = pd.DataFrame(
        {
            "Capacity [A.h]": [0.0, 0.1, 0.2, 0.3, 0.4],
            "Voltage [V]": [3.5, 3.6, 3.7, 3.8, 3.9],
        }
    )

    filters = {
        "Voltage [V]": {
            "filter_type": "savgol",
            "parameters": {"window_length": 3, "polyorder": 1},
        }
    }

    data_loader = iwdata.OCPDataLoader(ocp_data, filters=filters)
    config = data_loader.to_config()

    assert "data" in config
    assert "options" in config
    assert config["options"]["filters"] == filters


def test_ocp_data_loader_to_config_all_args():
    """Test OCPDataLoader to_config with both filters and interpolate."""
    ocp_data = pd.DataFrame(
        {
            "Capacity [A.h]": [0.0, 0.1, 0.2, 0.3, 0.4],
            "Voltage [V]": [3.5, 3.6, 3.7, 3.8, 3.9],
        }
    )

    filters = {
        "Voltage [V]": {
            "filter_type": "savgol",
            "parameters": {"window_length": 3, "polyorder": 1},
        }
    }

    data_loader = iwdata.OCPDataLoader(ocp_data, filters=filters)
    config = data_loader.to_config()

    assert "options" in config
    assert config["options"]["filters"] == filters


def test_ocp_data_loader_roundtrip():
    """Test OCPDataLoader config -> OCPDataLoader -> config roundtrip."""
    ocp_data = pd.DataFrame(
        {
            "Capacity [A.h]": [0.0, 0.1, 0.2, 0.3, 0.4],
            "Voltage [V]": [3.5, 3.6, 3.7, 3.8, 3.9],
        }
    )

    # Create OCPDataLoader with filters
    data_loader1 = iwdata.OCPDataLoader(
        ocp_data,
        filters={
            "Voltage [V]": {
                "filter_type": "savgol",
                "parameters": {"window_length": 3, "polyorder": 1},
            }
        },
    )

    # Convert to config
    config = data_loader1.to_config()

    # Create new OCPDataLoader from config
    data_loader2 = iwdata.OCPDataLoader(config["data"], **config.get("options", {}))

    # Check that both OCPDataLoaders produce same data
    pd.testing.assert_frame_equal(data_loader1.data, data_loader2.data)


def test_ocp_data_loader_copy():
    """Test OCPDataLoader copy method."""
    data = pd.read_csv(Path("tests/test_data/ocv_synthetic.csv"))
    original = iwdata.OCPDataLoader(data)
    copy = original.copy()

    # Verify it's a different instance
    assert copy is not original
    assert isinstance(copy, iwdata.OCPDataLoader)

    # Verify data is copied
    pd.testing.assert_frame_equal(original.data, copy.data)
    assert copy.data is not original.data

    # Verify all columns are preserved
    assert list(original.data.columns) == list(copy.data.columns)
    for col in original.data.columns:
        pd.testing.assert_series_equal(original.data[col], copy.data[col])


def test_data_loader_copy():
    """Test DataLoader copy method."""
    original = iwdata.DataLoader.from_local(
        Path("tests/test_data/cccv-synthetic-with-steps"),
        {"first_step": 2, "last_step": 4},
    )
    copy = original.copy()

    # Verify it's a different instance
    assert copy is not original
    assert isinstance(copy, iwdata.DataLoader)

    # Verify data is copied
    pd.testing.assert_frame_equal(original.data, copy.data)
    assert copy.data is not original.data

    # Verify steps are copied
    pd.testing.assert_frame_equal(original.steps, copy.steps)
    assert copy.steps is not original.steps

    # Verify additional attributes are copied
    assert copy.initial_voltage == original.initial_voltage


def test_copy_independence():
    """Test that modifications to copy don't affect original."""
    # Test with DataLoader
    original_dl = iwdata.DataLoader.from_local(
        Path("tests/test_data/cccv-synthetic-with-steps"),
        {"first_step": 2, "last_step": 4},
    )
    copy_dl = original_dl.copy()

    # Modify copy - use the first valid index
    first_idx = copy_dl.data.index[0]
    copy_dl.data.loc[first_idx, "Voltage [V]"] = 999.0
    copy_dl.initial_voltage = 999.0

    # Verify original is unchanged
    assert original_dl.data.loc[first_idx, "Voltage [V]"] != 999.0
    assert original_dl.initial_voltage != 999.0


def test_copy_with_filters_and_interpolation():
    """Test copy method with filters and interpolation applied."""
    # Test with filters
    filters = {
        "Voltage [V]": {
            "filter_type": "savgol",
            "parameters": {"window_length": 5, "polyorder": 2},
        }
    }

    original = iwdata.DataLoader.from_local(
        Path("tests/test_data/cccv-synthetic-with-steps"),
        {
            "first_step": 2,
            "last_step": 4,
            "filters": filters,
        },
    )
    copy = original.copy()

    # Verify filtered data is preserved
    pd.testing.assert_frame_equal(original.data, copy.data)

    # Test with interpolation
    original_interp = iwdata.DataLoader.from_local(
        Path("tests/test_data/cccv-synthetic-with-steps"),
        {
            "first_step": 2,
            "last_step": 4,
            "interpolate": 0.1,
        },
    )
    copy_interp = original_interp.copy()

    # Verify interpolated data is preserved
    pd.testing.assert_frame_equal(original_interp.data, copy_interp.data)

    # Verify time points are interpolated
    time_diff = np.diff(original_interp.data["Time [s]"])
    assert np.allclose(time_diff, 0.1, atol=1e-10)


def test_copy_preserves_methods():
    """Test that copied instances preserve all methods and functionality."""
    original = iwdata.DataLoader.from_local(
        Path("tests/test_data/cccv-synthetic-with-steps"),
        {"first_step": 0, "last_step": 8},
    )
    copy = original.copy()

    # Test that methods work on copy
    copy_experiment = copy.generate_experiment()
    original_experiment = original.generate_experiment()

    assert isinstance(copy_experiment, pybamm.Experiment)
    assert len(copy_experiment.steps) == len(original_experiment.steps)

    # Test interpolant generation
    copy_interpolant = copy.generate_interpolant()
    original_interpolant = original.generate_interpolant()

    assert isinstance(copy_interpolant, pybamm.Interpolant)
    np.testing.assert_array_equal(copy_interpolant.x[0], original_interpolant.x[0])
    np.testing.assert_array_equal(copy_interpolant.y[0], original_interpolant.y[0])

    # Test plotting
    copy_fig, copy_ax = copy.plot_data()
    original_fig, original_ax = original.plot_data()

    assert isinstance(copy_fig, plt.Figure)
    assert len(copy_ax) == 3


def test_get_step_with_sql_query():
    """Test that SQL query strings work correctly."""
    # Test with first_step_from_cycle
    data_loader = iwdata.DataLoader.from_local(
        Path("tests/test_data/cccv-synthetic-with-steps"),
        {"first_step": first_step_from_cycle(1), "last_step": last_step_from_cycle(1)},
    )

    # Verify we get cycle 1 data
    assert data_loader.steps["Cycle count"].iloc[0] == 1
    assert data_loader.steps["Cycle count"].iloc[-1] == 1

    # Test with integer step indices
    data_loader_int = iwdata.DataLoader.from_local(
        Path("tests/test_data/cccv-synthetic-with-steps"),
        {"first_step": 2, "last_step": 4},
    )

    # Verify correct steps are loaded
    assert data_loader_int.steps["Step count"].iloc[0] == 2
    assert data_loader_int.steps["Step count"].iloc[-1] == 4


def test_get_step_with_explicit_sql():
    """Test that explicit SQL queries work correctly."""
    # Test with explicit SQL query for step 3
    data_loader = iwdata.DataLoader.from_local(
        Path("tests/test_data/cccv-synthetic-with-steps"),
        {
            "first_step": 'SELECT * FROM steps WHERE "Step count" = 3 LIMIT 1',
            "last_step": 5,
        },
    )

    # Verify we get step 3 as the first step
    assert data_loader.steps["Step count"].iloc[0] == 3
    assert data_loader.steps["Step count"].iloc[-1] == 5

    # Test with explicit SQL query for cycle 0
    data_loader_cycle = iwdata.DataLoader.from_local(
        Path("tests/test_data/cccv-synthetic-with-steps"),
        {
            "first_step": 'SELECT * FROM steps WHERE "Cycle count" = 0 ORDER BY "Step count" LIMIT 1',
            "last_step": 'SELECT * FROM steps WHERE "Cycle count" = 0 ORDER BY "Step count" DESC LIMIT 1',
        },
    )

    # Verify we get cycle 0 data
    assert data_loader_cycle.steps["Cycle count"].iloc[0] == 0
    assert data_loader_cycle.steps["Cycle count"].iloc[-1] == 0


def test_get_step_with_integer():
    """Test that integer inputs work correctly."""
    data_loader = iwdata.DataLoader.from_local(
        Path("tests/test_data/cccv-synthetic-with-steps"),
        {"first_step": 2, "last_step": 4},
    )

    # Compare with existing dict-based approach
    data_loader_dict = iwdata.DataLoader.from_local(
        Path("tests/test_data/cccv-synthetic-with-steps"),
        {"first_step": 2, "last_step": 4},
    )

    # Should produce identical results
    pd.testing.assert_frame_equal(
        data_loader.data.sort_index(axis=1), data_loader_dict.data.sort_index(axis=1)
    )
    pd.testing.assert_frame_equal(
        data_loader.steps.sort_index(axis=1), data_loader_dict.steps.sort_index(axis=1)
    )


def test_get_step_dict_deprecation_warning():
    """Test that dict format triggers deprecation warning."""
    with pytest.warns(DeprecationWarning, match="ionworkspipeline"):
        iwdata.DataLoader.from_local(
            Path("tests/test_data/cccv-synthetic-with-steps"),
            {"first_step": {"step": 2}, "last_step": {"step": 4}},
        )


def test_get_step_sql_validation():
    """Test SQL query validation."""
    # Test query returning 0 rows
    with pytest.raises(ValueError, match="SQL query returned no results"):
        iwdata.DataLoader.from_local(
            Path("tests/test_data/cccv-synthetic-with-steps"),
            {"first_step": 'SELECT * FROM steps WHERE "Cycle count" = 999 LIMIT 1'},
        )

    # Test query returning >1 rows
    with pytest.raises(
        ValueError, match="SQL query returned.*results, expected exactly 1"
    ):
        iwdata.DataLoader.from_local(
            Path("tests/test_data/cccv-synthetic-with-steps"),
            {"first_step": 'SELECT * FROM steps WHERE "Cycle count" = 0'},
        )

    # Test invalid SQL syntax
    with pytest.raises(ValueError, match="Error executing SQL query"):
        iwdata.DataLoader.from_local(
            Path("tests/test_data/cccv-synthetic-with-steps"),
            {"first_step": "INVALID SQL SYNTAX"},
        )


def test_parameter_conflicts():
    """Test that providing both old and new parameters raises error."""
    with pytest.raises(
        ValueError, match="Cannot specify both first_step and first_step_dict"
    ):
        iwdata.DataLoader.from_local(
            Path("tests/test_data/cccv-synthetic-with-steps"),
            {"first_step": 2, "first_step_dict": {"step": 2}},
        )

    with pytest.raises(
        ValueError, match="Cannot specify both last_step and last_step_dict"
    ):
        iwdata.DataLoader.from_local(
            Path("tests/test_data/cccv-synthetic-with-steps"),
            {"last_step": 4, "last_step_dict": {"step": 4}},
        )


def test_dataloader_with_polars_input():
    """Test that DataLoader accepts Polars DataFrames as input."""
    import polars as pl

    # Load data with pandas
    time_series_pd = pd.read_csv(
        Path("tests/test_data/cccv-synthetic-with-steps") / "time_series.csv"
    )
    steps_pd = pd.read_csv(
        Path("tests/test_data/cccv-synthetic-with-steps") / "steps.csv"
    )

    # Convert to Polars
    time_series_pl = pl.from_pandas(time_series_pd)
    steps_pl = pl.from_pandas(steps_pd)

    # Create DataLoader with Polars input
    loader_polars = iwdata.DataLoader(time_series_pl, steps_pl)
    loader_pandas = iwdata.DataLoader(time_series_pd, steps_pd)

    # Results should be identical (both use pandas internally)
    pd.testing.assert_frame_equal(
        loader_polars.data.sort_index(axis=1), loader_pandas.data.sort_index(axis=1)
    )
    pd.testing.assert_frame_equal(
        loader_polars.steps.sort_index(axis=1), loader_pandas.steps.sort_index(axis=1)
    )


def test_dataloader_with_mixed_input():
    """Test that DataLoader accepts mixed Pandas/Polars input."""
    import polars as pl

    time_series_pd = pd.read_csv(
        Path("tests/test_data/cccv-synthetic-with-steps") / "time_series.csv"
    )
    steps_pd = pd.read_csv(
        Path("tests/test_data/cccv-synthetic-with-steps") / "steps.csv"
    )

    # Test Polars time_series + Pandas steps
    time_series_pl = pl.from_pandas(time_series_pd)
    loader1 = iwdata.DataLoader(time_series_pl, steps_pd)

    # Test Pandas time_series + Polars steps
    steps_pl = pl.from_pandas(steps_pd)
    loader2 = iwdata.DataLoader(time_series_pd, steps_pl)

    # Both should work and produce same results
    pd.testing.assert_frame_equal(
        loader1.data.sort_index(axis=1), loader2.data.sort_index(axis=1)
    )


def test_from_local_with_polars():
    """Test from_local with use_polars=True."""
    # Load with pandas (default)
    loader_pandas = iwdata.DataLoader.from_local(
        Path("tests/test_data/cccv-synthetic-with-steps")
    )

    # Load with polars
    loader_polars = iwdata.DataLoader.from_local(
        Path("tests/test_data/cccv-synthetic-with-steps"), use_polars=True
    )

    # Results should be similar (column names might differ slightly for unnamed columns)
    # Check that we have the same number of rows and columns
    assert loader_pandas.data.shape == loader_polars.data.shape
    assert loader_pandas.steps.shape == loader_polars.steps.shape

    # Check that key columns have the same values
    for col in ["Time [s]", "Voltage [V]", "Current [A]"]:
        pd.testing.assert_series_equal(
            loader_pandas.data[col].reset_index(drop=True),
            loader_polars.data[col].reset_index(drop=True),
            check_names=False,
        )


def test_ocp_dataloader_with_polars():
    """Test OCPDataLoader with Polars input."""
    import polars as pl

    # Create test OCP data
    data_pd = pd.DataFrame(
        {
            "Capacity [A.h]": [0.0, 0.5, 1.0, 1.5, 2.0],
            "Voltage [V]": [3.0, 3.2, 3.4, 3.6, 3.8],
        }
    )
    data_pl = pl.from_pandas(data_pd)

    # Create loaders with both types
    loader_pandas = iwdata.OCPDataLoader(data_pd)
    loader_polars = iwdata.OCPDataLoader(data_pl)

    # Results should be identical
    pd.testing.assert_frame_equal(loader_pandas.data, loader_polars.data)


def test_generic_dataloader_with_polars():
    """Test GenericDataLoader with Polars input."""
    import polars as pl
    from ionworksdata.load import GenericDataLoader

    data_pd = pd.DataFrame(
        {
            "Time [s]": [0, 1, 2, 3, 4],
            "Voltage [V]": [3.0, 3.1, 3.2, 3.3, 3.4],
            "Current [A]": [1.0, 1.0, 1.0, 0.0, 0.0],
        }
    )
    data_pl = pl.from_pandas(data_pd)

    # Create loaders with both types
    loader_pandas = GenericDataLoader(data_pd)
    loader_polars = GenericDataLoader(data_pl)

    # Results should be identical
    pd.testing.assert_frame_equal(loader_pandas.data, loader_polars.data)


def test_dataloader_polars_with_step_selection():
    """Test that Polars input works with step selection."""
    import polars as pl

    time_series_pd = pd.read_csv(
        Path("tests/test_data/cccv-synthetic-with-steps") / "time_series.csv"
    )
    steps_pd = pd.read_csv(
        Path("tests/test_data/cccv-synthetic-with-steps") / "steps.csv"
    )

    time_series_pl = pl.from_pandas(time_series_pd)
    steps_pl = pl.from_pandas(steps_pd)

    # Test with step selection
    loader_polars = iwdata.DataLoader(
        time_series_pl, steps_pl, first_step=1, last_step=3
    )
    loader_pandas = iwdata.DataLoader(
        time_series_pd, steps_pd, first_step=1, last_step=3
    )

    # Results should be identical
    pd.testing.assert_frame_equal(
        loader_polars.data.sort_index(axis=1), loader_pandas.data.sort_index(axis=1)
    )


def test_dataloader_polars_with_sql_query():
    """Test that Polars input works with SQL queries for step selection."""
    import polars as pl

    time_series_pd = pd.read_csv(
        Path("tests/test_data/cccv-synthetic-with-steps") / "time_series.csv"
    )
    steps_pd = pd.read_csv(
        Path("tests/test_data/cccv-synthetic-with-steps") / "steps.csv"
    )

    time_series_pl = pl.from_pandas(time_series_pd)
    steps_pl = pl.from_pandas(steps_pd)

    # Test with SQL query
    first_query = 'SELECT * FROM steps WHERE "Step count" = 1'
    last_query = 'SELECT * FROM steps WHERE "Step count" = 3'

    loader_polars = iwdata.DataLoader(
        time_series_pl, steps_pl, first_step=first_query, last_step=last_query
    )
    loader_pandas = iwdata.DataLoader(
        time_series_pd, steps_pd, first_step=first_query, last_step=last_query
    )

    # Results should be identical
    pd.testing.assert_frame_equal(
        loader_polars.data.sort_index(axis=1), loader_pandas.data.sort_index(axis=1)
    )


def test_dataloader_from_db():
    """Test DataLoader.from_db loads data from ionworks-api."""
    from unittest.mock import MagicMock, patch

    time_series = pd.DataFrame(
        {
            "Time [s]": [0, 1, 2, 3],
            "Voltage [V]": [3.8, 3.7, 3.6, 3.5],
            "Current [A]": [1.0, 1.0, 1.0, 1.0],
        }
    )
    steps = pd.DataFrame(
        {
            "Start index": [0],
            "End index": [3],
            "Start voltage [V]": [3.8],
        }
    )

    mock_measurement_detail = MagicMock()
    mock_measurement_detail.time_series = time_series
    mock_measurement_detail.steps = steps

    mock_client = MagicMock()
    mock_client.cell_measurement.detail.return_value = mock_measurement_detail

    with patch("ionworks.Ionworks", return_value=mock_client):
        data_loader = iwdata.DataLoader.from_db(
            "test-measurement-id-123", use_cache=False
        )

    mock_client.cell_measurement.detail.assert_called_once_with(
        "test-measurement-id-123"
    )
    assert isinstance(data_loader, iwdata.DataLoader)
    assert data_loader._measurement_id == "test-measurement-id-123"  # noqa: SLF001


def test_ocp_dataloader_from_db():
    """Test OCPDataLoader.from_db loads data from ionworks-api."""
    from unittest.mock import MagicMock, patch

    time_series = pd.DataFrame(
        {
            "Capacity [A.h]": [0.0, 0.5, 1.0, 1.5],
            "Voltage [V]": [4.2, 3.9, 3.7, 3.5],
        }
    )

    mock_measurement_detail = MagicMock()
    mock_measurement_detail.time_series = time_series

    mock_client = MagicMock()
    mock_client.cell_measurement.detail.return_value = mock_measurement_detail

    with patch("ionworks.Ionworks", return_value=mock_client):
        data_loader = iwdata.OCPDataLoader.from_db("test-ocp-id-456", use_cache=False)

    mock_client.cell_measurement.detail.assert_called_once_with("test-ocp-id-456")
    assert isinstance(data_loader, iwdata.OCPDataLoader)
    assert data_loader._measurement_id == "test-ocp-id-456"  # noqa: SLF001


def test_data_loader_to_config_with_db():
    """Test DataLoader to_config returns db format when loaded from database."""
    from unittest.mock import MagicMock, patch

    time_series = pd.DataFrame(
        {
            "Time [s]": [0, 1, 2, 3],
            "Voltage [V]": [3.8, 3.7, 3.6, 3.5],
            "Current [A]": [1.0, 1.0, 1.0, 1.0],
        }
    )
    steps = pd.DataFrame(
        {
            "Start index": [0],
            "End index": [3],
            "Start voltage [V]": [3.8],
        }
    )

    mock_measurement_detail = MagicMock()
    mock_measurement_detail.time_series = time_series
    mock_measurement_detail.steps = steps

    mock_client = MagicMock()
    mock_client.cell_measurement.detail.return_value = mock_measurement_detail

    with patch("ionworks.Ionworks", return_value=mock_client):
        data_loader = iwdata.DataLoader.from_db(
            "test-measurement-id-123", use_cache=False
        )

    config = data_loader.to_config()

    # Should have data key with db: prefix
    assert "data" in config
    assert config["data"] == "db:test-measurement-id-123"
    # Should not have options when no filters/interpolate
    assert "options" not in config


def test_data_loader_to_config_with_db_and_options():
    """Test DataLoader to_config with db format includes options."""
    from unittest.mock import MagicMock, patch

    time_series = pd.DataFrame(
        {
            "Time [s]": [0, 1, 2, 3, 4, 5, 6],
            "Voltage [V]": [3.8, 3.7, 3.6, 3.5, 3.4, 3.3, 3.2],
            "Current [A]": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        }
    )
    steps = pd.DataFrame(
        {
            "Start index": [0, 3],
            "End index": [2, 6],
            "Start voltage [V]": [3.8, 3.5],
            "End voltage [V]": [3.6, 3.2],
            "Step count": [0, 1],
        }
    )

    mock_measurement_detail = MagicMock()
    mock_measurement_detail.time_series = time_series
    mock_measurement_detail.steps = steps

    mock_client = MagicMock()
    mock_client.cell_measurement.detail.return_value = mock_measurement_detail

    with patch("ionworks.Ionworks", return_value=mock_client):
        data_loader = iwdata.DataLoader.from_db(
            "test-measurement-id-456",
            options={"first_step": 0, "last_step": 1},
            use_cache=False,
        )

    config = data_loader.to_config()

    # Should have data key with db: prefix
    assert "data" in config
    assert config["data"] == "db:test-measurement-id-456"
    # Should have options with step filtering
    assert "options" in config
    assert config["options"]["first_step"] == 0
    assert config["options"]["last_step"] == 1


def test_ocp_data_loader_to_config_with_db():
    """Test OCPDataLoader to_config returns db format when loaded from database."""
    from unittest.mock import MagicMock, patch

    time_series = pd.DataFrame(
        {
            "Capacity [A.h]": [0.0, 0.5, 1.0, 1.5],
            "Voltage [V]": [4.2, 3.9, 3.7, 3.5],
        }
    )

    mock_measurement_detail = MagicMock()
    mock_measurement_detail.time_series = time_series

    mock_client = MagicMock()
    mock_client.cell_measurement.detail.return_value = mock_measurement_detail

    with patch("ionworks.Ionworks", return_value=mock_client):
        data_loader = iwdata.OCPDataLoader.from_db("test-ocp-id-789", use_cache=False)

    config = data_loader.to_config()

    # Should have data key with db: prefix
    assert "data" in config
    assert config["data"] == "db:test-ocp-id-789"
    # Should not have options when no filters/interpolate
    assert "options" not in config


def test_ocp_data_loader_to_config_with_db_and_options():
    """Test OCPDataLoader to_config with db format includes options."""
    from unittest.mock import MagicMock, patch

    time_series = pd.DataFrame(
        {
            "Capacity [A.h]": [0.0, 0.5, 1.0, 1.5, 2.0],
            "Voltage [V]": [4.2, 3.9, 3.7, 3.5, 3.3],
        }
    )

    mock_measurement_detail = MagicMock()
    mock_measurement_detail.time_series = time_series

    mock_client = MagicMock()
    mock_client.cell_measurement.detail.return_value = mock_measurement_detail

    filters = {
        "Voltage [V]": {
            "filter_type": "savgol",
            "parameters": {"window_length": 3, "polyorder": 1},
        }
    }

    with patch("ionworks.Ionworks", return_value=mock_client):
        data_loader = iwdata.OCPDataLoader.from_db(
            "test-ocp-id-with-filters",
            options={"filters": filters},
            use_cache=False,
        )

    config = data_loader.to_config()

    # Should have data key with db: prefix
    assert "data" in config
    assert config["data"] == "db:test-ocp-id-with-filters"
    # Should have options with filters
    assert "options" in config
    assert config["options"]["filters"] == filters


def test_ocp_data_loader_capacity_column_explicit():
    """Test explicit capacity_column option."""
    data = pd.DataFrame(
        {
            "Voltage [V]": [4.0, 3.5, 3.0, 2.5],
            "Discharge capacity [A.h]": [0.0, 0.1, 0.2, 0.3],
        }
    )
    loader = iwdata.OCPDataLoader(
        data, options={"capacity_column": "Discharge capacity [A.h]"}
    )
    assert "Capacity [A.h]" in loader.data.columns
    np.testing.assert_array_almost_equal(
        loader.data["Capacity [A.h]"].values, [0.0, 0.1, 0.2, 0.3]
    )


def test_ocp_data_loader_capacity_column_auto_detect():
    """Test auto-detection of capacity column."""
    data = pd.DataFrame(
        {
            "Voltage [V]": [4.0, 3.5, 3.0, 2.5],
            "Discharge capacity [A.h]": [0.0, 0.1, 0.2, 0.3],
        }
    )
    loader = iwdata.OCPDataLoader(data)
    # Should auto-detect "Discharge capacity [A.h]" and alias it
    assert "Capacity [A.h]" in loader.data.columns


def test_ocp_data_loader_sort_option():
    """Test sort option for OCP data."""
    # Data with increasing voltage (wrong order for OCP)
    data = pd.DataFrame(
        {
            "Voltage [V]": [2.5, 3.0, 3.5, 4.0],
            "Capacity [A.h]": [0.3, 0.2, 0.1, 0.0],
        }
    )
    loader = iwdata.OCPDataLoader(data, options={"sort": True})
    # After sorting, voltage should decrease
    assert loader.data["Voltage [V]"].iloc[0] > loader.data["Voltage [V]"].iloc[-1]
    # Capacity should increase
    assert (
        loader.data["Capacity [A.h]"].iloc[0] < loader.data["Capacity [A.h]"].iloc[-1]
    )


def test_ocp_data_loader_remove_duplicates_option():
    """Test remove_duplicates option for OCP data."""
    data = pd.DataFrame(
        {
            "Voltage [V]": [4.0, 3.5, 3.5, 3.0],  # Duplicate voltage
            "Capacity [A.h]": [0.0, 0.1, 0.15, 0.2],
        }
    )
    loader = iwdata.OCPDataLoader(data, options={"remove_duplicates": True})
    # Should have removed the duplicate
    assert len(loader.data) == 3


def test_ocp_data_loader_calculate_dUdQ_cutoff():
    """Test calculate_dUdQ_cutoff method."""
    # Create data with a smooth OCP curve
    q = np.linspace(0, 1, 100)
    U = 4.0 - 0.5 * q + 0.1 * np.sin(10 * q)  # Some variation
    data = pd.DataFrame({"Capacity [A.h]": q, "Voltage [V]": U})
    loader = iwdata.OCPDataLoader(data)
    cutoff = loader.calculate_dUdQ_cutoff()
    assert isinstance(cutoff, float)
    assert cutoff > 0


def test_ocp_data_loader_calculate_dQdU_cutoff():
    """Test calculate_dQdU_cutoff method."""
    # Create data with a smooth OCP curve
    q = np.linspace(0, 1, 100)
    U = 4.0 - 0.5 * q + 0.1 * np.sin(10 * q)  # Some variation
    data = pd.DataFrame({"Capacity [A.h]": q, "Voltage [V]": U})
    loader = iwdata.OCPDataLoader(data)
    cutoff = loader.calculate_dQdU_cutoff()
    assert isinstance(cutoff, float)
    assert cutoff > 0


def test_ocp_data_loader_to_config_includes_preprocessing_options():
    """Test that to_config includes preprocessing options."""
    data = pd.DataFrame(
        {
            "Voltage [V]": [4.0, 3.5, 3.0, 2.5],
            "Discharge capacity [A.h]": [0.0, 0.1, 0.2, 0.3],
        }
    )
    loader = iwdata.OCPDataLoader(
        data,
        options={
            "capacity_column": "Discharge capacity [A.h]",
            "sort": True,
            "remove_duplicates": True,
        },
    )
    config = loader.to_config()
    assert "options" in config
    assert config["options"]["capacity_column"] == "Discharge capacity [A.h]"
    assert config["options"]["sort"] is True
    assert config["options"]["remove_duplicates"] is True


# =============================================================================
# Cache functionality tests
# =============================================================================


def test_cache_enabled_disabled(isolated_cache):
    """Test cache enable/disable functionality."""
    from ionworksdata.load import _CACHE_CONFIG

    # Default should be enabled (set by fixture)
    assert _CACHE_CONFIG["enabled"] is True

    # Disable cache
    iwdata.set_cache_enabled(False)
    assert _CACHE_CONFIG["enabled"] is False

    # Re-enable cache
    iwdata.set_cache_enabled(True)
    assert _CACHE_CONFIG["enabled"] is True


def test_cache_directory_configuration(isolated_cache):
    """Test cache directory configuration functions."""
    # Set custom directory
    custom_dir = Path("/tmp/custom_cache_dir")
    iwdata.set_cache_directory(custom_dir)
    assert iwdata.get_cache_directory() == custom_dir

    # Set directory as string
    iwdata.set_cache_directory("/tmp/another_cache_dir")
    assert iwdata.get_cache_directory() == Path("/tmp/another_cache_dir")


def test_cache_ttl_configuration(isolated_cache):
    """Test cache TTL configuration functions."""
    # Default TTL from fixture is 3600
    assert iwdata.get_cache_ttl() == 3600

    # Set custom TTL
    iwdata.set_cache_ttl(7200)
    assert iwdata.get_cache_ttl() == 7200

    # Set TTL to None (disable expiration)
    iwdata.set_cache_ttl(None)
    assert iwdata.get_cache_ttl() is None

    # Set back to a value
    iwdata.set_cache_ttl(1800)
    assert iwdata.get_cache_ttl() == 1800


def test_cache_ttl_expiration(isolated_cache):
    """Test that cache expires based on TTL."""
    import time

    from ionworksdata.load import (
        _CACHE_CONFIG,
        _get_cache_path,
        _load_from_cache,
        _save_to_cache,
    )

    _CACHE_CONFIG["ttl_seconds"] = 1  # 1 second TTL for testing

    test_data = {"time_series": {"a": [1, 2, 3]}, "steps": {"b": [4, 5, 6]}}
    measurement_id = "test-ttl-measurement"

    # Save to cache
    _save_to_cache(measurement_id, test_data)

    # Should load immediately
    cached = _load_from_cache(measurement_id)
    assert cached is not None
    assert cached["time_series"] == test_data["time_series"]

    # Wait for TTL to expire
    time.sleep(1.5)

    # Should return None after expiration
    cached_expired = _load_from_cache(measurement_id)
    assert cached_expired is None

    # Cache file should be deleted
    cache_path = _get_cache_path(measurement_id)
    assert not cache_path.exists()


def test_cache_ttl_disabled(isolated_cache):
    """Test that cache never expires when TTL is None."""
    import time

    from ionworksdata.load import (
        _CACHE_CONFIG,
        _load_from_cache,
        _save_to_cache,
    )

    _CACHE_CONFIG["ttl_seconds"] = None  # Disable TTL

    test_data = {"time_series": {"a": [1, 2, 3]}}
    measurement_id = "test-no-ttl-measurement"

    # Save to cache
    _save_to_cache(measurement_id, test_data)

    # Wait a bit
    time.sleep(0.1)

    # Should still load (no expiration)
    cached = _load_from_cache(measurement_id)
    assert cached is not None
    assert cached["time_series"] == test_data["time_series"]


def test_cache_save_and_load(isolated_cache):
    """Test basic cache save and load functionality."""
    from ionworksdata.load import (
        _get_cache_path,
        _load_from_cache,
        _save_to_cache,
    )

    # Test data
    test_data = {
        "time_series": pd.DataFrame(
            {"Time [s]": [0, 1, 2], "Voltage [V]": [3.5, 3.6, 3.7]}
        ),
        "steps": pd.DataFrame({"Start index": [0], "End index": [2]}),
    }
    measurement_id = "test-save-load-123"

    # Initially no cache
    assert _load_from_cache(measurement_id) is None

    # Save to cache
    _save_to_cache(measurement_id, test_data)

    # Cache file should exist
    cache_path = _get_cache_path(measurement_id)
    assert cache_path.exists()

    # Load from cache
    cached = _load_from_cache(measurement_id)
    assert cached is not None
    pd.testing.assert_frame_equal(cached["time_series"], test_data["time_series"])
    pd.testing.assert_frame_equal(cached["steps"], test_data["steps"])


def test_cache_clear(isolated_cache):
    """Test cache clear functionality."""
    from ionworksdata.load import (
        _load_from_cache,
        _save_to_cache,
    )

    # Save multiple cache entries
    for i in range(3):
        _save_to_cache(f"test-clear-{i}", {"data": i})

    # Verify all are cached
    for i in range(3):
        assert _load_from_cache(f"test-clear-{i}") is not None

    # Clear cache
    count = iwdata.clear_cache()
    assert count == 3

    # Verify all are cleared
    for i in range(3):
        assert _load_from_cache(f"test-clear-{i}") is None

    # Clearing empty cache returns 0
    assert iwdata.clear_cache() == 0


def test_cache_clear_nonexistent_directory(isolated_cache):
    """Test cache clear on nonexistent directory returns 0."""
    from ionworksdata.load import _CACHE_CONFIG

    # Set to a non-existent directory
    _CACHE_CONFIG["directory"] = Path("/nonexistent/path/that/does/not/exist")
    assert iwdata.clear_cache() == 0


def test_cache_disabled_no_save(isolated_cache):
    """Test that cache is not saved when disabled."""
    from ionworksdata.load import (
        _CACHE_CONFIG,
        _get_cache_path,
        _load_from_cache,
        _save_to_cache,
    )

    _CACHE_CONFIG["enabled"] = False

    test_data = {"data": "test"}
    measurement_id = "test-disabled-cache"

    # Try to save (should do nothing)
    _save_to_cache(measurement_id, test_data)

    # Cache file should not exist
    cache_path = _get_cache_path(measurement_id)
    assert not cache_path.exists()

    # Load should return None
    assert _load_from_cache(measurement_id) is None


def test_cache_corrupted_file_deleted(isolated_cache):
    """Test that corrupted cache files are deleted on load."""
    from ionworksdata.load import (
        _get_cache_path,
        _load_from_cache,
    )

    measurement_id = "test-corrupted-cache"
    cache_path = _get_cache_path(measurement_id)

    # Create cache directory
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    # Write corrupted data to cache file
    with open(cache_path, "wb") as f:
        f.write(b"not a valid pickle file")

    assert cache_path.exists()

    # Try to load - should return None and delete corrupted file
    cached = _load_from_cache(measurement_id)
    assert cached is None
    assert not cache_path.exists()


def test_cache_path_special_characters():
    """Test that cache path handles special characters in measurement IDs."""
    from ionworksdata.load import _get_cache_path

    # Test with special characters
    test_ids = [
        "simple-id",
        "id/with/slashes",
        "id with spaces",
        "id:with:colons",
        "id@with@at",
        "id#with#hash",
        "very_long_id_" + "x" * 100,
    ]

    for measurement_id in test_ids:
        cache_path = _get_cache_path(measurement_id)
        # Should be a valid path
        assert cache_path is not None
        # Should end with .pkl
        assert cache_path.suffix == ".pkl"
        # Filename should only contain safe characters
        filename = cache_path.name
        assert "/" not in filename


def test_dataloader_from_db_uses_cache(isolated_cache):
    """Test that DataLoader.from_db uses cache correctly."""
    from unittest.mock import MagicMock, patch

    from ionworksdata.load import _load_from_cache

    time_series = pd.DataFrame(
        {
            "Time [s]": [0, 1, 2, 3],
            "Voltage [V]": [3.8, 3.7, 3.6, 3.5],
            "Current [A]": [1.0, 1.0, 1.0, 1.0],
        }
    )
    steps = pd.DataFrame(
        {
            "Start index": [0],
            "End index": [3],
            "Start voltage [V]": [3.8],
        }
    )

    mock_measurement_detail = MagicMock()
    mock_measurement_detail.time_series = time_series
    mock_measurement_detail.steps = steps

    mock_client = MagicMock()
    mock_client.cell_measurement.detail.return_value = mock_measurement_detail

    measurement_id = "test-cache-dataloader"

    with patch("ionworks.Ionworks", return_value=mock_client):
        # First call should hit the API
        loader1 = iwdata.DataLoader.from_db(measurement_id)
        assert mock_client.cell_measurement.detail.call_count == 1

    # Data should now be cached
    cached = _load_from_cache(measurement_id)
    assert cached is not None

    with patch("ionworks.Ionworks", return_value=mock_client):
        # Second call should use cache (API not called again)
        mock_client.cell_measurement.detail.reset_mock()
        loader2 = iwdata.DataLoader.from_db(measurement_id)
        assert mock_client.cell_measurement.detail.call_count == 0

    # Both loaders should have same data
    pd.testing.assert_frame_equal(
        loader1.data.reset_index(drop=True),
        loader2.data.reset_index(drop=True),
    )


def test_dataloader_from_db_use_cache_false(isolated_cache):
    """Test that DataLoader.from_db bypasses cache when use_cache=False."""
    from unittest.mock import MagicMock, patch

    from ionworksdata.load import _save_to_cache

    time_series = pd.DataFrame(
        {
            "Time [s]": [0, 1, 2, 3],
            "Voltage [V]": [3.8, 3.7, 3.6, 3.5],
            "Current [A]": [1.0, 1.0, 1.0, 1.0],
        }
    )
    steps = pd.DataFrame(
        {
            "Start index": [0],
            "End index": [3],
            "Start voltage [V]": [3.8],
        }
    )

    mock_measurement_detail = MagicMock()
    mock_measurement_detail.time_series = time_series
    mock_measurement_detail.steps = steps

    mock_client = MagicMock()
    mock_client.cell_measurement.detail.return_value = mock_measurement_detail

    measurement_id = "test-bypass-cache"

    # Pre-populate cache with different data
    old_data = {
        "time_series": pd.DataFrame({"Time [s]": [99]}),
        "steps": pd.DataFrame({"Start index": [99]}),
    }
    _save_to_cache(measurement_id, old_data)

    with patch("ionworks.Ionworks", return_value=mock_client):
        # Call with use_cache=False should bypass cache and hit API
        loader = iwdata.DataLoader.from_db(measurement_id, use_cache=False)
        assert mock_client.cell_measurement.detail.call_count == 1

    # Should have fresh data, not cached data
    assert loader.data["Time [s]"].iloc[0] == 0  # Not 99


def test_ocp_dataloader_from_db_uses_cache(isolated_cache):
    """Test that OCPDataLoader.from_db uses cache correctly."""
    from unittest.mock import MagicMock, patch

    from ionworksdata.load import _load_from_cache

    time_series = pd.DataFrame(
        {
            "Capacity [A.h]": [0.0, 0.5, 1.0, 1.5],
            "Voltage [V]": [4.2, 3.9, 3.7, 3.5],
        }
    )

    mock_measurement_detail = MagicMock()
    mock_measurement_detail.time_series = time_series

    mock_client = MagicMock()
    mock_client.cell_measurement.detail.return_value = mock_measurement_detail

    measurement_id = "test-cache-ocp-dataloader"

    with patch("ionworks.Ionworks", return_value=mock_client):
        # First call should hit the API
        loader1 = iwdata.OCPDataLoader.from_db(measurement_id)
        assert mock_client.cell_measurement.detail.call_count == 1

    # Data should now be cached
    cached = _load_from_cache(measurement_id)
    assert cached is not None

    with patch("ionworks.Ionworks", return_value=mock_client):
        # Second call should use cache (API not called again)
        mock_client.cell_measurement.detail.reset_mock()
        loader2 = iwdata.OCPDataLoader.from_db(measurement_id)
        assert mock_client.cell_measurement.detail.call_count == 0

    # Both loaders should have same data
    pd.testing.assert_frame_equal(
        loader1.data.reset_index(drop=True),
        loader2.data.reset_index(drop=True),
    )
