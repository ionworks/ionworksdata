import pandas as pd
import polars as pl
import ionworksdata as iwdata
import numpy as np


def test_steps_with_cycle_count():
    data = pd.DataFrame(
        {
            "Time [s]": [0, 1, 2, 3, 4],
            "Current [A]": [0, 1, 1, 0, -1],
            "Voltage [V]": [4, 3, 2, 3, 2],
            "Step number": [1, 2, 3, 4, 5],
            "Cycle from cycler": [1, 1, 1, 2, 2],
        }
    )
    # Add step count first
    data_pl = pl.from_pandas(data)
    options = {"step column": "Step number"}
    data_with_step_count = iwdata.transform.set_step_count(data_pl, options=options)
    steps = iwdata.steps.summarize(data_with_step_count).to_pandas()
    assert steps["Step count"].tolist() == [0, 1, 2, 3, 4]
    assert steps["Cycle from cycler"].tolist() == [1, 1, 1, 2, 2]
    assert steps["Cycle count"].tolist() == [0, 0, 0, 1, 1]


def test_steps_with_preexisting_step_count():
    """Test creating steps table from time series that already has Step count."""
    data = pd.DataFrame(
        {
            "Time [s]": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            "Current [A]": [4.0, 4.0, -1.0, -1.0, 4.0, 4.0, -1.0, -1.0],
            "Voltage [V]": [4.0, 2.0, 3.0, 4.0, 3.0, 2.0, 3.0, 4.0],
            "Step from cycler": [0, 0, 1, 1, 0, 0, 1, 1],
            "Cycle from cycler": [0, 0, 0, 0, 1, 1, 1, 1],
        }
    )
    data_pl = pl.from_pandas(data)

    # First set step count on the time series
    options = {"step column": "Step from cycler"}
    data_with_step_count = iwdata.transform.set_step_count(data_pl, options=options)

    # Verify step count was added to time series
    assert "Step count" in data_with_step_count.columns
    assert data_with_step_count["Step count"].to_list() == [0, 0, 1, 1, 2, 2, 3, 3]

    # Now create steps table from time series that already has Step count
    steps = iwdata.steps.summarize(data_with_step_count)

    # Verify steps table was created correctly
    assert steps["Start index"].to_list() == [0, 2, 4, 6]
    assert steps["End index"].to_list() == [1, 3, 5, 7]
    assert steps["Step type"].to_list() == [
        "Constant current discharge",
        "Constant current charge",
        "Constant current discharge",
        "Constant current charge",
    ]
    assert steps["Step from cycler"].to_list() == [0, 1, 0, 1]
    assert steps["Cycle from cycler"].to_list() == [0, 0, 1, 1]
    assert steps["Step count"].to_list() == [0, 1, 2, 3]
    assert steps["Cycle count"].to_list() == [0, 0, 1, 1]


def test_steps_using_step_count_as_step_column():
    """Test creating steps table using 'Step count' as the step_column.

    This tests that the method properly handles the case where 'Step count'
    is used as the grouping column, ensuring no duplicate column conflicts.
    """
    data = pd.DataFrame(
        {
            "Time [s]": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            "Current [A]": [4.0, 4.0, -1.0, -1.0, 4.0, 4.0, -1.0, -1.0],
            "Voltage [V]": [4.0, 2.0, 3.0, 4.0, 3.0, 2.0, 3.0, 4.0],
            "Step from cycler": [0, 0, 1, 1, 0, 0, 1, 1],
            "Cycle from cycler": [0, 0, 0, 0, 1, 1, 1, 1],
        }
    )
    data_pl = pl.from_pandas(data)

    # First set step count on the time series
    options = {"step column": "Step from cycler"}
    data_with_step_count = iwdata.transform.set_step_count(data_pl, options=options)

    # Verify step count was added to time series
    assert "Step count" in data_with_step_count.columns
    assert data_with_step_count["Step count"].to_list() == [0, 0, 1, 1, 2, 2, 3, 3]

    # Now use "Step count" - this is now the default
    steps = iwdata.steps.summarize(data_with_step_count)

    # Verify steps table was created correctly
    # The steps should be grouped by the Step count values
    assert steps["Start index"].to_list() == [0, 2, 4, 6]
    assert steps["End index"].to_list() == [1, 3, 5, 7]
    assert steps["Step type"].to_list() == [
        "Constant current discharge",
        "Constant current charge",
        "Constant current discharge",
        "Constant current charge",
    ]
    # The output should preserve the original "Step count" values from the time series
    # which were [0, 0, 1, 1, 2, 2, 3, 3], so grouped they are [0, 1, 2, 3]
    assert steps["Step count"].to_list() == [0, 1, 2, 3]
    # "Step from cycler" should also be present even though we used "Step count" as step_column
    assert steps["Step from cycler"].to_list() == [0, 1, 0, 1]
    assert steps["Cycle from cycler"].to_list() == [0, 0, 1, 1]
    assert steps["Cycle count"].to_list() == [0, 0, 1, 1]


def test_steps_using_step_count_with_nonsequential_values():
    """Test that original Step count values are preserved when used as step_column.

    This test uses non-sequential Step count values to verify they are preserved
    in the steps table output rather than being replaced with a sequential index.
    """
    data = pd.DataFrame(
        {
            "Time [s]": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "Current [A]": [4.0, 4.0, -1.0, -1.0, 4.0, 4.0],
            "Voltage [V]": [4.0, 2.0, 3.0, 4.0, 3.0, 2.0],
            # Using non-sequential step count values: 5, 10, 15
            "Step count": [5, 5, 10, 10, 15, 15],
            "Cycle from cycler": [0, 0, 0, 0, 1, 1],
        }
    )
    data_pl = pl.from_pandas(data)

    # Create steps table using "Step count" (now the default)
    steps = iwdata.steps.summarize(data_pl)

    # Verify that the original Step count values are preserved
    assert steps["Start index"].to_list() == [0, 2, 4]
    assert steps["End index"].to_list() == [1, 3, 5]
    assert steps["Step type"].to_list() == [
        "Constant current discharge",
        "Constant current charge",
        "Constant current discharge",
    ]
    # The key test: Step count should be [5, 10, 15], not [0, 1, 2]
    assert steps["Step count"].to_list() == [5, 10, 15]
    assert steps["Cycle from cycler"].to_list() == [0, 0, 1]
    assert steps["Cycle count"].to_list() == [0, 0, 1]


def test_cycling():
    steps = pd.DataFrame(
        {
            "Step count": [0, 1, 2, 3, 4, 5, 6, 7, 8],
            "Step type": [
                "Rest",
                "Constant current discharge",
                "Rest",
                "Constant current charge",
                "Constant voltage charge",
                "Rest",
                "EIS",
                "Constant current discharge",
                "Rest",
            ],
            "Discharge capacity [A.h]": [0, 1, 0, 0, 0, 0, 0, 1, 0],
            "Charge capacity [A.h]": [0, 0, 0, 0.9, 0.1, 0, 0, 0, 0],
        }
    )
    steps["Label"] = ""
    steps["Group number"] = np.nan
    steps_expected = steps.copy()
    steps_expected["Label"] = ["Cycling"] * 6 + [""] + ["Cycling"] * 2
    steps_expected["Group number"] = [0, 1, 1, 2, 2, 2, np.nan, 0, 0]
    options = {"cell_metadata": {"Nominal cell capacity [A.h]": 1}}
    steps_labeled = iwdata.steps.label_cycling(steps, options=options)
    pd.testing.assert_frame_equal(
        steps_labeled.sort_index(axis=1), steps_expected.sort_index(axis=1)
    )
    assert iwdata.steps.validate(steps_labeled, "Cycling")


def test_gitt():
    steps = pd.DataFrame(
        {
            "Step count": [0, 1, 2, 3, 4, 5, 6, 7, 8],
            "Step type": ["Constant current discharge", "Rest"] * 2
            + ["EIS"]
            + ["Constant current discharge", "Rest"] * 2,
            "Discharge capacity [A.h]": [0.15, 0] * 2 + [0] + [0.15, 0] * 2,
            "Charge capacity [A.h]": [0, 0] * 2 + [0] + [0, 0] * 2,
        }
    )
    steps["Label"] = ""
    steps["Group number"] = np.nan
    steps_expected = steps.copy()
    steps_expected["Label"] = ["GITT"] * 4 + [""] + ["GITT"] * 4
    steps_expected["Group number"] = [0, 0, 1, 1, None, 0, 0, 1, 1]
    options = {"cell_metadata": {"Nominal cell capacity [A.h]": 1}}
    steps_labeled = iwdata.steps.label_pulse(steps, options=options)
    pd.testing.assert_frame_equal(
        steps_labeled.sort_index(axis=1), steps_expected.sort_index(axis=1)
    )
    assert iwdata.steps.validate(steps_labeled, "GITT")


def test_hppt():
    steps = pd.DataFrame(
        {
            "Step count": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            "Step type": ["Constant current discharge", "Rest"] * 8,
            "Discharge capacity [A.h]": [0.15, 0, 0.01, 0, 0.01, 0, 0.01, 0] * 2,
            "Charge capacity [A.h]": [0, 0, 0, 0, 0, 0, 0, 0] * 2,
        }
    )
    steps["Label"] = ""
    steps["Group number"] = np.nan
    steps_expected = steps.copy()
    steps_expected["Label"] = ["HPPT"] * 16
    steps_expected["Group number"] = [0.0] * 8 + [1.0] * 8
    options = {"cell_metadata": {"Nominal cell capacity [A.h]": 1}}
    steps_labeled = iwdata.steps.label_pulse(steps, options=options)
    pd.testing.assert_frame_equal(
        steps_labeled.sort_index(axis=1), steps_expected.sort_index(axis=1)
    )
    assert iwdata.steps.validate(steps_labeled, "HPPT")


def test_eis():
    steps = pd.DataFrame(
        {
            "Step count": [0, 0, 1, 1, 2, 2, 3, 3],
            "Step type": ["EIS", "EIS", None, None, "EIS", "EIS", None, None],
        }
    )
    steps["Label"] = ""
    steps["Group number"] = np.nan
    steps_expected = steps.copy()
    steps_expected["Label"] = [
        "EIS",
        "EIS",
        "",
        "",
        "EIS",
        "EIS",
        "",
        "",
    ]
    steps_expected["Group number"] = [0, 0, np.nan, np.nan, 1, 1, np.nan, np.nan]
    steps_labeled = iwdata.steps.label_eis(steps)
    pd.testing.assert_frame_equal(steps_labeled, steps_expected)
    assert iwdata.steps.validate(steps_labeled, "EIS")


def test_cycling_invalid():
    steps = pd.DataFrame(
        {
            "Step count": [0, 1, 2],
            "Step type": [
                "Rest",
                "Constant current discharge",
                "Rest",
            ],
            "Discharge capacity [A.h]": [0, 1, 0],
            "Charge capacity [A.h]": [0, 0, 0],
            "Cycle count": [0, 1, 2],
        }
    )
    steps["Label"] = ""
    steps["Group number"] = np.nan
    steps_expected = steps.copy()
    steps_expected["Label"] = ["Cycling"] * 3
    steps_expected["Group number"] = [0.0, 1.0, 1.0]
    options = {"cell_metadata": {"Nominal cell capacity [A.h]": 1}}
    steps_labeled = iwdata.steps.label_cycling(steps, options=options)
    pd.testing.assert_frame_equal(
        steps_labeled.sort_index(axis=1), steps_expected.sort_index(axis=1)
    )
    assert not iwdata.steps.validate(steps_labeled, "Cycling")


def test_annotate():
    steps = pd.DataFrame(
        {
            "Start index": [0, 4, 6],
            "End index": [3, 5, 8],
            "Extra column": ["a", "b", "c"],
        }
    )
    time_series = pd.DataFrame(
        {
            "Time [s]": [0, 1, 2, 3, 4, 5, 6, 7, 8],
        }
    )
    time_series_expected = time_series.copy()
    time_series_expected["Extra column"] = ["a"] * 4 + ["b"] * 2 + ["c"] * 3
    time_series_labeled = iwdata.steps.annotate(time_series, steps, ["Extra column"])
    pd.testing.assert_frame_equal(time_series_labeled, time_series_expected)


def test_annotate_with_polars():
    """Test annotate with Polars DataFrames."""

    steps = pd.DataFrame(
        {
            "Start index": [0, 4, 6],
            "End index": [3, 5, 8],
            "Extra column": ["a", "b", "c"],
        }
    )
    time_series = pd.DataFrame(
        {
            "Time [s]": [0, 1, 2, 3, 4, 5, 6, 7, 8],
        }
    )

    # Convert to Polars
    steps_pl = pl.from_pandas(steps)
    time_series_pl = pl.from_pandas(time_series)

    # Test with Polars input
    time_series_labeled_pl = iwdata.steps.annotate(
        time_series_pl, steps_pl, ["Extra column"]
    )

    # Test with Pandas input
    time_series_labeled_pd = iwdata.steps.annotate(time_series, steps, ["Extra column"])

    # Polars input should return Polars, Pandas should return Pandas
    assert isinstance(time_series_labeled_pl, pl.DataFrame)
    assert isinstance(time_series_labeled_pd, pd.DataFrame)

    # Results should be identical when converted to same type
    pd.testing.assert_frame_equal(
        time_series_labeled_pl.to_pandas(), time_series_labeled_pd
    )


def test_set_cycle_capacity_with_polars():
    """Test set_cycle_capacity with Polars DataFrames."""

    # Create test steps data with new format
    steps_pd = pd.DataFrame(
        {
            "Cycle count": [0, 0, 1, 1],
            "Mean current [A]": [-1.0, 1.0, -1.0, 1.0],
            "Discharge capacity [A.h]": [0.0, 0.9, 0.0, 1.0],
            "Charge capacity [A.h]": [1.0, 0.0, 1.1, 0.0],
        }
    )

    # Convert to Polars
    steps_pl = pl.from_pandas(steps_pd)

    # Test with Polars input
    result_pl = iwdata.steps.set_cycle_capacity(steps_pl)

    # Both should return Polars DataFrames
    assert isinstance(result_pl, pl.DataFrame)

    # Check that cycle capacities are calculated correctly
    assert "Cycle charge capacity [A.h]" in result_pl.columns
    assert "Cycle discharge capacity [A.h]" in result_pl.columns

    # Verify values
    cycle_0_charge = result_pl.filter(pl.col("Cycle count") == 0)[
        "Cycle charge capacity [A.h]"
    ][0]
    cycle_0_discharge = result_pl.filter(pl.col("Cycle count") == 0)[
        "Cycle discharge capacity [A.h]"
    ][0]

    assert cycle_0_charge == 1.0  # Sum of charge capacity for cycle 0
    assert cycle_0_discharge == 0.9  # Sum of discharge capacity for cycle 0


def test_set_cycle_energy_with_polars():
    """Test set_cycle_energy with Polars DataFrames."""

    # Create test steps data with energy columns
    steps_pd = pd.DataFrame(
        {
            "Cycle count": [0, 0, 1, 1],
            "Mean current [A]": [-1.0, 1.0, -1.0, 1.0],
            "Discharge energy [W.h]": [0.0, 0.9, 0.0, 1.0],
            "Charge energy [W.h]": [1.0, 0.0, 1.1, 0.0],
        }
    )

    # Convert to Polars
    steps_pl = pl.from_pandas(steps_pd)

    # Test with Polars input
    result_pl = iwdata.steps.set_cycle_energy(steps_pl)

    # Both should return Polars DataFrames
    assert isinstance(result_pl, pl.DataFrame)

    # Check that cycle energies are calculated correctly
    assert "Cycle charge energy [W.h]" in result_pl.columns
    assert "Cycle discharge energy [W.h]" in result_pl.columns

    # Verify values
    cycle_0_charge = result_pl.filter(pl.col("Cycle count") == 0)[
        "Cycle charge energy [W.h]"
    ][0]
    cycle_0_discharge = result_pl.filter(pl.col("Cycle count") == 0)[
        "Cycle discharge energy [W.h]"
    ][0]

    assert cycle_0_charge == 1.0  # Sum of charge energy for cycle 0
    assert cycle_0_discharge == 0.9  # Sum of discharge energy for cycle 0


def test_summarize_with_polars():
    """Test summarize with Polars DataFrames."""

    # Create test time series data
    time_series_pd = pd.DataFrame(
        {
            "Time [s]": [0, 1, 2, 3, 4, 5],
            "Voltage [V]": [3.0, 3.1, 3.2, 3.3, 3.4, 3.5],
            "Current [A]": [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            "Step from cycler": [0, 0, 0, 1, 1, 1],
            "Capacity [A.h]": [0.0, 0.1, 0.2, 0.2, 0.2, 0.2],
        }
    )

    # Convert to Polars and add step count
    time_series_pl = pl.from_pandas(time_series_pd)
    options = {"step column": "Step from cycler"}
    time_series_pl = iwdata.transform.set_step_count(time_series_pl, options=options)
    time_series_pd = time_series_pl.to_pandas()

    # Test with Polars input
    steps_pl = iwdata.steps.summarize(time_series_pl)

    # Test with Pandas input
    steps_pd = iwdata.steps.summarize(time_series_pd)

    # Both should return Polars DataFrames
    assert isinstance(steps_pl, pl.DataFrame)
    assert isinstance(steps_pd, pl.DataFrame)

    # Results should be identical
    pd.testing.assert_frame_equal(steps_pl.to_pandas(), steps_pd.to_pandas())

    # Verify step types are identified
    assert len(steps_pl) == 2
    assert "Step type" in steps_pl.columns


def test_step_capacity_always_positive():
    """Test that step-level discharge and charge capacity are always >= 0."""
    # Test case 1: Pure discharge steps
    data1 = pd.DataFrame(
        {
            "Time [s]": [0, 1, 2, 3, 4, 5],
            "Voltage [V]": [4.0, 3.9, 3.8, 3.7, 3.6, 3.5],
            "Current [A]": [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            "Step count": [0, 0, 0, 1, 1, 1],
        }
    )
    steps1 = iwdata.steps.summarize(data1).to_pandas()
    assert all(steps1["Discharge capacity [A.h]"] >= 0)
    assert all(steps1["Charge capacity [A.h]"] >= 0)

    # Test case 2: Pure charge steps
    data2 = pd.DataFrame(
        {
            "Time [s]": [0, 1, 2, 3, 4, 5],
            "Voltage [V]": [3.0, 3.1, 3.2, 3.3, 3.4, 3.5],
            "Current [A]": [-1.0, -1.0, -1.0, 0.0, 0.0, 0.0],
            "Step count": [0, 0, 0, 1, 1, 1],
        }
    )
    steps2 = iwdata.steps.summarize(data2).to_pandas()
    assert all(steps2["Discharge capacity [A.h]"] >= 0)
    assert all(steps2["Charge capacity [A.h]"] >= 0)

    # Test case 3: Mixed discharge and charge steps
    data3 = pd.DataFrame(
        {
            "Time [s]": [0, 1, 2, 3, 4, 5, 6, 7],
            "Voltage [V]": [4.0, 3.9, 3.8, 3.7, 3.6, 3.7, 3.8, 3.9],
            "Current [A]": [1.0, 1.0, -1.0, -1.0, 0.0, 0.0, 1.0, 1.0],
            "Step count": [0, 0, 1, 1, 2, 2, 3, 3],
        }
    )
    steps3 = iwdata.steps.summarize(data3).to_pandas()
    assert all(steps3["Discharge capacity [A.h]"] >= 0)
    assert all(steps3["Charge capacity [A.h]"] >= 0)

    # Test case 4: With rest steps (zero capacity)
    data4 = pd.DataFrame(
        {
            "Time [s]": [0, 1, 2, 3, 4, 5],
            "Voltage [V]": [4.0, 3.9, 3.8, 3.8, 3.8, 3.8],
            "Current [A]": [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            "Step count": [0, 0, 1, 1, 1, 1],
        }
    )
    steps4 = iwdata.steps.summarize(data4).to_pandas()
    assert all(steps4["Discharge capacity [A.h]"] >= 0)
    assert all(steps4["Charge capacity [A.h]"] >= 0)


def test_step_energy_always_positive():
    """Test that step-level discharge and charge energy are always >= 0."""
    # Test case 1: Pure discharge steps
    data1 = pd.DataFrame(
        {
            "Time [s]": [0, 1, 2, 3, 4, 5],
            "Voltage [V]": [4.0, 3.9, 3.8, 3.7, 3.6, 3.5],
            "Current [A]": [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            "Step count": [0, 0, 0, 1, 1, 1],
        }
    )
    steps1 = iwdata.steps.summarize(data1).to_pandas()
    if "Discharge energy [W.h]" in steps1.columns:
        assert all(steps1["Discharge energy [W.h]"] >= 0)
    if "Charge energy [W.h]" in steps1.columns:
        assert all(steps1["Charge energy [W.h]"] >= 0)

    # Test case 2: Pure charge steps
    data2 = pd.DataFrame(
        {
            "Time [s]": [0, 1, 2, 3, 4, 5],
            "Voltage [V]": [3.0, 3.1, 3.2, 3.3, 3.4, 3.5],
            "Current [A]": [-1.0, -1.0, -1.0, 0.0, 0.0, 0.0],
            "Step count": [0, 0, 0, 1, 1, 1],
        }
    )
    steps2 = iwdata.steps.summarize(data2).to_pandas()
    if "Discharge energy [W.h]" in steps2.columns:
        assert all(steps2["Discharge energy [W.h]"] >= 0)
    if "Charge energy [W.h]" in steps2.columns:
        assert all(steps2["Charge energy [W.h]"] >= 0)

    # Test case 3: Mixed discharge and charge steps
    data3 = pd.DataFrame(
        {
            "Time [s]": [0, 1, 2, 3, 4, 5, 6, 7],
            "Voltage [V]": [4.0, 3.9, 3.8, 3.7, 3.6, 3.7, 3.8, 3.9],
            "Current [A]": [1.0, 1.0, -1.0, -1.0, 0.0, 0.0, 1.0, 1.0],
            "Step count": [0, 0, 1, 1, 2, 2, 3, 3],
        }
    )
    steps3 = iwdata.steps.summarize(data3).to_pandas()
    if "Discharge energy [W.h]" in steps3.columns:
        assert all(steps3["Discharge energy [W.h]"] >= 0)
    if "Charge energy [W.h]" in steps3.columns:
        assert all(steps3["Charge energy [W.h]"] >= 0)

    # Test case 4: With rest steps (zero energy)
    data4 = pd.DataFrame(
        {
            "Time [s]": [0, 1, 2, 3, 4, 5],
            "Voltage [V]": [4.0, 3.9, 3.8, 3.8, 3.8, 3.8],
            "Current [A]": [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            "Step count": [0, 0, 1, 1, 1, 1],
        }
    )
    steps4 = iwdata.steps.summarize(data4).to_pandas()
    if "Discharge energy [W.h]" in steps4.columns:
        assert all(steps4["Discharge energy [W.h]"] >= 0)
    if "Charge energy [W.h]" in steps4.columns:
        assert all(steps4["Charge energy [W.h]"] >= 0)
