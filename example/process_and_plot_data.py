"""
Plot a full RPT (Reference Performance Test) dataset using ionworksdata.

This example demonstrates how to:
1. Read a CSV file using ionworksdata's measurement_details reader
2. Automatically label steps as cycling, GITT, or HPPT
3. Annotate the time series with step labels
4. Visualise raw and labeled data
"""

import matplotlib.pyplot as plt
import numpy as np
import pathlib
import ionworksdata as iwdata

this_dir = pathlib.Path(__file__).parent

# ---------------------------------------------------------------------------
# Helper: plot selected variables against time
# ---------------------------------------------------------------------------


def _contiguous_blocks(values):
    """Split a sorted sequence of integers into lists of consecutive runs."""
    blocks = []
    for v in values:
        if blocks and v == blocks[-1][-1] + 1:
            blocks[-1].append(v)
        else:
            blocks.append([v])
    return blocks


LABEL_COLORS = {"": "black", "Cycling": "blue", "HPPT": "red", "GITT": "green"}


def plot_variables(data, vars_to_plot, steps=None, title=None, split_by_label=False):
    n = len(vars_to_plot)
    fig, axes = plt.subplots(n, 1, figsize=(6, 2 * n), sharex=True)
    axes = np.atleast_1d(axes)

    if split_by_label:
        if steps is None:
            raise ValueError("steps must be provided if split_by_label is True")
        for label, color in LABEL_COLORS.items():
            axes[0].plot([], [], color=color, label=label)
            step_nums = steps.loc[steps["Label"] == label, "Step count"]
            for block in _contiguous_blocks(step_nums):
                block_data = data.loc[data["Step count"].isin(block)]
                for ax, v in zip(axes, vars_to_plot):
                    ax.plot(block_data["Time [s]"], block_data[v], color=color)
        axes[0].legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0)
    else:
        for ax, v in zip(axes, vars_to_plot):
            ax.plot(data["Time [s]"], data[v])

    # Format axes
    for ax, v in zip(axes, vars_to_plot):
        ax.set_ylabel(v)
        ax.grid(alpha=0.5)
    axes[-1].set_xlim(data["Time [s]"].min(), data["Time [s]"].max())
    axes[-1].set_xlabel("Time [s]")
    if title:
        fig.suptitle(title)

    fig.tight_layout()
    return fig, axes


# ---------------------------------------------------------------------------
# 1. Read the CSV and label steps with measurement_details
# ---------------------------------------------------------------------------
# measurement_details() is a convenience function that:
#   - reads the raw CSV via the built-in CSV reader
#   - detects step boundaries and computes a cumulative "Step count"
#   - summarises each step (duration, capacity, type, etc.)
#   - labels steps as Cycling, GITT, HPPT, or EIS using the nominal
#     cell capacity
#
# extra_column_mappings tells the CSV reader which raw columns map to
# the standard "Step from cycler" and "Cycle from cycler" names so that
# step counting works correctly.

data_path = this_dir / "data" / "full_rpt" / "data.csv"
result = iwdata.read.measurement_details(
    data_path,
    measurement={},
    reader="csv",
    extra_column_mappings={
        "Step": "Step from cycler",
        "Cycle": "Cycle from cycler",
    },
    options={"cell_metadata": {"Nominal cell capacity [A.h]": 5}},
)

# measurement_details returns a dict with three keys:
#   "time_series" - polars DataFrame with standardised columns
#   "steps"       - polars DataFrame with one row per step, including labels
#   "measurement" - dict of metadata (cycler name, start time, etc.)
time_series = result["time_series"].to_pandas()
steps = result["steps"].to_pandas()

# ---------------------------------------------------------------------------
# 2. Plot the raw (unlabeled) data
# ---------------------------------------------------------------------------
fig, axes = plot_variables(
    time_series,
    ["Current [A]", "Voltage [V]", "Temperature [degC]", "Step count"],
    title="Full RPT data",
)

# ---------------------------------------------------------------------------
# 3. Annotate time series with step labels and group numbers
# ---------------------------------------------------------------------------
# annotate() copies columns from the steps table onto the time series so
# that each row knows which label and group it belongs to.  This is useful
# for colour-coding plots by experiment type.
time_series = iwdata.steps.annotate(time_series, steps, ["Label", "Group number"])

# ---------------------------------------------------------------------------
# 4. Plot the labeled data, colour-coded by experiment type
# ---------------------------------------------------------------------------
fig, axes = plot_variables(
    time_series,
    ["Current [A]", "Voltage [V]", "Step count", "Group number"],
    steps=steps,
    title="Labeled RPT data",
    split_by_label=True,
)

plt.show()
