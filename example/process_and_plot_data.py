"""
Plot a full RPT (Reference Performance Test) dataset using ionworksdata.

This example demonstrates how to:
1. Read a CSV file using ionworksdata's measurement_details reader
2. Annotate the time series with each step's type
3. Visualise raw data and data coloured by step type
"""

import pathlib

import ionworksdata as iwdata
import matplotlib.pyplot as plt
import numpy as np

this_dir = pathlib.Path(__file__).parent

# ---------------------------------------------------------------------------
# Helper: plot selected variables against time
# ---------------------------------------------------------------------------


def plot_variables(data, vars_to_plot, title=None, split_by_type=False):
    n = len(vars_to_plot)
    fig, axes = plt.subplots(n, 1, figsize=(6, 2 * n), sharex=True)
    axes = np.atleast_1d(axes)

    if split_by_type:
        step_types = sorted(data["Step type"].unique())
        cmap = plt.get_cmap("tab10")
        colors = {t: cmap(i % 10) for i, t in enumerate(step_types)}
        for step_type, color in colors.items():
            axes[0].plot([], [], color=color, label=step_type)
        # One line per step, so steps of the same type are not joined across gaps.
        for _, step_data in data.groupby("Step count"):
            color = colors[step_data["Step type"].iloc[0]]
            for ax, v in zip(axes, vars_to_plot, strict=False):
                ax.plot(step_data["Time [s]"], step_data[v], color=color)
        axes[0].legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0)
    else:
        for ax, v in zip(axes, vars_to_plot, strict=False):
            ax.plot(data["Time [s]"], data[v])

    # Format axes
    for ax, v in zip(axes, vars_to_plot, strict=False):
        ax.set_ylabel(v)
        ax.grid(alpha=0.5)
    axes[-1].set_xlim(data["Time [s]"].min(), data["Time [s]"].max())
    axes[-1].set_xlabel("Time [s]")
    if title:
        fig.suptitle(title)

    fig.tight_layout()
    return fig, axes


data_path = this_dir / "data" / "full_rpt" / "data.csv"
result = iwdata.read.measurement_details(
    data_path,
    measurement={},
    reader="csv",
    # Names the raw step/cycle columns so step counting works.
    extra_column_mappings={
        "Step": "Step from cycler",
        "Cycle": "Cycle from cycler",
    },
)

time_series = result["time_series"].to_pandas()
steps = result["steps"].to_pandas()

fig, axes = plot_variables(
    time_series,
    ["Current [A]", "Voltage [V]", "Temperature [degC]", "Step count"],
    title="Full RPT data",
)

time_series = iwdata.steps.annotate(time_series, steps, ["Step type"]).to_pandas()

fig, axes = plot_variables(
    time_series,
    ["Current [A]", "Voltage [V]", "Step count"],
    title="RPT data by step type",
    split_by_type=True,
)

plt.show()
