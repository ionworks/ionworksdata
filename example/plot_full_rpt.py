import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pathlib
import ionworksdata as iwdata

this_dir = pathlib.Path(__file__).parent


def plot_variables(data, vars_to_plot, steps=None, title=None, split_by_label=False):
    n = len(vars_to_plot)
    fig, axes = plt.subplots(n, 1, figsize=(6, 2 * n), sharex=True)
    axes = np.atleast_1d(axes)

    if split_by_label:
        if steps is None:
            raise ValueError("steps must be provided if split_by_label is True")
        labels = ["", "Cycling", "HPPT", "GITT"]
        colors = ["black", "blue", "red", "green"]
        for label, color in zip(labels, colors):
            axes[0].plot([], [], color=color, label=label)
            step_nums = steps.loc[steps["Label"] == label, "Step count"]
            # Split step numbers into contiguous blocks
            blocks = []
            if len(step_nums) > 0:
                current_block = [step_nums.iloc[0]]
                for step in step_nums.iloc[1:]:
                    if step == current_block[-1] + 1:
                        current_block.append(step)
                    else:
                        blocks.append(current_block)
                        current_block = [step]
                blocks.append(current_block)

            # Plot each contiguous block separately
            for block in blocks:
                data_label = data.loc[data["Step count"].isin(block)]
                for ax, v in zip(axes, vars_to_plot):
                    ax.plot(data_label["Time [s]"], data_label[v], color=color)
                    ax.set_ylabel(v)
                    ax.grid(alpha=0.5)
        axes[0].legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0)
    else:
        for ax, v in zip(axes, vars_to_plot):
            ax.plot(data["Time [s]"], data[v])
            ax.set_ylabel(v)
            ax.grid(alpha=0.5)

    x_min = data["Time [s]"].min()
    x_max = data["Time [s]"].max()
    axes[-1].set_xlim(x_min, x_max)
    axes[-1].set_xlabel("Time [s]")

    if title:
        fig.suptitle(title)

    fig.tight_layout()
    return fig, axes


data = pd.read_csv(this_dir / "data" / "full_rpt" / "data.csv")
data.rename(columns={"Step": "Step number", "Cycle": "Cycle number"}, inplace=True)

# Plot the full data
fig, axes = plot_variables(
    data,
    ["Current [A]", "Voltage [V]", "Temperature [degC]", "Step number", "Cycle number"],
    title="Full RPT data",
)

# Add step count (required for summarize)
data = iwdata.transform.set_step_count(data, options={"step column": "Step number"})
# Note: default step column is "Step from cycler", but this example data uses "Step number"

# Get the steps information and add cycling and pulse labels
steps = iwdata.steps.summarize(data)
options = {"cell_metadata": {"Nominal cell capacity [A.h]": 5}}
steps = iwdata.steps.label_cycling(steps, options=options)
for direction in ["charge", "discharge"]:
    options["current direction"] = direction
    steps = iwdata.steps.label_pulse(steps, options=options)
for step_count in data["Step count"].unique():
    group_number = steps.loc[steps["Step count"] == step_count, "Group number"].values[
        0
    ]
    data.loc[data["Step count"] == step_count, "Group number"] = group_number

# Plot the labeled data
fig, axes = plot_variables(
    data,
    ["Current [A]", "Voltage [V]", "Step count", "Group number"],
    steps=steps,
    title="Labeled RPT data",
    split_by_label=True,
)

plt.show()
