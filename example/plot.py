import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pathlib
import iwutil

this_dir = pathlib.Path(__file__).parent


def plot_variables(data, vars_to_plot):
    n = len(vars_to_plot)
    fig, axes = plt.subplots(n, 1, figsize=(6, 2 * n), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, v in zip(axes, vars_to_plot):
        ax.plot(data["Time [s]"], data[v])
        ax.set_ylabel(v)
        ax.grid(alpha=0.5)

    x_min = data["Time [s]"].min()
    x_max = data["Time [s]"].max()
    axes[-1].set_xlim(x_min, x_max)
    axes[-1].set_xlabel("Time [s]")

    fig.tight_layout()
    return fig, axes


# gitt
data = pd.read_csv(this_dir / "data" / "gitt" / "data.csv")
fig, axes = plot_variables(
    data, ["Current [A]", "Voltage [V]", "Step number", "Cycle number"]
)
axes[0].set_title("Example GITT data")
fig.tight_layout()
iwutil.save.fig(fig, this_dir / "figures" / "gitt.png")

# hppt
data = pd.read_csv(this_dir / "data" / "hppt" / "data.csv")
fig, axes = plot_variables(
    data, ["Current [A]", "Voltage [V]", "Step number", "Cycle number"]
)
axes[0].set_title("Example HPPT data")
fig.tight_layout()
iwutil.save.fig(fig, this_dir / "figures" / "hppt.png")

# constant current
data = pd.read_csv(this_dir / "data" / "1C" / "data.csv")
fig, axes = plot_variables(data, ["Current [A]", "Voltage [V]"])
axes[0].set_title("Example constant current data")
fig.tight_layout()
iwutil.save.fig(fig, this_dir / "figures" / "1C.png")

# ocv
data = pd.read_csv(this_dir / "data" / "ocv" / "data.csv")
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
ax.plot(data["Capacity [A.h]"], data["Voltage [V]"])
ax.set_xlabel("Capacity [A.h]")
ax.set_ylabel("Voltage [V]")
ax.grid(alpha=0.5)
ax.set_xlim(0, data["Capacity [A.h]"].max())
ax.set_title("Example OCV data")
fig.tight_layout()
iwutil.save.fig(fig, this_dir / "figures" / "ocv.png")

plt.show()
