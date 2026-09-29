# Examples

This directory contains scripts to generate synthetic battery data using
[PyBaMM](https://pybamm.org/) and process and visualize it using `ionworksdata`.

## Prerequisites

Install the project with its dependencies:

```bash
pip install -e ".[dev]"
```

## Generate data

Generate a full RPT (Reference Performance Test) dataset that chains cycling,
HPPT, and GITT blocks (repeated twice):

```bash
python example/generate.py
```

**Output:** `example/data/full_rpt/data.csv`

The generated CSV contains columns for time, current, voltage, temperature,
step number, and cycle number.

## Plot data

Once data has been generated, plot the full RPT data coloured by step type:

```bash
python example/process_and_plot_data.py
```

This script demonstrates:

- `iwdata.read.measurement_details` -- read a CSV into a time series and a
  per-step summary
- `iwdata.steps.annotate` -- copy step-level columns (e.g. "Step type") onto
  the time-series rows for colour-coded plotting
