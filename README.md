# Ionworks Data Processing

A library for processing experimental battery data into a common format for use in Ionworks software.

## Overview

**Ionworks Data Processing** (`ionworksdata`) provides readers for cycler file formats (Maccor, Biologic, Neware, Novonix, Repower, CSV, and more), transforms time series data into a standardized format, and summarizes and labels steps for analysis or use in other Ionworks tools (e.g. the [Ionworks pipeline](https://pipeline.docs.ionworks.com/) and [ionworks-schema](https://github.com/ionworks/ionworks-schema)).

Full API and usage details are in the [Ionworks Data Processing documentation](https://data.docs.ionworks.com/).

## Package structure

- **`read`** — Read raw data from files: `time_series`, `time_series_and_steps`, `measurement_details`; readers include `biologic`, `biologic_mpt`, `maccor`, `neware`, `novonix`, `repower`, `csv`, and others (reader is auto-detected when not specified).
- **`transform`** — Transform time series into Ionworks-compatible form (step count, cycle count, capacity, energy, etc.).
- **`steps`** — Summarize time series into step-level data and label steps (cycling, pulse, EIS) for processing or visualization.
- **`load`** — Load processed data for use in other Ionworks software (`DataLoader`, `OCPDataLoader`).

## Installation

```bash
pip install ionworksdata
```

## Quick start

### Processing time series data

Extract time series data from a cycler file with `read.time_series`. The reader can be specified explicitly or auto-detected. The function returns a Polars DataFrame.

```python
import ionworksdata as iwdata

# With explicit reader
data = iwdata.read.time_series("path/to/file.mpt", "biologic_mpt")

# With auto-detection (reader is optional)
data = iwdata.read.time_series("path/to/file.mpt")
```

The function automatically performs several processing steps and adds columns to the output.

#### Data processing steps

1. **Reader-specific processing** (varies by reader):
   - Column renaming to standardized names (e.g. "Voltage" → "Voltage [V]")
   - Numeric coercion (removing thousands separators, converting strings to numbers)
   - Dropping message/error rows (for some readers)
   - Parsing timestamp columns and computing time if needed
   - Converting time units (e.g. hours to seconds)
   - Fixing unsigned current (if current is always positive, negate during charge)
   - Validating and fixing decreasing times (if `time_offset_fix` option is set)

2. **Standard data processing** (applied to all readers):
   - Removing rows with null values in current or voltage columns
   - Converting numeric columns to float64
   - Resetting time to start at zero
   - Offsetting duplicate timestamps by a small amount (1e-6 s) to preserve all data points
   - Setting discharge current to be positive (charge current remains negative)

3. **Post-processing**:
   - Adding `Step count`, `Cycle count`, `Discharge capacity [A.h]`, `Charge capacity [A.h]`, `Discharge energy [W.h]`, `Charge energy [W.h]`

#### Output columns

| Column | Description |
|--------|-------------|
| `Time [s]` | Time in seconds |
| `Current [A]` | Current in amperes |
| `Voltage [V]` | Voltage in volts |
| `Step count` | Cumulative step count (always present) |
| `Cycle count` | Cumulative cycle count, defaults to 0 if no cycle information (always present) |
| `Discharge capacity [A.h]` | Discharge capacity in ampere-hours (always present) |
| `Charge capacity [A.h]` | Charge capacity in ampere-hours (always present) |
| `Discharge energy [W.h]` | Discharge energy in watt-hours (always present) |
| `Charge energy [W.h]` | Charge energy in watt-hours (always present) |
| `Step from cycler` | Step number from cycler file (if provided) |
| `Cycle from cycler` | Cycle number from cycler file (if provided) |
| `Temperature [degC]` | Temperature in degrees Celsius (if provided) |
| `Frequency [Hz]` | Frequency in hertz (if provided) |

For expected and returned columns per reader, see the [API documentation](https://data.docs.ionworks.com/). Extra columns can be mapped via `extra_column_mappings`:

```python
data = iwdata.read.time_series(
    "path/to/file.mpt", "biologic_mpt",
    extra_column_mappings={"Old column name": "My new column"},
)
```

### Processing step data

From processed time series, step summary data is obtained with `steps.summarize`:

```python
steps = iwdata.steps.summarize(data)
```

This detects steps from the `Step count` column and computes metrics per step. The output always includes:

| Column | Description |
|--------|-------------|
| `Cycle count` | Cumulative cycle count (defaults to 0 if no cycle information) |
| `Cycle from cycler` | Cycle number from cycler file (only if provided in input) |
| `Discharge capacity [A.h]` | Discharge capacity for the step |
| `Charge capacity [A.h]` | Charge capacity for the step |
| `Discharge energy [W.h]` | Discharge energy for the step |
| `Charge energy [W.h]` | Charge energy for the step |
| `Step from cycler` | Step number from cycler file (only if provided in input) |

Additional per-step columns include start/end time and index, start/end/min/max/mean/std for voltage and current, duration, step type, and (after labeling) cycle-level capacity and energy. See the [API documentation](https://data.docs.ionworks.com/) for the full list.

**Note:** Step identification uses `Step count` and, when available, `Cycle from cycler` for cycle tracking.

Alternatively, get time series and steps in one call:

```python
# With explicit reader
data, steps = iwdata.read.time_series_and_steps("path/to/file.mpt", "biologic_mpt")

# With auto-detection (reader is optional)
data, steps = iwdata.read.time_series_and_steps("path/to/file.mpt")
```

### Labeling steps

Steps can be labeled using the `steps` module (e.g. cycling, pulse, and EIS):

```python
options = {"cell_metadata": {"Nominal cell capacity [A.h]": 5}}
steps = iwdata.steps.label_cycling(steps, options)
for direction in ["charge", "discharge"]:
    options["current direction"] = direction
    steps = iwdata.steps.label_pulse(steps, options)
steps = iwdata.steps.label_eis(steps)
```

### Measurement details

`read.measurement_details` returns a dictionary with `measurement`, `time_series`, and `steps`. Pass the file path, a measurement dict (e.g. test name), and optionally the reader and options; the function fills in time series and steps and updates the measurement dict (e.g. cycler name, start time). Steps are labeled with default labels unless you pass a custom `labels` list:

```python
measurement = {"name": "My test"}
measurement_details = iwdata.read.measurement_details(
    "path/to/file.mpt",
    measurement,
    "biologic_mpt",
    options={"cell_metadata": {"Nominal cell capacity [A.h]": 5}},
)
measurement = measurement_details["measurement"]
time_series = measurement_details["time_series"]
steps = measurement_details["steps"]
```

## Data format

Processed data follows the format expected by Ionworks software. Column names, units, and conventions are described in the [Ionworks Data Processing documentation](https://data.docs.ionworks.com/).

## Resources

- [Ionworks Data Processing documentation](https://data.docs.ionworks.com/) — API reference, readers, and data format.
- [Ionworks Pipeline documentation](https://pipeline.docs.ionworks.com/) — Using processed data in pipelines.
- [Ionworks documentation](https://docs.ionworks.com) — Platform and product overview.

