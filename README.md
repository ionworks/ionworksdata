# Ionworks data

Package to extract, transform, and validate battery data for use in Ionworks software.

## Package overview

`ionworksdata` is organized into the following modules:

- Read: read raw data from files
- Transform: transform time series data into Ionworks-compatible format
- Steps: label steps for easy processing or visualization

## Data format

The data is processed to be compatible with the Ionworks software. Documentation for the data format can be found in the [Knowledge base](https://www.notion.so/ionworks/Ionworks-Data-Data-management-tools-1ce7637d96ae801da6a1eb0e81669cac?pvs=4).

# Data processing

## Processing time series data

Time series data can be extracted from a cycler file using the `read.time_series` function, which takes the filename and optionally the reader name (e.g. `csv`, `biologic_mpt`, `maccor`, `neware`, `repower`). If the reader is not specified, it will be automatically detected from the file. The function returns a Polars DataFrame.

```python
# With explicit reader
data = iwdata.read.time_series("path/to/file.mpt", "biologic_mpt")

# With auto-detection (reader is optional)
data = iwdata.read.time_series("path/to/file.mpt")
```

The function automatically performs several processing steps and adds columns to the output:

### Data Processing Steps

1. **Reader-specific processing** (varies by reader):

   - Column renaming to standardized names (e.g., "Voltage" → "Voltage [V]")
   - Numeric coercion (removing thousands separators, converting strings to numbers)
   - Dropping message/error rows (for some readers)
   - Parsing timestamp columns and computing time if needed
   - Converting time units (e.g., hours to seconds)
   - Fixing unsigned current (if current is always positive, negate during charge)
   - Validating and fixing decreasing times (if `time_offset_fix` option is set)

2. **Standard data processing** (applied to all readers):

   - Removing rows with null values in current or voltage columns
   - Converting numeric columns to float64
   - Resetting time to start at zero
   - Offsetting duplicate timestamps by a small amount (1e-6 seconds) to preserve all data points
   - Setting discharge current to be positive (charge current remains negative)

3. **Post-processing**:
   - Adding `Step count`: Cumulative step count (always present)
   - Adding `Discharge capacity [A.h]`: Discharge capacity in ampere-hours (always present)
   - Adding `Charge capacity [A.h]`: Charge capacity in ampere-hours (always present)
   - Adding `Discharge energy [W.h]`: Discharge energy in watt-hours (always present)
   - Adding `Charge energy [W.h]`: Charge energy in watt-hours (always present)

### Output Columns

The output always includes:

- `Time [s]`: Time in seconds
- `Current [A]`: Current in amperes
- `Voltage [V]`: Voltage in volts
- `Step count`: Cumulative step count
- `Discharge capacity [A.h]`: Discharge capacity in ampere-hours
- `Charge capacity [A.h]`: Charge capacity in ampere-hours
- `Discharge energy [W.h]`: Discharge energy in watt-hours
- `Charge energy [W.h]`: Charge energy in watt-hours

Optional columns (if present in source data):

- `Step from cycler`: Step number from cycler file
- `Cycle from cycler`: Cycle number from cycler file
- `Temperature [degC]`: Temperature in degrees Celsius
- `Frequency [Hz]`: Frequency in hertz

For information on the expected and returned columns, see the reader documentation. Additional columns can be added by passing a dictionary to the `extra_column_mappings` argument.

```python
data = iwdata.read.time_series(
    "path/to/file.mpt", "biologic_mpt", extra_column_mappings={"My new column": "Old column name"}
)
```

## Processing step data

Given a processed time series data, the step summary data can be extracted as follows:

```python
steps = iwdata.steps.summarize(data)
```

This function identifies distinct steps within battery cycling data by detecting changes in the "Step count" column (which must be present in the input data). For each identified step, it extracts and calculates relevant metrics (voltage, current, capacity, energy, etc.) and determines the step type.

The output always includes:

- `Cycle count`: Cumulative cycle count (defaults to 0 if no cycle information is available)
- `Cycle from cycler`: Cycle number from cycler file (only if provided in the input data)
- `Discharge capacity [A.h]`: Discharge capacity for the step
- `Charge capacity [A.h]`: Charge capacity for the step
- `Discharge energy [W.h]`: Discharge energy for the step
- `Charge energy [W.h]`: Charge energy for the step
- `Step from cycler`: Step number from cycler file (only if provided in the input data)

**Note:** The `step_column` and `cycle_column` parameters have been removed. The function now always uses "Step count" for step identification and "Cycle from cycler" (if available) for cycle tracking.

Alternatively, the time series and step data can be extracted together using the `read.time_series_and_steps` function.

```python
# With explicit reader
data, steps = iwdata.read.time_series_and_steps("path/to/file.mpt", "biologic_mpt")

# With auto-detection (reader is optional)
data, steps = iwdata.read.time_series_and_steps("path/to/file.mpt")
```

## Labeling steps

Steps can be labeled using the functions in the `steps` module. For example, the following code labels the steps as cycling and pulse (charge and discharge).

```python
options = {"cell_metadata": {"Nominal cell capacity [A.h]": 5}}
steps = iwdata.steps.label_cycling(steps, options)
for direction in ["charge", "discharge"]:
    options["current direction"] = direction
    steps = iwdata.steps.label_pulse(steps, options)
```

## Measurement details

The function `ionworksdata.read.measurement_details` can be used to return a `measurement_details` dictionary, which has keys "measurement", "time_series", and "steps". You need to first create the measurement dictionary (which contains details about the test, such as its name), and then pass it to the function along with the reader name and filename, and any additional arguments. The function will return the updated measurement details dictionary, which includes information extracted from the file, such as the start time, and the cycler name. This function also automatically labels the steps with some sensible defaults (custom labels can be added by passing a list of dictionaries to the `labels` argument).

```python3
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
