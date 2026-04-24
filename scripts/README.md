# Conversion Scripts

This directory contains scripts to convert data from the old format to the new format with separate discharge/charge capacity and energy columns.

## What Changed?

### Old Format (main branch)

- Single `Capacity [A.h]` column (cumulative across all steps)
- Single `Energy [W.h]` column (cumulative across all steps)
- Steps have `Delta capacity [A.h]` column

### New Format

- `Discharge capacity [A.h]` and `Charge capacity [A.h]` columns
- `Discharge energy [W.h]` and `Charge energy [W.h]` columns
- Capacities and energies **reset to 0 at each step boundary**
- Steps have `Discharge capacity [A.h]` and `Charge capacity [A.h]` columns

## Scripts

### 1. `convert_to_new_format.py`

Main conversion script that converts time series and steps data from old to new format.

**Features:**

- Splits single capacity/energy columns into discharge/charge components
- **Calculates capacity/energy from scratch if columns don't exist**
- Applies step-based resets (each step starts at 0)
- Removes obsolete step columns (Start capacity, End capacity, etc.)
- Handles both individual files and entire directories
- Supports both `A.h` and `mA.h.cm-2` units
- Preserves all other columns unchanged

**Usage:**

Convert individual files:

```bash
python scripts/convert_to_new_format.py \\
    input_time_series.csv input_steps.csv \\
    output_time_series.csv output_steps.csv
```

Convert entire directory (recursively finds all `time_series.csv` and `steps.csv` files):

```bash
python scripts/convert_to_new_format.py input_dir/ output_dir/
```

**Example:**

```bash
# Convert a single measurement
python scripts/convert_to_new_format.py \\
    data/old/measurement_001/time_series.csv \\
    data/old/measurement_001/steps.csv \\
    data/new/measurement_001/time_series.csv \\
    data/new/measurement_001/steps.csv

# Convert all measurements in a directory
python scripts/convert_to_new_format.py data/old/ data/new/
```

### 2. `test_conversion.py`

Test script that demonstrates and verifies the conversion works correctly.

**Usage:**

```bash
python scripts/test_conversion.py
```

This will:

1. Create sample data in old format
2. Convert it to new format
3. Verify all columns are converted correctly
4. Check that step resets are applied
5. Display before/after comparison

## Conversion Details

### Time Series Conversion

The script handles two scenarios:

**Scenario 1: Existing Capacity/Energy Columns**

1. Reads the old `Capacity [A.h]` and/or `Energy [W.h]` columns
2. Calculates deltas between consecutive rows
3. Splits deltas based on current/power direction:
   - Positive current → discharge capacity increases
   - Negative current → charge capacity increases
4. Applies step resets: each step starts at capacity = 0
5. Removes old single-column format

**Scenario 2: No Capacity/Energy Columns**

1. Calculates capacity from current using trapezoidal integration
2. Calculates energy from power using trapezoidal integration
3. Splits into discharge/charge components
4. Applies step resets: each step starts at 0
5. Adds new discharge/charge columns

**Step Reset Algorithm:**

```python
# Detect step boundaries
step_changes = np.diff(step_numbers) != 0

# Find start of each step
step_start_indices = np.where(step_changes)[0]

# For each row, subtract the value at the start of its step
# This ensures each step starts at 0
```

### Steps Conversion

The script:

1. Reads the old `Delta capacity [A.h]` column (if present)
2. Determines step direction from:
   - `Mean current [A]` column (preferred)
   - `Type` column (if current not available)
3. Assigns delta to discharge or charge based on direction:
   - Positive current → discharge capacity
   - Negative current → charge capacity
4. **Removes obsolete columns:**
   - `Start capacity [A.h]` / `Start capacity [mA.h.cm-2]`
   - `End capacity [A.h]` / `End capacity [mA.h.cm-2]`
   - `Delta capacity [A.h]` / `Delta capacity [mA.h.cm-2]`

## Verification

After conversion, verify the results:

```bash
# Check time series columns
head -1 output_time_series.csv | grep "Discharge capacity"
head -1 output_time_series.csv | grep "Charge capacity"

# Check steps columns
head -1 output_steps.csv | grep "Discharge capacity"
head -1 output_steps.csv | grep "Charge capacity"

# Verify no old columns remain
head -1 output_time_series.csv | grep "Capacity \[A.h\]" && echo "ERROR: Old column still exists"
head -1 output_steps.csv | grep "Delta capacity" && echo "ERROR: Old column still exists"
```

## Important Notes

1. **Capacity/Energy Optional**: The script can handle files with or without existing capacity/energy columns. If they don't exist, they will be calculated from current/power.

2. **Step Column Recommended**: Time series data should have a `Step from cycler` or `Step number` column for step resets to work. Without it, capacity/energy will accumulate continuously.

3. **Obsolete Columns Removed**: The script automatically removes old-format step columns like `Start capacity`, `End capacity`, and `Delta capacity`.

4. **Direction Inference**: For steps data, if neither `Mean current` nor `Type` columns are available, all steps will be assumed to be discharge steps.

5. **Backup Your Data**: Always keep a backup of your original data before running the conversion.

6. **One-Way Conversion**: This converts from old → new format. Converting back would require additional logic.

7. **Cycle Capacity**: The script notes if cycle capacity columns need recalculation but doesn't automatically update them. You may need to regenerate these using the label module.

## Troubleshooting

### "Cannot determine step direction"

- Steps data is missing `Mean current` and `Type` columns
- Add these columns or all steps will be treated as discharge

### "no corresponding steps.csv"

- When converting directories, each `time_series.csv` must have a matching `steps.csv` in the same directory

### Values don't match expectations

- Run `test_conversion.py` to see a working example
- Check that your data has the expected old format columns
- Verify step numbers are present and correct

## Examples

See `test_conversion.py` for a complete working example that:

- Creates synthetic data in old format
- Converts it to new format
- Verifies the conversion
- Shows before/after comparison
