from __future__ import annotations

import polars as pl

import ionworksdata as iwdata


def get_cycle_metrics(
    steps: pl.DataFrame,
    options: dict | None = None,
) -> pl.DataFrame:
    """
    Compute cycle-based metrics from a steps DataFrame.

    This function takes a steps DataFrame (typically from Steps.get_step_types_for())
    and calculates key battery performance metrics for each cycle, including
    coulombic efficiency, energy efficiency, capacity retention, and capacity fade.

    Parameters
    ----------
    steps : pl.DataFrame
        A DataFrame containing step-level data with columns including:
        - "Cycle count": Cycle identifier for each step
        - "Discharge capacity [A.h]" or "Discharge capacity [mA.h.cm-2]"
        - "Charge capacity [A.h]" or "Charge capacity [mA.h.cm-2]"
        - Optionally "Discharge energy [W.h]" and "Charge energy [W.h]"
        - Optionally "Mean current [A]", "Duration [s]" for current calculations
          (positive current = discharge, negative current = charge)
        - Optionally "Min voltage [V]", "Max voltage [V]" for voltage range
        - Optionally "Mean temperature [degC]" for temperature
    options : dict, optional
        Options for the calculation. The default is None, which uses:
        - "current units": "total" (options: "total" for A.h, "density" for mA.h.cm-2)

    Returns
    -------
    pl.DataFrame
        A DataFrame with one row per cycle containing:

        **Capacity metrics:**
        - "Cycle count": The cycle index (starting from 0)
        - "Discharge capacity [A.h]": Total discharge capacity for the cycle
        - "Charge capacity [A.h]": Total charge capacity for the cycle
        - "Coulombic efficiency": Discharge capacity / Charge capacity
        - "Capacity retention": Discharge capacity / First cycle discharge capacity
        - "Capacity fade": 1 - Capacity retention

        **Energy metrics (if energy columns available):**
        - "Discharge energy [W.h]": Total discharge energy
        - "Charge energy [W.h]": Total charge energy
        - "Energy efficiency": Discharge energy / Charge energy
        - "Energy retention": Discharge energy / First cycle discharge energy
        - "Energy fade": 1 - Energy retention

        **Current metrics (if Step type and Mean current columns available):**
        - "Mean discharge current [A]": Duration-weighted mean current during discharge
        - "Mean charge current [A]": Duration-weighted mean current during charge

        **Voltage metrics (if voltage columns available):**
        - "Min voltage [V]": Minimum voltage during the cycle
        - "Max voltage [V]": Maximum voltage during the cycle

        **Time and duration metrics (if time/duration columns available):**
        - "Start time [s]": Start time of the cycle
        - "Cycle duration [s]": Total duration of the cycle
        - "Discharge duration [s]": Total duration of discharge steps
        - "Charge duration [s]": Total duration of charge steps

        **Throughput metrics (cumulative):**
        - "Capacity throughput [A.h]": Cumulative abs capacity (discharge + charge)
        - "Energy throughput [W.h]": Cumulative abs energy (if energy available)

        **Temperature metrics (if temperature column available):**
        - "Mean temperature [degC]": Duration-weighted mean temperature

    Notes
    -----
    - Efficiencies are returned as ratios (not percentages), e.g., 0.99 for 99%
    - Division by zero cases (e.g., zero charge capacity) result in None values
    - Capacity/energy retention/fade are calculated relative to the first cycle's
      discharge capacity/energy
    - Current values are duration-weighted averages across relevant steps
    """
    _, capacity_units = iwdata.util.get_current_and_capacity_units(options)

    # Validate required columns
    discharge_cap_col = f"Discharge capacity [{capacity_units}]"
    charge_cap_col = f"Charge capacity [{capacity_units}]"

    if "Cycle count" not in steps.columns:
        raise ValueError("Steps DataFrame must contain 'Cycle count' column")
    if discharge_cap_col not in steps.columns:
        raise ValueError(f"Steps DataFrame must contain '{discharge_cap_col}' column")
    if charge_cap_col not in steps.columns:
        raise ValueError(f"Steps DataFrame must contain '{charge_cap_col}' column")

    # Check for optional columns
    has_energy = (
        "Discharge energy [W.h]" in steps.columns
        and "Charge energy [W.h]" in steps.columns
    )
    has_current = "Mean current [A]" in steps.columns
    has_duration = "Duration [s]" in steps.columns
    has_voltage = (
        "Min voltage [V]" in steps.columns and "Max voltage [V]" in steps.columns
    )
    has_temperature = "Mean temperature [degC]" in steps.columns
    has_start_time = "Start time [s]" in steps.columns

    # Build aggregation expressions for basic capacity metrics
    agg_exprs = [
        pl.col(discharge_cap_col).sum().alias(discharge_cap_col),
        pl.col(charge_cap_col).sum().alias(charge_cap_col),
    ]

    # Energy aggregations
    if has_energy:
        agg_exprs.extend(
            [
                pl.col("Discharge energy [W.h]").sum().alias("Discharge energy [W.h]"),
                pl.col("Charge energy [W.h]").sum().alias("Charge energy [W.h]"),
            ]
        )

    # Voltage aggregations
    if has_voltage:
        agg_exprs.extend(
            [
                pl.col("Min voltage [V]").min().alias("Min voltage [V]"),
                pl.col("Max voltage [V]").max().alias("Max voltage [V]"),
            ]
        )

    # Duration aggregation
    if has_duration:
        agg_exprs.append(pl.col("Duration [s]").sum().alias("Cycle duration [s]"))

    # Start time aggregation
    if has_start_time:
        agg_exprs.append(pl.col("Start time [s]").min().alias("Start time [s]"))

    # Group by cycle and aggregate
    cycle_data = steps.group_by("Cycle count").agg(agg_exprs).sort("Cycle count")

    # Rename Cycle count to Cycle number for output
    cycle_data = cycle_data.rename({"Cycle count": "Cycle number"})

    # Calculate coulombic efficiency (discharge / charge)
    cycle_data = cycle_data.with_columns(
        pl.when(pl.col(charge_cap_col) > 0)
        .then(pl.col(discharge_cap_col) / pl.col(charge_cap_col))
        .otherwise(None)
        .alias("Coulombic efficiency")
    )

    # Calculate energy efficiency if energy columns are available
    if has_energy:
        cycle_data = cycle_data.with_columns(
            pl.when(pl.col("Charge energy [W.h]") > 0)
            .then(pl.col("Discharge energy [W.h]") / pl.col("Charge energy [W.h]"))
            .otherwise(None)
            .alias("Energy efficiency")
        )

    # Calculate capacity retention and fade relative to first cycle
    first_cycle_discharge = cycle_data.get_column(discharge_cap_col)[0]

    if first_cycle_discharge is not None and first_cycle_discharge > 0:
        cycle_data = cycle_data.with_columns(
            (pl.col(discharge_cap_col) / first_cycle_discharge).alias(
                "Capacity retention"
            )
        )
        cycle_data = cycle_data.with_columns(
            (1.0 - pl.col("Capacity retention")).alias("Capacity fade")
        )
    else:
        cycle_data = cycle_data.with_columns(
            [
                pl.lit(None).cast(pl.Float64).alias("Capacity retention"),
                pl.lit(None).cast(pl.Float64).alias("Capacity fade"),
            ]
        )

    # Calculate energy retention and fade relative to first cycle
    if has_energy:
        first_cycle_energy = cycle_data.get_column("Discharge energy [W.h]")[0]

        if first_cycle_energy is not None and first_cycle_energy > 0:
            cycle_data = cycle_data.with_columns(
                (pl.col("Discharge energy [W.h]") / first_cycle_energy).alias(
                    "Energy retention"
                )
            )
            cycle_data = cycle_data.with_columns(
                (1.0 - pl.col("Energy retention")).alias("Energy fade")
            )
        else:
            cycle_data = cycle_data.with_columns(
                [
                    pl.lit(None).cast(pl.Float64).alias("Energy retention"),
                    pl.lit(None).cast(pl.Float64).alias("Energy fade"),
                ]
            )

    # Calculate current and duration metrics by current sign
    # Positive current = discharge, negative current = charge
    if has_current and has_duration:
        # Filter discharge steps (positive current) and compute duration-weighted mean
        discharge_steps = steps.filter(pl.col("Mean current [A]") > 0)
        if discharge_steps.height > 0:
            discharge_agg = discharge_steps.group_by("Cycle count").agg(
                [
                    (
                        (pl.col("Mean current [A]") * pl.col("Duration [s]")).sum()
                        / pl.col("Duration [s]").sum()
                    ).alias("Mean discharge current [A]"),
                    pl.col("Duration [s]").sum().alias("Discharge duration [s]"),
                ]
            )
            cycle_data = cycle_data.join(
                discharge_agg.rename({"Cycle count": "Cycle number"}),
                on="Cycle number",
                how="left",
            )
        else:
            cycle_data = cycle_data.with_columns(
                [
                    pl.lit(None).cast(pl.Float64).alias("Mean discharge current [A]"),
                    pl.lit(None).cast(pl.Float64).alias("Discharge duration [s]"),
                ]
            )

        # Filter charge steps (negative current) and compute duration-weighted mean
        charge_steps = steps.filter(pl.col("Mean current [A]") < 0)
        if charge_steps.height > 0:
            charge_agg = charge_steps.group_by("Cycle count").agg(
                [
                    (
                        (pl.col("Mean current [A]") * pl.col("Duration [s]")).sum()
                        / pl.col("Duration [s]").sum()
                    ).alias("Mean charge current [A]"),
                    pl.col("Duration [s]").sum().alias("Charge duration [s]"),
                ]
            )
            cycle_data = cycle_data.join(
                charge_agg.rename({"Cycle count": "Cycle number"}),
                on="Cycle number",
                how="left",
            )
        else:
            cycle_data = cycle_data.with_columns(
                [
                    pl.lit(None).cast(pl.Float64).alias("Mean charge current [A]"),
                    pl.lit(None).cast(pl.Float64).alias("Charge duration [s]"),
                ]
            )

    # Calculate duration-weighted mean temperature
    if has_temperature and has_duration:
        temp_agg = steps.group_by("Cycle count").agg(
            [
                (
                    (pl.col("Mean temperature [degC]") * pl.col("Duration [s]")).sum()
                    / pl.col("Duration [s]").sum()
                ).alias("Mean temperature [degC]"),
            ]
        )
        cycle_data = cycle_data.join(
            temp_agg.rename({"Cycle count": "Cycle number"}),
            on="Cycle number",
            how="left",
        )

    # Calculate capacity throughput (cumulative absolute capacity)
    cycle_data = cycle_data.with_columns(
        (pl.col(discharge_cap_col) + pl.col(charge_cap_col))
        .cum_sum()
        .alias(f"Capacity throughput [{capacity_units}]")
    )

    # Calculate energy throughput (cumulative absolute energy)
    if has_energy:
        cycle_data = cycle_data.with_columns(
            (pl.col("Discharge energy [W.h]") + pl.col("Charge energy [W.h]"))
            .cum_sum()
            .alias("Energy throughput [W.h]")
        )

    return cycle_data
