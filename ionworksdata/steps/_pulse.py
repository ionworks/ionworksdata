"""
Pulse step labeling (GITT/HPPT).
"""

from __future__ import annotations

import iwutil
import numpy as np
import pandas as pd
import polars as pl

from ionworksdata.logger import logger


def label_pulse(
    steps: pd.DataFrame | pl.DataFrame,
    options: dict | None = None,
) -> pd.DataFrame:
    """
    Label the "pulse" portion of the test.

    Sets the "Label" column to either "GITT" or "HPPT" and the "Group number" column
    to the group number. For pulse, the group number increments from zero with each
    "long" pulse (i.e. the step which changes the SOC). If all groups in a contiguous
    block of pulse steps have one rest step, the block is labelled as "GITT",
    otherwise it is labelled as "HPPT".

    Parameters
    ----------
    steps : pd.DataFrame | pl.DataFrame
        The steps dataframe.
    options : dict, optional
        Options for the labeling. The default is None, which uses the following
        default options:

        - "cell_metadata": a dictionary of cell metadata. Required.
        - "lower pulse capacity cutoff": the minimum percentage capacity required
          for a step to be considered a pulse step. Default is 1 / 100.
        - "upper pulse capacity cutoff": the maximum percentage capacity required
          for a step to be considered a pulse step. Default is 1 / 5.
        - "min pulses": the minimum number of pulses required for the data to be
          considered valid. Default is 1.
        - "current direction": the direction of the current for the pulse. Default
          is "discharge". Other options are "charge".

    Returns
    -------
    pd.DataFrame
        The steps dataframe with the pulse steps labeled.
    """
    # Convert to pandas if needed
    if isinstance(steps, pl.DataFrame):
        steps = steps.to_pandas()
    steps = steps.copy()

    # Set default options
    default_options = {
        "cell_metadata": "[required]",
        "lower pulse capacity cutoff": 1 / 100,
        "upper pulse capacity cutoff": 1 / 5,
        "min pulses": 1,
        "current direction": iwutil.OptionSpec(
            "discharge", ["charge", "delithiation", "lithiation"]
        ),
    }
    options_validated = iwutil.check_and_combine_options(default_options, options)
    assert options_validated is not None
    cell_metadata = options_validated.get("cell_metadata", None)
    if cell_metadata is None:
        raise ValueError("cell_metadata is required")
    Q = cell_metadata["Nominal cell capacity [A.h]"]
    lower_pulse_cutoff = options_validated["lower pulse capacity cutoff"]
    upper_pulse_cutoff = options_validated["upper pulse capacity cutoff"]
    current_direction = options_validated["current direction"]

    # Get constant current steps
    pulse_steps = steps[
        steps["Step type"].isin(
            ["Constant current discharge", "Constant current charge"]
        )
    ]

    # Filter by upper capacity cutoff
    discharge_cap = pulse_steps["Discharge capacity [A.h]"].abs()
    charge_cap = pulse_steps["Charge capacity [A.h]"].abs()
    dQ = abs(discharge_cap - charge_cap)
    capacity_filter = dQ < Q * upper_pulse_cutoff
    pulse_steps = pulse_steps[capacity_filter]

    if len(pulse_steps) <= options_validated["min pulses"]:
        logger.warning(
            "Insufficient pulse steps found in the data, unable to add labels."
        )
        return steps

    pulse_step_counts = pulse_steps["Step count"]

    # Find the "long" pulse steps. Recalculate dQ on the new pulse_steps dataframe.
    discharge_cap = pulse_steps["Discharge capacity [A.h]"].abs()
    charge_cap = pulse_steps["Charge capacity [A.h]"].abs()
    dQ = abs(discharge_cap - charge_cap)
    capacity_filter = dQ > Q * lower_pulse_cutoff
    long_pulse_steps = pulse_steps[capacity_filter]
    short_pulse_steps = pulse_steps[~capacity_filter]
    short_pulse_step_counts = short_pulse_steps["Step count"]

    # Filter "long" pulse steps by current direction
    if current_direction in ["discharge", "delithiation"]:
        long_pulse_steps_remove = long_pulse_steps[
            long_pulse_steps["Step type"] == "Constant current charge"
        ]
        long_pulse_steps = long_pulse_steps[
            long_pulse_steps["Step type"] == "Constant current discharge"
        ]
    elif current_direction in ["charge", "lithiation"]:
        long_pulse_steps_remove = long_pulse_steps[
            long_pulse_steps["Step type"] == "Constant current discharge"
        ]
        long_pulse_steps = long_pulse_steps[
            long_pulse_steps["Step type"] == "Constant current charge"
        ]
    else:
        raise ValueError("Undefined current direction")

    long_pulse_step_counts = long_pulse_steps["Step count"]
    long_pulse_step_counts_remove = long_pulse_steps_remove["Step count"]
    # Get rest steps
    rest_step_counts = steps[steps["Step type"] == "Rest"]["Step count"]

    # Only keep rests that are +/- 1 step away from "long" pulse steps (where we have
    # only kept steps in the correct direction) or short pulse steps (to catch HPPT)
    long_mask = (
        abs(rest_step_counts.values[:, None] - long_pulse_step_counts.values) == 1
    ).any(axis=1)
    short_mask = (
        abs(rest_step_counts.values[:, None] - short_pulse_step_counts.values) == 1
    ).any(axis=1)
    rest_mask = long_mask | short_mask
    rest_step_counts = rest_step_counts[rest_mask]

    # Combine all step numbers, sort them, and get unique values
    step_counts = (
        pd.concat([pulse_step_counts, rest_step_counts]).sort_values().unique()
    )
    # Remove steps that are in the remove list, so we can break up charge and discharge
    # blocks
    step_counts = step_counts[
        ~np.isin(step_counts, long_pulse_step_counts_remove.values)
    ]

    # Find start and stop idx of each contiguous block of step counts, only keep blocks
    # that have at least one "long" pulse step
    blocks: list[list[int]] = []
    current_block: list[int] = []
    for step in step_counts:
        if len(current_block) == 0:
            current_block.append(step)
        elif step == current_block[-1] + 1:
            current_block.append(step)
        else:
            if any(s in long_pulse_step_counts for s in current_block):
                blocks.append(current_block)
            current_block = [step]
    # Check final block
    if len(current_block) > 0 and any(
        s in long_pulse_step_counts for s in current_block
    ):
        blocks.append(current_block)
    # Update step_counts to only include steps in valid blocks
    step_counts = (
        pd.Series([s for block in blocks for s in block]).sort_values().unique()
    )

    # Add group numbers and label pulse type for each block
    for block in blocks:
        block_steps = steps.loc[block]
        block_step_counts = block_steps["Step count"]

        # Add a group number which increments after each "long" pulse step, and resets
        # to zero for each block
        groupseries = (
            steps.loc[block_step_counts, "Step count"]
            .isin(long_pulse_step_counts)
            .cumsum()
            - 1
        )
        block_steps["Group number"] = groupseries

        # Label pulse type: GITT if one rest per group, otherwise HPPT
        groups = block_steps.groupby("Group number")
        if all((group["Step type"] == "Rest").sum() == 1 for _, group in groups):
            label = "GITT"
        else:
            label = "HPPT"

        # If there are short pulses before the SOC change pulse ("long" pulse), this
        # is labelled as Group number -1. For example this happens during HPPT if the
        # short pulses come before the SOC change pulse.
        # For GITT, we don't keep any short pulses before the SOC change pulse (these
        # usually correspond to short charge steps at the start of the test).
        # For HPPT we do keep the short pulses before the SOC change pulse, but we add
        # one to the group number for all steps to account for the "-1" group.
        if label == "GITT":
            # Don't keep any short pulses before the SOC change pulse ("long" pulse)
            block_step_counts = block_steps[block_steps["Group number"] >= 0][
                "Step count"
            ]
        elif label == "HPPT":
            # Add one to the group number for all steps (the "-1" group is short
            # pulses before and SOC change pulse)
            if block_steps["Group number"].min() == -1:
                block_steps["Group number"] = block_steps["Group number"] + 1

        # Add the group number and label back to the steps dataframe
        steps.loc[block_step_counts, "Group number"] = block_steps["Group number"]
        steps.loc[block_step_counts, "Label"] = label

    return steps
