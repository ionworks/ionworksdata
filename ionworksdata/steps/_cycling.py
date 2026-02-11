"""
Cycling step labeling.
"""

from __future__ import annotations

import iwutil
import numpy as np
import pandas as pd
import polars as pl

from ionworksdata.logger import logger


def label_cycling(
    steps: pd.DataFrame | pl.DataFrame,
    options: dict | None = None,
) -> pd.DataFrame:
    """
    Label the "cycling" portion of the test.

    Cycling is defined as constant current (and optionally constant voltage) steps
    where the capacity is greater than a certain percentage of the nominal cell
    capacity. Sets the "Label" column to "Cycling" and the "Group number" column to
    the group number. For cycling, a "group" is all the steps in one direction
    (including rest) - i.e. constant current discharge + rest is one group, constant
    current charge + rest is another group. The group number is incremented with each
    new group.

    Parameters
    ----------
    steps : pd.DataFrame | pl.DataFrame
        The steps dataframe.
    options : dict, optional
        Options for the labeling. The default is None, which uses the following
        default options:

        - "cell_metadata": a dictionary of cell metadata. Required.
        - "constant current threshold": the threshold for the capacity to be
          considered a constant current step. Default is 1/2.

    Returns
    -------
    pd.DataFrame
        The steps dataframe with the cycling steps labeled.
    """
    # Convert to pandas for this function's logic which relies on:
    # - .loc indexing with boolean masks and step count slices
    # - .values attribute for numpy broadcasting operations
    # - pd.concat for combining Series with sort_values() and unique()
    if isinstance(steps, pl.DataFrame):
        steps = steps.to_pandas()
    steps = steps.copy()

    # Set default options
    default_options = {
        "cell_metadata": "[required]",
        "constant current threshold": 1 / 2,
    }
    options_validated = iwutil.check_and_combine_options(
        default_options, options, filter_unknown=True
    )
    assert options_validated is not None
    cell_metadata = options_validated.get("cell_metadata", None)
    if cell_metadata is None:
        raise ValueError("cell_metadata is required")
    Q = cell_metadata["Nominal cell capacity [A.h]"]

    # Filter by total capacity (sum of discharge and charge for each step)
    discharge_cap = steps["Discharge capacity [A.h]"].abs()
    charge_cap = steps["Charge capacity [A.h]"].abs()
    dQ = abs(discharge_cap - charge_cap)
    capacity_filter = dQ > Q * options_validated["constant current threshold"]
    constant_current_steps = steps[capacity_filter]

    if len(constant_current_steps) == 0:
        logger.warning(
            "Insufficient constant current steps found in the data, "
            "unable to add labels."
        )
        return steps

    cc_step_counts = constant_current_steps["Step count"]
    cc_discharge_step_counts = constant_current_steps[
        constant_current_steps["Step type"] == "Constant current discharge"
    ]["Step count"]
    cc_charge_step_counts = constant_current_steps[
        constant_current_steps["Step type"] == "Constant current charge"
    ]["Step count"]

    # Get constant voltage steps
    cv_step_counts = steps[
        steps["Step type"].isin(
            ["Constant voltage discharge", "Constant voltage charge"]
        )
    ]["Step count"]
    # Only keep CV steps that are +/- 1 step away from CC steps
    cv_mask = (abs(cv_step_counts.values[:, None] - cc_step_counts.values) == 1).any(
        axis=1
    )
    cv_step_counts = cv_step_counts[cv_mask]

    # Get rest steps
    rest_step_counts = steps[steps["Step type"] == "Rest"]["Step count"]

    # Only keep rest steps that are +/- 1 step away from CC or CV steps
    rest_mask = (
        abs(rest_step_counts.values[:, None] - cc_step_counts.values) == 1
    ).any(axis=1) | (
        abs(rest_step_counts.values[:, None] - cv_step_counts.values) == 1
    ).any(axis=1)
    rest_step_counts = rest_step_counts[rest_mask]

    # Combine all step numbers, sort them, and get unique values
    step_counts = (
        pd.concat([cc_step_counts, cv_step_counts, rest_step_counts])
        .sort_values()
        .unique()
    )

    # Label as cycling steps
    steps.loc[step_counts, "Label"] = "Cycling"

    # Add a group number to each cycling step, which increments each time we see a
    # constant current (dis)charge step and resets when steps are not contiguous
    group_nums = np.zeros(len(step_counts))
    for i, step in enumerate(step_counts):
        # Reset group number if steps are not contiguous
        if i > 0 and step != step_counts[i - 1] + 1:
            group_nums[i:] = 0
        # Increment group number on (dis)charge steps, except first step after reset
        if (
            step in cc_discharge_step_counts.values
            or step in cc_charge_step_counts.values
        ) and (i > 0 and step == step_counts[i - 1] + 1):
            group_nums[i:] += 1
    steps.loc[step_counts, "Group number"] = group_nums.astype(np.int64)

    return steps
