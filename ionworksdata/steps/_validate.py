"""
Step validation functions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl


def validate(steps: pd.DataFrame | pl.DataFrame, label_name: str) -> bool:
    """
    Validate the steps dataframe for a given label.

    Parameters
    ----------
    steps : pd.DataFrame | pl.DataFrame
        The steps dataframe to validate.
    label_name : str
        The name of the label to validate.

    Returns
    -------
    bool
        True if the steps dataframe is valid for the given label, False otherwise.
    """
    # Convert to pandas for validation logic which relies on:
    # - .iloc indexing for sequential access
    # - .values attribute for numpy array operations
    # - Complex iteration patterns with pd.DataFrame construction
    if isinstance(steps, pl.DataFrame):
        steps_pd = steps.to_pandas()
    else:
        steps_pd = steps
    label_steps = steps_pd[steps_pd["Label"] == label_name]

    # Check valid group number transitions (can increment by 0, 1 or reset to 0)
    group_nums = label_steps["Group number"].values
    valid_transitions = (
        (group_nums[1:] == group_nums[:-1])
        | (group_nums[1:] == group_nums[:-1] + 1)
        | (group_nums[1:] == 0)
    )
    if not np.all(valid_transitions):
        return False

    # Split into blocks where group number changes
    blocks = []
    if len(label_steps) > 0:
        current_block = [label_steps.iloc[0]]
        for _, step in label_steps.iloc[1:].iterrows():
            if step["Group number"] == current_block[-1]["Group number"]:
                current_block.append(step)
            else:
                blocks.append(pd.DataFrame(current_block))
                current_block = [step]
        blocks.append(pd.DataFrame(current_block))

    # Check each block
    for block in blocks:
        # Check step counts are contiguous or all the same (As is the case for EIS)
        step_counts = block["Step count"].values
        if not (
            np.all(step_counts[1:] == step_counts[:-1] + 1)
            or np.all(step_counts == step_counts[0])
        ):
            return False

        # If cycle count exists, check it's constant within block
        if "Cycle count" in block.columns:
            if not block["Cycle count"].isna().all() and not np.all(
                block["Cycle count"] == block["Cycle count"].iloc[0]
            ):
                return False

    return True
