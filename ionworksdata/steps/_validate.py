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
    if isinstance(steps, pd.DataFrame):
        steps_pl = pl.from_pandas(steps)
    else:
        steps_pl = steps

    label_steps = steps_pl.filter(pl.col("Label") == label_name)
    if label_steps.height == 0:
        return True

    group_nums = label_steps["Group number"].to_numpy()
    valid_transitions = (
        (group_nums[1:] == group_nums[:-1])
        | (group_nums[1:] == group_nums[:-1] + 1)
        | (group_nums[1:] == 0)
    )
    if not np.all(valid_transitions):
        return False

    group_col = label_steps["Group number"]
    boundaries = (group_col != group_col.shift(1)).fill_null(True)
    block_ids = boundaries.cum_sum()
    blocks = label_steps.with_columns(block_ids.alias("__block_id"))

    for block_id in blocks["__block_id"].unique().to_list():
        block = blocks.filter(pl.col("__block_id") == block_id)
        step_counts = block["Step count"].to_numpy()
        if not (
            np.all(step_counts[1:] == step_counts[:-1] + 1)
            or np.all(step_counts == step_counts[0])
        ):
            return False
        if "Cycle count" in block.columns:
            cycle = block["Cycle count"]
            cycle_nn = cycle.drop_nulls()
            if cycle_nn.len() > 0 and not (cycle_nn == cycle_nn[0]).all():
                return False

    return True
