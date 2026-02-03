"""
EIS step labeling.
"""

from __future__ import annotations

import pandas as pd
import polars as pl

from ionworksdata.logger import logger


def label_eis(
    steps: pd.DataFrame | pl.DataFrame,
    options: dict | None = None,
) -> pd.DataFrame:
    """
    Label EIS steps.

    Sets the "Label" column to "EIS" and the "Group number" column to the group
    number. For EIS, the group number increments from zero with each contiguous block
    of EIS steps.

    Parameters
    ----------
    steps : pd.DataFrame | pl.DataFrame
        A step summary dataframe (as returned by `iwdata.steps.summarize`)
    options : dict, optional
        Options for the labeling. No options are currently supported.

    Returns
    -------
    pd.DataFrame
        The dataframe with the updated "Label" and "Group number" columns.
    """
    # Convert to pandas if needed
    if isinstance(steps, pl.DataFrame):
        steps = steps.to_pandas()
    steps = steps.copy()

    # Find indices of all EIS steps
    eis_idxs = steps.index[steps["Step type"] == "EIS"]
    if len(eis_idxs) == 0:
        logger.warning(
            "Insufficient EIS steps found in the data, unable to add labels."
        )
        return steps

    steps.loc[eis_idxs, "Label"] = "EIS"

    # Label contiguous blocks of EIS steps with increasing group numbers
    if len(eis_idxs) > 0:
        # Get step numbers for EIS steps
        eis_step_nums = steps.loc[eis_idxs, "Step count"]
        # Find gaps in step numbers to identify separate groups
        gaps = eis_step_nums.diff() > 1
        # Cumulative sum of gaps gives group numbers
        steps.loc[eis_idxs, "Group number"] = gaps.cumsum()

    return steps
