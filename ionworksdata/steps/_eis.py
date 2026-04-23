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
) -> pl.DataFrame:
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
    pl.DataFrame
        The dataframe with the updated "Label" and "Group number" columns.
    """
    if isinstance(steps, pd.DataFrame):
        steps_pl = pl.from_pandas(steps)
    else:
        steps_pl = steps.clone()

    eis_steps = steps_pl.filter(pl.col("Step type") == "EIS")
    if eis_steps.height == 0:
        logger.warning(
            "Insufficient EIS steps found in the data, unable to add labels."
        )
        return steps_pl

    eis_step_nums = eis_steps["Step count"]
    gaps = eis_step_nums.diff() > 1
    group_nums = gaps.cum_sum()
    step_to_group = dict(
        zip(eis_step_nums.to_list(), group_nums.to_list(), strict=False)
    )
    orig_groups = (
        steps_pl["Group number"].to_list()
        if "Group number" in steps_pl.columns
        else [None] * steps_pl.height
    )

    steps_pl = steps_pl.with_columns(
        pl.when(pl.col("Step type") == "EIS")
        .then(pl.lit("EIS"))
        .otherwise(pl.col("Label"))
        .alias("Label")
    )
    group_series = pl.Series(
        "Group number",
        [
            step_to_group.get(s, orig_groups[i])
            for i, s in enumerate(steps_pl["Step count"].to_list())
        ],
    )
    steps_pl = steps_pl.with_columns(group_series)
    return steps_pl
