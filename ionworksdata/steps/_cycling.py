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
) -> pl.DataFrame:
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
    pl.DataFrame
        The steps dataframe with the cycling steps labeled.
    """
    if isinstance(steps, pd.DataFrame):
        steps_pl = pl.from_pandas(steps)
    else:
        steps_pl = steps.clone()

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

    discharge_cap = steps_pl["Discharge capacity [A.h]"].abs()
    charge_cap = steps_pl["Charge capacity [A.h]"].abs()
    dQ = (discharge_cap - charge_cap).abs()
    threshold = Q * options_validated["constant current threshold"]
    constant_current_steps = steps_pl.filter(dQ > threshold)

    if constant_current_steps.height == 0:
        logger.warning(
            "Insufficient constant current steps found in the data, "
            "unable to add labels."
        )
        return steps_pl

    cc_step_counts = constant_current_steps["Step count"]
    cc_discharge = constant_current_steps.filter(
        pl.col("Step type") == "Constant current discharge"
    )["Step count"]
    cc_charge = constant_current_steps.filter(
        pl.col("Step type") == "Constant current charge"
    )["Step count"]

    cv_steps = steps_pl.filter(
        pl.col("Step type").is_in(
            ["Constant voltage discharge", "Constant voltage charge"]
        )
    )
    cv_step_counts = cv_steps["Step count"]
    cc_arr = cc_step_counts.to_numpy()
    cv_arr = cv_step_counts.to_numpy()
    cv_mask = (np.abs(cv_arr[:, None] - cc_arr) == 1).any(axis=1)
    cv_step_counts = pl.Series(cv_step_counts.name, cv_arr[cv_mask])

    rest_steps = steps_pl.filter(pl.col("Step type") == "Rest")
    rest_step_counts = rest_steps["Step count"]
    rest_arr = rest_step_counts.to_numpy()
    rest_mask = (np.abs(rest_arr[:, None] - cc_arr) == 1).any(axis=1) | (
        np.abs(rest_arr[:, None] - cv_step_counts.to_numpy()) == 1
    ).any(axis=1)
    rest_step_counts = pl.Series(rest_step_counts.name, rest_arr[rest_mask])

    step_counts = (
        pl.concat([cc_step_counts, cv_step_counts, rest_step_counts]).unique().sort()
    )
    step_counts_arr = step_counts.to_numpy()

    step_counts_list = step_counts.to_list()
    steps_pl = steps_pl.with_columns(
        pl.when(pl.col("Step count").is_in(step_counts_list))
        .then(pl.lit("Cycling"))
        .otherwise(pl.col("Label"))
        .alias("Label")
    )

    group_nums = np.zeros(len(step_counts_arr), dtype=np.int64)
    cc_discharge_set = set(cc_discharge.to_list())
    cc_charge_set = set(cc_charge.to_list())
    for i, step in enumerate(step_counts_arr):
        if i > 0 and step != step_counts_arr[i - 1] + 1:
            group_nums[i:] = 0
        if (step in cc_discharge_set or step in cc_charge_set) and (
            i > 0 and step == step_counts_arr[i - 1] + 1
        ):
            group_nums[i:] += 1

    step_to_group = dict(zip(step_counts_arr.tolist(), group_nums.tolist()))
    group_series = pl.Series(
        "Group number",
        [step_to_group.get(s, None) for s in steps_pl["Step count"].to_list()],
    )
    steps_pl = steps_pl.with_columns(group_series)
    return steps_pl
