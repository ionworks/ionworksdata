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
) -> pl.DataFrame:
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
    pl.DataFrame
        The steps dataframe with the pulse steps labeled.
    """
    if isinstance(steps, pd.DataFrame):
        steps_pl = pl.from_pandas(steps)
    else:
        steps_pl = steps.clone()

    default_options = {
        "cell_metadata": "[required]",
        "lower pulse capacity cutoff": 1 / 100,
        "upper pulse capacity cutoff": 1 / 5,
        "min pulses": 1,
        "current direction": iwutil.OptionSpec(
            "discharge", ["charge", "delithiation", "lithiation"]
        ),
    }
    options_validated = iwutil.check_and_combine_options(
        default_options, options, filter_unknown=True
    )
    assert options_validated is not None
    cell_metadata = options_validated.get("cell_metadata", None)
    if cell_metadata is None:
        raise ValueError("cell_metadata is required")
    Q = cell_metadata["Nominal cell capacity [A.h]"]
    lower_pulse_cutoff = options_validated["lower pulse capacity cutoff"]
    upper_pulse_cutoff = options_validated["upper pulse capacity cutoff"]
    current_direction = options_validated["current direction"]

    pulse_steps = steps_pl.filter(
        pl.col("Step type").is_in(
            ["Constant current discharge", "Constant current charge"]
        )
    )
    discharge_cap = pulse_steps["Discharge capacity [A.h]"].abs()
    charge_cap = pulse_steps["Charge capacity [A.h]"].abs()
    dQ = (discharge_cap - charge_cap).abs()
    pulse_steps = pulse_steps.filter(dQ < Q * upper_pulse_cutoff)

    if pulse_steps.height <= options_validated["min pulses"]:
        logger.warning(
            "Insufficient pulse steps found in the data, unable to add labels."
        )
        return steps_pl

    pulse_step_counts = pulse_steps["Step count"]
    discharge_cap = pulse_steps["Discharge capacity [A.h]"].abs()
    charge_cap = pulse_steps["Charge capacity [A.h]"].abs()
    dQ = (discharge_cap - charge_cap).abs()
    long_pulse_steps = pulse_steps.filter(dQ > Q * lower_pulse_cutoff)
    short_pulse_steps = pulse_steps.filter(dQ <= Q * lower_pulse_cutoff)
    short_pulse_step_counts = short_pulse_steps["Step count"]

    if current_direction in ["discharge", "delithiation"]:
        long_pulse_steps_remove = long_pulse_steps.filter(
            pl.col("Step type") == "Constant current charge"
        )
        long_pulse_steps = long_pulse_steps.filter(
            pl.col("Step type") == "Constant current discharge"
        )
    elif current_direction in ["charge", "lithiation"]:
        long_pulse_steps_remove = long_pulse_steps.filter(
            pl.col("Step type") == "Constant current discharge"
        )
        long_pulse_steps = long_pulse_steps.filter(
            pl.col("Step type") == "Constant current charge"
        )
    else:
        raise ValueError("Undefined current direction")

    long_pulse_step_counts = long_pulse_steps["Step count"]
    long_pulse_step_counts_remove = long_pulse_steps_remove["Step count"]
    long_pulse_set = set(long_pulse_step_counts.to_list())

    rest_steps = steps_pl.filter(pl.col("Step type") == "Rest")
    rest_step_counts = rest_steps["Step count"]
    rest_arr = rest_step_counts.to_numpy()
    long_arr = long_pulse_step_counts.to_numpy()
    short_arr = short_pulse_step_counts.to_numpy()
    long_mask = (np.abs(rest_arr[:, None] - long_arr) == 1).any(axis=1)
    short_mask = (np.abs(rest_arr[:, None] - short_arr) == 1).any(axis=1)
    rest_step_counts = pl.Series(
        rest_step_counts.name, rest_arr[long_mask | short_mask]
    )

    step_counts = pl.concat([pulse_step_counts, rest_step_counts]).unique().sort()
    remove_set = set(long_pulse_step_counts_remove.to_list())
    step_counts_list = [s for s in step_counts.to_list() if s not in remove_set]
    step_counts_arr = np.array(sorted(set(step_counts_list)))

    blocks: list[list[int]] = []
    current_block: list[int] = []
    for step in step_counts_arr:
        step_val = int(step)
        if len(current_block) == 0:
            current_block.append(step_val)
        elif step_val == current_block[-1] + 1:
            current_block.append(step_val)
        else:
            if any(s in long_pulse_set for s in current_block):
                blocks.append(current_block)
            current_block = [step_val]
    if len(current_block) > 0 and any(s in long_pulse_set for s in current_block):
        blocks.append(current_block)

    step_to_label: dict[int, str] = {}
    step_to_group: dict[int, int] = {}

    for block in blocks:
        block_df = steps_pl.filter(pl.col("Step count").is_in(block))
        is_long = block_df["Step count"].is_in(long_pulse_step_counts.to_list())
        group_vals = (is_long.cum_sum() - 1).cast(pl.Int64)
        block_df = block_df.with_columns(group_vals.alias("_grp"))

        rest_in_block = block_df.filter(pl.col("Step type") == "Rest")
        n_rest_per_group = rest_in_block.group_by("_grp").len()
        one_rest_per_group = (
            n_rest_per_group.filter(pl.col("len") == 1).height
            == n_rest_per_group.height
        )
        label = "GITT" if one_rest_per_group else "HPPT"

        if label == "GITT":
            block_df = block_df.filter(pl.col("_grp") >= 0)
        elif block_df["_grp"].min() == -1:
            block_df = block_df.with_columns((pl.col("_grp") + 1).alias("_grp"))

        for row in block_df.iter_rows(named=True):
            sc = row["Step count"]
            step_to_label[sc] = label
            step_to_group[sc] = int(row["_grp"])

    if not step_to_label:
        return steps_pl

    step_counts_list = steps_pl["Step count"].to_list()
    orig_labels = (
        steps_pl["Label"].to_list()
        if "Label" in steps_pl.columns
        else [""] * len(steps_pl)
    )
    orig_groups = (
        steps_pl["Group number"].to_list()
        if "Group number" in steps_pl.columns
        else [None] * len(steps_pl)
    )
    label_series = pl.Series(
        "Label",
        [
            step_to_label.get(sc, orig_labels[i])
            for i, sc in enumerate(step_counts_list)
        ],
    )
    group_series = pl.Series(
        "Group number",
        [
            step_to_group.get(sc, orig_groups[i])
            for i, sc in enumerate(step_counts_list)
        ],
    )
    steps_pl = steps_pl.with_columns([label_series, group_series])
    return steps_pl
