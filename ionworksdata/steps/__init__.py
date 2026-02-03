"""
Step analysis and labeling module for battery cycling data.

This module provides functions to:
- Summarize time series data into step-level information
- Label steps as cycling, pulse (GITT/HPPT), or EIS
- Annotate time series with step labels
- Validate labeled steps

Example usage::

    import ionworksdata as iwdata

    # Summarize time series into steps
    steps = iwdata.steps.summarize(data)

    # Label cycling steps
    steps = iwdata.steps.label_cycling(steps, options)

    # Label pulse steps (GITT/HPPT)
    steps = iwdata.steps.label_pulse(steps, options)

    # Label EIS steps
    steps = iwdata.steps.label_eis(steps)

    # Annotate time series with labels
    time_series = iwdata.steps.annotate(time_series, steps, ["Label"])

    # Validate labeled steps
    is_valid = iwdata.steps.validate(steps, "Cycling")
"""

from ._core import (
    summarize,
    identify,
    set_cycle_capacity,
    set_cycle_energy,
    infer_type,
    annotate,
)
from ._validate import validate
from ._cycling import label_cycling
from ._pulse import label_pulse
from ._eis import label_eis

__all__ = [
    "summarize",
    "identify",
    "set_cycle_capacity",
    "set_cycle_energy",
    "infer_type",
    "annotate",
    "validate",
    "label_cycling",
    "label_pulse",
    "label_eis",
]
