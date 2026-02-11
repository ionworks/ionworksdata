from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import polars as pl
import warnings

import iwutil


def check_and_combine_options(
    default_options: dict[str, Any], options: dict[str, Any] | None
) -> Any:
    """Combine options with defaults. Only keys in default_options are passed to iwutil;
    other keys (e.g. for other layers in the pipeline) are ignored.
    """
    opts = options or {}
    filtered = {k: v for k, v in opts.items() if k in default_options}
    return iwutil.check_and_combine_options(default_options, filtered)


def get_current_and_capacity_units(options: dict | None) -> tuple[str, str]:
    options = options or {}
    current_format = options.get("current units", "total")
    if current_format not in ["total", "density"]:
        raise ValueError("Invalid current units option")
    current_units = {"total": "A", "density": "mA.cm-2"}[current_format]
    capacity_units = {"total": "A.h", "density": "mA.h.cm-2"}[current_format]
    return current_units, capacity_units


def check_for_duplicates(column_renamings: dict, data: pl.DataFrame) -> None:
    duplicates = []
    for c, n in column_renamings.items():
        if c in data.columns:
            if n in duplicates:
                message = f"Duplicate columns for {n} found: {c}"
                warnings.warn(message, category=UserWarning, stacklevel=2)
            else:
                duplicates.append(n)


def check_and_convert_datetime(start_datetime: str | datetime) -> datetime:
    """Check that the datetime is valid and convert to datetime object if necessary."""
    if isinstance(start_datetime, str):
        start_datetime = datetime.fromisoformat(start_datetime)

    # error if the datetime is not timezone aware
    if start_datetime.tzinfo is None:
        raise ValueError("Start datetime must be timezone aware")

    # convert to UTC
    start_datetime = start_datetime.astimezone(timezone.utc)

    # error if the datetime seems to recent
    if start_datetime > datetime.now(timezone.utc):
        raise ValueError("Start datetime cannot be in the future")
    elif start_datetime > datetime.now(timezone.utc) - timedelta(seconds=1):
        raise ValueError("Do not use datetime.now() as the start datetime.")
    return start_datetime


def monotonic_time_offset(
    time_points: np.ndarray,
    start_time: float,
    offset_initial_time: bool | None = None,
) -> np.ndarray:
    """
    Returns a time array that is strictly increasing and greater than start_time.
    If offset_initial_time is True, the first time is also greater than start_time.

    Parameters
    ----------
    time_points : np.ndarray
        The time points to offset.
    start_time : float
        The time to offset the time points by.
    offset_initial_time : bool, optional
        Whether to offset the initial time. If None, the initial time is offset if it
        is not equal to start_time. Default is `True` if the first time is not equal
        to start_time, otherwise `False`.

    Returns
    -------
    np.ndarray
        The offset time points.
    """
    if offset_initial_time is None:
        offset_initial_time = time_points[0] != start_time

    time_points = np.asarray(time_points, dtype=np.float64)
    start_time_np = np.float64(start_time)

    def next_float(x: np.float64) -> np.float64:
        return np.nextafter(x, np.inf)

    t = time_points + (-time_points[0] + start_time_np)

    # Ensure t[0] > start_time
    if offset_initial_time:
        t[0] = next_float(start_time_np)

    # Make sure that there are no floating point collisions
    if np.any(np.diff(t) <= 0):
        for i in range(1, len(t)):
            if t[i] <= t[i - 1]:
                t[i] = next_float(t[i - 1])

    return t
