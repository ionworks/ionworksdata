from __future__ import annotations

import warnings
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from iwutil import check_and_combine_options
from matplotlib.widgets import Slider

# Compatibility patch for NumPy 2.0: alias np.trapz to np.trapezoid
# This fixes PyBaMM which uses np.trapz internally
try:
    _ = np.trapz
except AttributeError:
    try:
        np.trapz = np.trapezoid
    except AttributeError:
        pass

import pybamm


def _trapezoid(y, x=None, dx=1.0, axis=-1):
    """
    Compatibility wrapper for numpy trapezoid integration.

    Uses np.trapezoid for numpy >= 2.0.0, otherwise falls back to np.trapz.

    Parameters
    ----------
    y : array_like
        Input array to integrate.
    x : array_like, optional
        The sample points corresponding to the y values.
    dx : scalar, optional
        The spacing between sample points when x is None.
    axis : int, optional
        The axis along which to integrate.

    Returns
    -------
    float or ndarray
        Definite integral as approximated by trapezoidal rule.
    """
    # Try np.trapezoid first (NumPy >= 2.0), fall back to np.trapz (NumPy < 2.0)
    try:
        return np.trapezoid(y, x=x, dx=dx, axis=axis)
    except AttributeError:
        return np.trapz(y, x=x, dx=dx, axis=axis)


class PiecewiseLinearTimeseries:
    """
    A class to preprocess and linearize time series data using piecewise linear
    approximation.

    Attributes:
    -----------
    t_data : array-like
        The time data points.
    y_data : array-like
        The corresponding data points.
    atol : float, optional
        Absolute tolerance for the solver. If None, uses the default solver `atol`
        of `1e-6`.
    rtol : float, optional
        Relative tolerance for the solver. If None, uses the default solver `rtol`
        of `1e-4`.
    name : str, optional
        The name of the timeseries. Default is "Piecewise linear timeseries".
    options : dict, optional
        Additional options for preprocessing. If None, default options are used.

        solver_max_save_points : int, optional
            Maximum number of points to save in the solver. Disabled by default.
        interactive_preprocessing : bool
            Whether to use interactive preprocessing to select atol and rtol.
            Default is False.
        window_max : int
            Maximum window size for removing neighboring points. Default is 10.
    """

    def __init__(
        self,
        t_data: np.ndarray,
        y_data: np.ndarray,
        atol: float | None = None,
        rtol: float | None = None,
        name: str | None = None,
        options: dict[str, Any] | None = None,
    ):
        if atol is None:
            atol = _default_atol()
        self.atol = atol

        if rtol is None:
            rtol = _default_rtol()
        self.rtol = rtol

        self.name = name or "Piecewise linear timeseries"

        # Set options
        options = options or {}
        default_options = {
            "solver_max_save_points": None,
            "interactive_preprocessing": False,
            "window_max": _default_window_max(),
        }
        options = check_and_combine_options(default_options, options)
        self.options = options

        self.t_data = t_data
        self.y_data = y_data

        self._linearize()

    def _linearize(self) -> None:
        """
        Linearizes the time series data based on the provided solver and options.
        This method processes the time series data (`t_data` and `y_data`) and
        linearizes it according to the specified solver and options. The method
        also identifies discontinuities in the data and reduces the number of
        save points if necessary.
        Attributes:
        -----------
        t_data : array-like
            The time data points.
        y_data : array-like
            The corresponding data values.
        atol : float
            Absolute tolerance for the solver.
        rtol : float
            Relative tolerance for the solver.
        options : dict
            Dictionary containing various options for preprocessing and solver
            settings.
        """

        t = self.t_data
        y = self.y_data
        atol = self.atol
        rtol = self.rtol
        name = self.name

        solver_max_save_points = self.options["solver_max_save_points"]
        interactive_preprocessing = self.options["interactive_preprocessing"]
        window_max = self.options["window_max"]

        # Process input data
        if interactive_preprocessing:
            if pybamm.is_notebook():
                warnings.warn(
                    "Interactive preprocessing is not supported in Jupyter notebooks. "
                    "Using default preprocessing instead.",
                    stacklevel=2,
                )
            else:
                atol, rtol = interactive_time_stepping_fit(
                    t, y, atol=atol, rtol=rtol, window_max=window_max, name=name
                )
                self.atol = atol
                self.rtol = rtol
        segments_full = process_input_data(
            t, y, atol=atol, rtol=rtol, window_max=window_max
        )

        t_sparse = t[segments_full]
        y_sparse = y[segments_full]
        self.t_sparse = t_sparse
        self.y_sparse = y_sparse

        # Find discontinuities in the data
        points_discon = find_input_discontinuities(
            t_sparse, y_sparse, atol=atol, rtol=rtol
        )
        self.t_discon = t_sparse[points_discon]

        # If the number of points is too large, reduce the number of save points
        if solver_max_save_points is not None and len(t) > solver_max_save_points:
            idx_reduced = np.linspace(
                0, len(t) - 1, solver_max_save_points // 2, dtype=int
            )
            t_data_reduced = t[idx_reduced]

            t_linspace = np.linspace(t[0], t[-1], solver_max_save_points // 2)

            t_interp = np.sort(np.unique(np.concatenate((t_data_reduced, t_linspace))))
        else:
            t_interp = t

        self.t_interp = t_interp

    def interpolant(
        self, interpolator: str = "linear", name: str | None = None, **kwargs
    ) -> pybamm.Interpolant:
        r"""
        Generate an interpolant for the given sparse time series data.

        Parameters
        ----------
        interpolator : str, optional
            The type of interpolation to use. Default is "linear".
        name : str, optional
            The name of the interpolant. Default is the name of the timeseries.
        \*\*kwargs
            Additional keyword arguments to pass to pybamm.Interpolant.

        Returns
        -------
        pybamm.Interpolant
            An interpolant object for the sparse time series data.
        """
        name = name or self.name

        itp = pybamm.Interpolant(
            self.t_sparse,
            self.y_sparse,
            pybamm.t,
            interpolator=interpolator,
            name=name,
            **kwargs,
        )
        return itp


def process_input_data(
    t: np.ndarray,
    y: np.ndarray,
    rtol: float,
    atol: float,
    window_max: int,
) -> np.ndarray:
    """
    Process data to find significant segments and points for efficient representation.

    This function identifies key points in the data that represent significant changes
    or important features, allowing for a more compact representation of the data
    while preserving its essential characteristics.

    Parameters
    ----------
    t : array-like
        Full time array of the data.
    y : array-like
        Full value array of the data.
    rtol : float
        Relative tolerance for identifying significant changes.
    atol : float
        Absolute tolerance for identifying significant changes.
    window_max : int
        Maximum window size for removing neighboring points.

    Returns
    -------
    array
        An array of indices representing key points in the data.
    """

    def atol_check(y: np.ndarray, atol: float):
        y_diff = np.concatenate(([0], np.abs(np.diff(y))))
        mask = y_diff > atol

        mask[:-1] = mask[:-1] | mask[1:]
        mask[0] = True
        mask[-1] = True
        return mask

    # Step 1: Find violations of the absolute tolerance
    mask = atol_check(y, atol)
    segments_full = np.where(mask)[0]

    # Step 2: Find regions of contiguous segments violating
    # the absolute tolerance
    segments = find_contiguous_segments(mask)

    # Step 3: Find linear segments within each contiguous segment
    # subject to the relative and absolute tolerances
    points_linear = np.array([], dtype=int)
    for seg in segments:
        t_new = t[seg[0] : seg[1] + 1]
        y_new = y[seg[0] : seg[1] + 1]
        points_seg = find_linear_segments(t_new, y_new, rtol=rtol, atol=atol)
        points_seg = points_seg + seg[0]
        points_linear = np.concatenate((points_linear, points_seg))

        mask[seg[0] + 1 : seg[1]] = False

    points_linear = np.unique(np.sort(points_linear))

    # Combine segments_full and points to find all significant segments
    segments = np.union1d(segments_full, points_linear)

    mask = atol_check(y[segments], atol)
    segments = segments[mask]

    # Step 4: Remove neighboring points within a window size
    for window in range(2, window_max + 1):
        for i in range(len(segments) - 2, window - 2, -1):
            lb_segment = max(0, i + 1 - window)
            ub_segment = min(len(segments) - 1, i + 1)
            slice_range = slice(segments[lb_segment], segments[ub_segment] + 1)
            t_vec, y_vec = t[slice_range], y[slice_range]

            if calculate_normalized_linear_fit_error(t_vec, y_vec, atol) < rtol:
                # Delete the intermediate segments.
                segments = np.delete(segments, slice(lb_segment + 1, ub_segment))

    # Step 5: Final atol check to remove redundant points
    mask = atol_check(y[segments], atol)
    segments = segments[mask]

    return segments


def find_input_discontinuities(
    t: np.ndarray,
    y: np.ndarray,
    atol: float,
    rtol: float,
    scale_factor: float | None = None,
) -> np.ndarray:
    """
    Find discontinuities in the data based on changes in slope.

    This function identifies points where the change in slope exceeds a threshold,
    which is determined by both absolute and relative tolerances.

    Parameters
    ----------
    t : array-like
        Time array of the data.
    y : array-like
        Value array of the data.
    atol : float
        Absolute tolerance for identifying discontinuities.
    rtol : float
        Relative tolerance for identifying discontinuities.
    scale_factor : float, optional (default=1)
        Factor to scale the threshold for identifying discontinuities.

    Returns
    -------
    array
        An array of indices where discontinuities are detected.
    """
    if scale_factor is None:
        scale_factor = 1
    t_diff = np.diff(t)
    t_diff[t_diff == 0] = atol
    slopes = np.diff(y) / t_diff

    mask = np.abs(np.diff(slopes)) > scale_factor * (rtol * np.abs(slopes[:-1]) + atol)
    points_discon = np.arange(len(t))[np.concatenate(([True], mask, [True]))]
    return points_discon


def find_contiguous_segments(mask: np.ndarray) -> list[list[int]]:
    """
    Find contiguous segments in a boolean mask.

    Parameters
    ----------
    mask : array-like
        Boolean mask to find segments in.

    Returns
    -------
    list
        List of [start, end] indices for each contiguous segment.
    """
    segments = []
    start = None

    for i, val in enumerate(mask):
        if val:
            if start is None:
                start = i
        else:
            if start is not None and i >= start + 2:
                segments.append([start, i])
            start = None

    if start is not None and len(mask) >= start + 2:
        segments.append([start, len(mask) - 1])

    return segments


def calculate_linear_fit_error(t: np.ndarray, y: np.ndarray, atol: float) -> float:
    """
    Calculate the error of a linear fit to the data.

    Parameters
    ----------
    t : array-like
        Time array for the segment.
    y : array-like
        Value array for the segment.
    atol : float
        Absolute tolerance for calculations.

    Returns
    -------
    float
        Error of the linear fit.
    """
    if len(t) <= 2 or t[0] == t[-1]:
        return 0
    slope = (y[-1] - y[0]) / (t[-1] - t[0])
    return _trapezoid(np.abs(y - (slope * (t - t[0]) + y[0])), t)


def calculate_normalized_linear_fit_error(
    t: np.ndarray, y: np.ndarray, atol: float
) -> float:
    """
    Calculate the normalized error of a linear fit to the data.

    Parameters
    ----------
    t : array-like
        Time array for the segment.
    y : array-like
        Value array for the segment.
    atol : float
        Absolute tolerance for calculations.

    Returns
    -------
    float
        Normalized error of the linear fit.
    """
    numerator = calculate_linear_fit_error(t, y, atol)
    if numerator == 0:
        return 0
    return numerator / (np.abs(_trapezoid(y, t)) + atol)


def find_linear_segment_end(
    t: np.ndarray, y: np.ndarray, rtol: float, atol: float
) -> int:
    """
    Find the end index of a linear segment.

    Parameters
    ----------
    t : array-like
        Time array for the segment.
    y : array-like
        Value array for the segment.
    rtol : float
        Relative tolerance for identifying significant changes.
    atol : float
        Absolute tolerance for calculations.

    Returns
    -------
    int
        Index where the segment stops being linear within tolerance.
    """
    for i in range(2, len(t)):
        if calculate_normalized_linear_fit_error(t[: i + 1], y[: i + 1], atol) > rtol:
            return i - 1
    return len(t) - 1


def find_linear_segments(
    t: np.ndarray, y: np.ndarray, rtol: float, atol: float
) -> np.ndarray:
    """
    Find all linear segments in the data.

    Parameters
    ----------
    t : array-like
        Time array for the data.
    y : array-like
        Value array for the data.
    rtol : float
        Relative tolerance for identifying significant changes.
    atol : float
        Absolute tolerance for calculations.

    Returns
    -------
    array
        Array of indices where linear segments end.
    """
    points = [0, len(t) - 1]

    start = 0
    while True:
        seg_end = find_linear_segment_end(t, y, rtol=rtol, atol=atol)
        start += seg_end
        if start in points:
            break
        points.append(start)
        t = t[seg_end:]
        y = y[seg_end:]
    return np.sort(points)


def find_optimal_midpoint(t: np.ndarray, y: np.ndarray, N_max: int, atol: float) -> int:
    """
    Find the optimal midpoint in a segment that minimizes linear fit error.

    Parameters
    ----------
    t : array-like
        Time array for the segment.
    y : array-like
        Value array for the segment.
    N_max : int
        Maximum number of segments to consider in optimization steps.
    atol : float
        Absolute tolerance for calculations.

    Returns
    -------
    int
        Index of the optimal midpoint.
    """
    err_best = np.inf
    idx_best = 0
    idx_left = 0
    idx_right = 0

    reduced_segments = len(t) > N_max
    if reduced_segments:
        segments = np.linspace(0, len(t) - 1, N_max, dtype=int)
    else:
        segments = np.arange(len(t))

    for idx, i in enumerate(segments):
        t_vec1 = t[: i + 1]
        y_vec1 = y[: i + 1]
        t_vec2 = t[i:]
        y_vec2 = y[i:]
        err = calculate_linear_fit_error(
            t_vec1, y_vec1, atol
        ) + calculate_linear_fit_error(t_vec2, y_vec2, atol)
        if err_best >= err - atol:
            err_best = err
            idx_best = i
            idx_left = segments[max(0, idx - 1)]
            idx_right = segments[min(len(segments) - 1, idx + 1)]

    if reduced_segments:
        t_vec = t[idx_left : idx_right + 1]
        y_vec = y[idx_left : idx_right + 1]
        return idx_left + find_optimal_midpoint(t_vec, y_vec, N_max, atol)
    else:
        return idx_best


def calc_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the R^2 (coefficient of determination).

    Parameters
    ----------
    y_true : array-like, shape (n,)
        Observed (true) values.
    y_pred : array-like, shape (n,)
        Predicted values.

    Returns
    -------
    r2_weighted : float
        The R^2 value.
    """
    # Weighted sum of squares of residuals
    ss_res = np.sum((y_true - y_pred) ** 2)

    # Weighted total sum of squares
    ss_tot = np.sum((y_true - np.average(y_true)) ** 2)

    # Compute R^2. Add a small number to the denominator to avoid division by zero.
    r2 = 1 - ss_res / (ss_tot + 1e-100)

    return r2


def interactive_time_stepping_fit(
    t: np.ndarray, y: np.ndarray, atol: float, rtol: float, window_max: int, name: str
) -> tuple[float, float]:
    # Create the plot
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.35)

    # Plot the raw data (which doesn't change)
    ax.plot(t, y, "o", label="Raw data", color="gray", alpha=0.5, ms=4)

    # Initial plot for the piecewise linear fit
    segments = process_input_data(t, y, atol=atol, rtol=rtol, window_max=window_max)
    t_sparse = t[segments]
    y_sparse = y[segments]

    [line_fit] = ax.plot(
        t_sparse, y_sparse, "x-", label="Piecewise linear fit", color="red", lw=2
    )
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")

    def make_title(t_sparse: np.ndarray, y_sparse: np.ndarray) -> str:
        y_linear = np.interp(t, t_sparse, y_sparse)
        R2 = calc_r2(y, y_linear)
        title = f"Piecewise linear fit - {100 * (1 - len(t_sparse) / len(t)):.2f}% reduction, {R2:.5f} R²"
        return title

    ax.set_title(make_title(t_sparse, y_sparse))
    ax.legend()

    # Function to update the piecewise fit without clearing the plot
    def update_plot(atol: float, rtol: float) -> None:
        segments = process_input_data(t, y, rtol=rtol, atol=atol, window_max=window_max)
        t_sparse = t[segments]
        y_sparse = y[segments]

        # Update the data of the piecewise fit line
        line_fit.set_xdata(t_sparse)
        line_fit.set_ydata(y_sparse)

        # Update the title to reflect data reduction
        ax.set_title(make_title(t_sparse, y_sparse))

        # Redraw only the updated parts
        fig.canvas.draw_idle()

    # Add sliders for atol and rtol
    ax_atol = plt.axes([0.25, 0.2, 0.65, 0.03], facecolor="lightgoldenrodyellow")
    ax_rtol = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor="lightgoldenrodyellow")

    s_atol = Slider(ax_atol, "log10(atol)", -10, 0, valinit=np.log10(atol))
    s_rtol = Slider(ax_rtol, "log10(rtol)", -10, 0, valinit=np.log10(rtol))

    # Update the plot when sliders are changed
    def on_slider_change(val: float) -> None:
        update_plot(10**s_atol.val, 10**s_rtol.val)

    s_atol.on_changed(on_slider_change)
    s_rtol.on_changed(on_slider_change)

    plt.show()

    atol_opt = 10**s_atol.val
    rtol_opt = 10**s_rtol.val

    print(
        f"{name} linearization:\n Final atol: {atol_opt}\n Final rtol: {rtol_opt}\n",
        end="",
    )

    return atol_opt, rtol_opt


def _default_atol() -> float:
    """
    Default absolute tolerance for the solver. Matches
    the `IDAKLUSolver` default.
    """
    return 1e-6


def _default_rtol() -> float:
    """
    Default relative tolerance for the solver. Matches
    the `IDAKLUSolver` default.
    """
    return 1e-4


def _default_window_max() -> int:
    """
    Default window size for removing neighboring points.
    """
    return 10
