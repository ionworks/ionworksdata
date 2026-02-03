import numpy as np
import pybamm
import pytest

import matplotlib.pyplot as plt
from unittest.mock import patch

from ionworksdata.piecewise_linear_timeseries import (
    PiecewiseLinearTimeseries,
    calculate_linear_fit_error,
    calculate_normalized_linear_fit_error,
    find_contiguous_segments,
    find_input_discontinuities,
    find_linear_segment_end,
    find_linear_segments,
    find_optimal_midpoint,
    process_input_data,
)


@pytest.fixture
def setup_data_processing():
    t = np.linspace(0, 10, 1000)
    y = np.sin(t)
    atol = 1e-6
    rtol = 1e-4
    window_max = 10
    return t, y, atol, rtol, window_max


def test_piecewise_linear_timeseries_initialization(setup_data_processing):
    t_data, y_data, atol, rtol, _ = setup_data_processing

    pwl = PiecewiseLinearTimeseries(t_data, y_data, atol, rtol)

    assert np.array_equal(pwl.t_data, t_data)
    assert np.array_equal(pwl.y_data, y_data)
    assert pwl.name == "Piecewise linear timeseries"

    pwl_named = PiecewiseLinearTimeseries(t_data, y_data, atol, rtol, name="Test")
    assert pwl_named.name == "Test"


def test_piecewise_linear_timeseries_linearize(setup_data_processing):
    t_data, y_data, atol, rtol, _ = setup_data_processing

    pwl = PiecewiseLinearTimeseries(t_data, y_data, atol, rtol)

    assert hasattr(pwl, "t_sparse")
    assert hasattr(pwl, "y_sparse")
    assert hasattr(pwl, "t_discon")
    assert hasattr(pwl, "t_interp")


def test_piecewise_linear_timeseries_interpolant(setup_data_processing):
    t_data, y_data, atol, rtol, _ = setup_data_processing

    pwl = PiecewiseLinearTimeseries(t_data, y_data, atol, rtol)
    interpolant = pwl.interpolant()

    assert isinstance(interpolant, pybamm.Interpolant)


def test_piecewise_linear_timeseries_arguments(setup_data_processing):
    t_data, y_data, _, _, _ = setup_data_processing

    pwl = PiecewiseLinearTimeseries(t_data, y_data)

    assert pwl.atol == 1e-6
    assert pwl.rtol == 1e-4


def test_find_contiguous_segments(setup_data_processing):
    mask = np.array([True, True, False, False, True, True, True, False, True])
    segments = find_contiguous_segments(mask)
    assert segments == [[0, 2], [4, 7]]

    # Test with all True
    all_true = np.ones(10, dtype=bool)
    assert find_contiguous_segments(all_true) == [[0, 9]]

    # Test with all False
    all_false = np.zeros(10, dtype=bool)
    assert find_contiguous_segments(all_false) == []


def test_calculate_linear_fit_error(setup_data_processing):
    t, y, atol, _, _ = setup_data_processing
    # Test with perfect linear data
    y_linear = np.array([1, 2, 3, 4])
    t_linear = np.arange(len(y_linear))
    error_linear = calculate_linear_fit_error(t_linear, y_linear, atol)
    assert np.isclose(error_linear, 0, atol=atol * 10)

    # Test with non-linear data
    error_nonlinear = calculate_linear_fit_error(t[:10], y[:10], atol)
    assert error_nonlinear > 0


def test_calculate_normalized_linear_fit_error(setup_data_processing):
    t, y, atol, _, _ = setup_data_processing
    error = calculate_normalized_linear_fit_error(t[:10], y[:10], atol)
    assert error > 0
    assert error < 1

    # Test with constant data
    y_const = np.ones(4)
    t_const = np.arange(len(y_const))
    error_const = calculate_normalized_linear_fit_error(t_const, y_const, atol)
    assert np.isclose(error_const, 0, atol=1e-7)


def test_find_linear_segment_end(setup_data_processing):
    t, y, atol, rtol, _ = setup_data_processing
    # Test with linear data
    t_linear = np.arange(100)
    y_linear = 2 * t_linear + 1
    end_linear = find_linear_segment_end(t_linear, y_linear, rtol=rtol, atol=atol)
    assert end_linear == 99  # Should find the entire segment as linear

    # Test with non-linear data
    end_nonlinear = find_linear_segment_end(t[:100], y[:100], rtol=rtol, atol=atol)
    assert end_nonlinear > 1
    assert end_nonlinear < 100


def test_find_linear_segments(setup_data_processing):
    t, _, atol, rtol, _ = setup_data_processing

    # Test with piecewise linear function
    def piecewise_func(t):
        if t < 3:
            return t
        elif 3 <= t < 7:
            return 3
        else:
            return t - 4

    y_piecewise = np.vectorize(piecewise_func)(t)
    segments = find_linear_segments(t, y_piecewise, rtol, atol)
    assert len(segments) > 3  # Should find at least the 3 main segments
    assert segments[0] == 0
    assert segments[-1] == len(t) - 1


def test_find_optimal_midpoint(setup_data_processing):
    _, _, atol, _, _ = setup_data_processing
    # Test with V-shaped data
    t_v = np.linspace(-5, 5, 101)
    y_v = np.abs(t_v)
    midpoint = find_optimal_midpoint(t_v, y_v, 50, atol)
    assert np.isclose(midpoint, 50, atol=1)  # Should be close to the center

    # Test with linear data
    t_linear = np.arange(100)
    y_linear = 2 * t_linear + 1
    midpoint_linear = find_optimal_midpoint(t_linear, y_linear, 50, atol)
    assert midpoint_linear == 99  # Should be the end for linear data


def test_process_data_segments(setup_data_processing):
    t, y, _, rtol, window_max = setup_data_processing

    # Very high atol to force the function to follow the rtol
    atol = 1e-10

    # Test with sine wave
    segments_sine = process_input_data(
        t, y, rtol=rtol, atol=atol, window_max=window_max
    )
    assert len(segments_sine) > 1
    assert len(segments_sine) < len(t)

    # Test that all segments abide by the rtol
    for i in range(1, len(segments_sine)):
        t_vec = t[segments_sine[i - 1] : segments_sine[i] + 1]
        y_vec = y[segments_sine[i - 1] : segments_sine[i] + 1]
        error = calculate_normalized_linear_fit_error(t_vec, y_vec, atol)
        assert error <= rtol

    # Test that the function is symmetric
    segments_sine_neg = process_input_data(
        t, -y, rtol=rtol, atol=atol, window_max=window_max
    )
    assert np.all(np.asarray(segments_sine) == np.asarray(segments_sine_neg))

    # Test with linear data
    y_linear = 2 * t + 1
    segments = process_input_data(
        t, y_linear, rtol=rtol, atol=atol, window_max=window_max
    )

    assert len(segments) <= 3  # Should have very few segments for linear data


def test_find_peak(setup_data_processing):
    t, _, atol, rtol, _ = setup_data_processing
    # Test with a single discontinuity
    y = np.ones_like(t)

    # Add discontinuity
    discon = 500
    y[discon:] += 1

    # Add peaks
    peaks = [2, 31, 100, 503, 700, len(t) - 3]
    for peak in peaks:
        y[peak] += 0.1 * (1 if peak % 2 == 0 else -1)

    segments_exact = [[0, len(t) - 1]]
    segments_exact.append([discon - 1, discon])
    for peak in peaks:
        segments_exact.append([peak - 1, peak, peak + 1])

    segments_exact = np.unique(np.sort(np.concatenate(segments_exact)))

    segments = process_input_data(t, y, rtol=rtol, atol=atol, window_max=10)

    assert np.all(np.asarray(segments) == segments_exact)


def test_find_discontinuities(setup_data_processing):
    t, y, atol, rtol, _ = setup_data_processing
    # Test with a single discontinuity
    y = np.sin(t)
    y[500:] += 1
    discontinuities = find_input_discontinuities(t, y, rtol=rtol, atol=atol)
    assert 500 in discontinuities


def test_solver_max_save_points_option(setup_data_processing):
    """Test that the solver_max_save_points option limits the number of saved points."""
    t_data, y_data, atol, rtol, _ = setup_data_processing

    # Test with default options (should not limit points)
    pwl_default = PiecewiseLinearTimeseries(t_data, y_data, atol, rtol)
    default_points = len(pwl_default.t_interp)

    # Test with limited save points
    max_points = 50
    options = {"solver_max_save_points": max_points}
    pwl_limited = PiecewiseLinearTimeseries(t_data, y_data, atol, rtol, options=options)
    limited_points = len(pwl_limited.t_interp)

    # The limited version should have fewer or equal points
    assert limited_points <= max_points
    assert limited_points <= default_points

    # Test with very small limit
    very_small_limit = 5
    options_small = {"solver_max_save_points": very_small_limit}
    pwl_small = PiecewiseLinearTimeseries(
        t_data, y_data, atol, rtol, options=options_small
    )
    small_points = len(pwl_small.t_interp)

    assert small_points <= very_small_limit
    assert small_points >= 2  # Should always have at least start and end points


def test_calc_r2():
    """Test the calc_r2 function for R² calculation."""
    from ionworksdata.piecewise_linear_timeseries import calc_r2

    # Test with perfect fit (R² = 1.0)
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1, 2, 3, 4, 5])
    r2_perfect = calc_r2(y_true, y_pred)
    assert np.isclose(r2_perfect, 1.0, atol=1e-10)

    # Test with poor fit (R² close to 0)
    y_pred_poor = np.array([3, 3, 3, 3, 3])  # Constant prediction
    r2_poor = calc_r2(y_true, y_pred_poor)
    assert np.isclose(r2_poor, 0.0, atol=1e-10)

    # Test with negative R² (worse than mean)
    y_pred_worse = np.array([5, 4, 3, 2, 1])  # Inverse of true values
    r2_worse = calc_r2(y_true, y_pred_worse)
    assert r2_worse < 0


def test_interactive_preprocessing(setup_data_processing, capsys):
    t, y, atol, rtol, _ = setup_data_processing

    # Mock plt.show() to prevent the plot from displaying
    with patch("matplotlib.pyplot.show"):
        options = {"interactive_preprocessing": True}
        pwl = PiecewiseLinearTimeseries(t, y, atol=atol, rtol=rtol, options=options)

    # Check that the expected output was printed to stdout
    assert pwl.options["interactive_preprocessing"] is True

    captured = capsys.readouterr()
    assert "Piecewise linear timeseries linearization:" in captured.out
    assert "Final atol:" in captured.out
    assert "Final rtol:" in captured.out

    plt.close("all")
