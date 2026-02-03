import pytest
import ionworksdata as iwdata
from datetime import datetime, timezone, timedelta
import numpy as np
from ionworksdata.util import monotonic_time_offset


@pytest.mark.parametrize(
    "options, expected",
    [
        (None, ("A", "A.h")),
        ({"current units": "total"}, ("A", "A.h")),
        ({"current units": "density"}, ("mA.cm-2", "mA.h.cm-2")),
    ],
)
def test_get_current_and_capacity_units(options, expected):
    current_units, capacity_units = iwdata.util.get_current_and_capacity_units(options)
    assert current_units == expected[0]
    assert capacity_units == expected[1]


def test_get_current_and_capacity_units_invalid_option():
    with pytest.raises(ValueError, match="Invalid current units option"):
        iwdata.util.get_current_and_capacity_units({"current units": "invalid"})


def test_check_and_convert_datetime():
    assert iwdata.util.check_and_convert_datetime(
        datetime(2021, 1, 1, tzinfo=timezone.utc)
    ) == datetime(2021, 1, 1, tzinfo=timezone.utc)
    with pytest.raises(ValueError, match="Start datetime must be timezone aware"):
        iwdata.util.check_and_convert_datetime(datetime.now())
    with pytest.raises(ValueError, match="Start datetime cannot be in the future"):
        iwdata.util.check_and_convert_datetime(
            datetime.now(timezone.utc) + timedelta(seconds=1)
        )
    assert iwdata.util.check_and_convert_datetime(
        "2021-01-01T00:00:00+00:00"
    ) == datetime(2021, 1, 1, tzinfo=timezone.utc)
    with pytest.raises(ValueError, match="Do not use datetime.now"):
        iwdata.util.check_and_convert_datetime(datetime.now(timezone.utc))


def test_monotonic_time_offset():
    """Test the monotonic_time_offset function."""
    # Test basic functionality
    time_points = np.array([0.0, 1.0, 2.0, 3.0])
    start_time = 10.0
    result = monotonic_time_offset(time_points, start_time)

    # Check that all times are greater than start_time
    assert np.all(result > start_time)
    # Check that the array is strictly increasing
    assert np.all(np.diff(result) > 0)
    # Check that the relative differences are preserved
    assert np.allclose(np.diff(result), np.diff(time_points))

    # Test with offset_initial_time=True
    result_with_offset = monotonic_time_offset(
        time_points, start_time, offset_initial_time=True
    )
    assert result_with_offset[0] > start_time
    assert np.all(np.diff(result_with_offset) > 0)

    # Test with offset_initial_time=False
    result_without_offset = monotonic_time_offset(
        time_points, start_time, offset_initial_time=False
    )
    assert result_without_offset[0] == start_time
    assert np.all(np.diff(result_without_offset) > 0)

    # Test with floating point precision issues
    time_points_fp = np.array([0.0, 1e-15, 2e-15, 3e-15])
    result_fp = monotonic_time_offset(time_points_fp, start_time)
    assert np.all(result_fp > start_time)
    assert np.all(np.diff(result_fp) > 0)

    # Test with single time point
    single_time = np.array([5.0])
    result_single = monotonic_time_offset(single_time, start_time)
    assert len(result_single) == 1
    assert result_single[0] > start_time

    # Test with non-zero initial time
    time_points_nonzero = np.array([10.0, 11.0, 12.0])
    start_time_nonzero = 5.0
    result_nonzero = monotonic_time_offset(time_points_nonzero, start_time_nonzero)
    assert np.all(result_nonzero > start_time_nonzero)
    assert np.all(np.diff(result_nonzero) > 0)
    assert np.allclose(np.diff(result_nonzero), np.diff(time_points_nonzero))

    # Test with floating point precision edge case using np.nextafter
    time_points_nextafter = np.array([100.0, np.nextafter(100.0, np.inf), 101.0])
    start_time_nextafter = 50.0
    result_nextafter = monotonic_time_offset(
        time_points_nextafter, start_time_nextafter
    )
    assert np.all(result_nextafter > start_time_nextafter)
    assert np.all(np.diff(result_nextafter) > 0)
    # The relative differences should be preserved
    assert np.allclose(np.diff(result_nextafter), np.diff(time_points_nextafter))
