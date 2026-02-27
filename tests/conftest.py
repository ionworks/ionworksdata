# Specify the fixtures to be used in the tests
# Any methods in the files referenced below that are decorated with @fixture
# will be available to use in any test
# A test that calls a fixture will run the fixture method and pass the result
# to the test method
# See https://docs.pytest.org/en/stable/fixture.html for more information

# Use non-interactive backend for tests (avoids crash when no display, e.g. CI/macOS)
import matplotlib

matplotlib.use("Agg")

import tempfile
from pathlib import Path

import pytest


def pytest_configure(config):
    """Configure pytest to ignore expected warnings."""
    # Ignore CSV reader start time warnings
    config.addinivalue_line(
        "filterwarnings",
        "ignore:CSV reader does not support reading start time from file:UserWarning",
    )
    # Ignore Polars migration warnings from DataLoader
    config.addinivalue_line(
        "filterwarnings",
        "ignore:DataLoader.data and .steps now return Polars DataFrames:FutureWarning",
    )


pytest_plugins = [
    "tests.fixtures.data",
]


@pytest.fixture
def isolated_cache():
    """Fixture that provides an isolated cache directory and restores config after."""
    from ionworksdata.load import _CACHE_CONFIG

    original_config = {
        "enabled": _CACHE_CONFIG["enabled"],
        "directory": _CACHE_CONFIG["directory"],
        "ttl_seconds": _CACHE_CONFIG["ttl_seconds"],
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        _CACHE_CONFIG["directory"] = Path(tmpdir)
        _CACHE_CONFIG["ttl_seconds"] = 3600
        _CACHE_CONFIG["enabled"] = True
        yield Path(tmpdir)

    # Restore original config
    _CACHE_CONFIG["enabled"] = original_config["enabled"]
    _CACHE_CONFIG["directory"] = original_config["directory"]
    _CACHE_CONFIG["ttl_seconds"] = original_config["ttl_seconds"]
