# Specify the fixtures to be used in the tests
# Any methods in the files referenced below that are decorated with @fixture
# will be available to use in any test
# A test that calls a fixture will run the fixture method and pass the result
# to the test method
# See https://docs.pytest.org/en/stable/fixture.html for more information


def pytest_configure(config):
    """Configure pytest to ignore expected warnings."""
    # Ignore CSV reader start time warnings
    config.addinivalue_line(
        "filterwarnings",
        "ignore:CSV reader does not support reading start time from file:UserWarning",
    )


pytest_plugins = [
    "tests.fixtures.data",
]
