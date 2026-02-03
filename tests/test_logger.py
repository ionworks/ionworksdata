import pytest

import ionworksdata as iwdata


def test_logger():
    logger = iwdata.logger
    assert logger.level == 40
    iwdata.set_logging_level("INFO")
    assert logger.level == 20
    iwdata.set_logging_level("ERROR")
    assert logger.level == 40
    iwdata.set_logging_level("VERBOSE")
    assert logger.level == 15
    iwdata.set_logging_level("NOTICE")
    assert logger.level == 25
    iwdata.set_logging_level("SUCCESS")
    assert logger.level == 35

    iwdata.set_logging_level("SPAM")
    assert logger.level == 5
    iwdata.logger.spam("Test spam level")
    iwdata.logger.verbose("Test verbose level")
    iwdata.logger.notice("Test notice level")
    iwdata.logger.success("Test success level")

    # reset
    iwdata.set_logging_level("WARNING")


def test_exceptions():
    with pytest.raises(ValueError):
        iwdata.get_new_logger("test", None)
