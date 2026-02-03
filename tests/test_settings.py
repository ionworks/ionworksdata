"""Tests for the settings module."""

import pytest

import ionworksdata as iwdata


def test_settings_basic_functionality():
    """Test basic settings functionality."""
    # Test getting settings
    settings = iwdata.get_settings()
    assert hasattr(settings, "current_std_tol")
    assert hasattr(settings, "voltage_std_tol")
    assert hasattr(settings, "power_std_tol")
    assert hasattr(settings, "rest_tol")
    assert hasattr(settings, "eis_tol")
    assert hasattr(settings, "zero_current_percent_tol")
    assert hasattr(settings, "eis_tolerance")
    assert hasattr(settings, "sign_tolerance")


def test_settings_update():
    """Test updating settings."""
    # Get original values
    original_current_tol = iwdata.settings.get_current_std_tol()
    original_voltage_tol = iwdata.settings.get_voltage_std_tol()

    # Update settings
    iwdata.update_settings(current_std_tol=1e-2, voltage_std_tol=2e-2)

    # Check updated values
    assert iwdata.settings.get_current_std_tol() == 1e-2
    assert iwdata.settings.get_voltage_std_tol() == 2e-2

    # Reset to original values
    iwdata.update_settings(
        current_std_tol=original_current_tol, voltage_std_tol=original_voltage_tol
    )


def test_settings_reset():
    """Test resetting settings to defaults."""
    # Change a setting
    iwdata.update_settings(current_std_tol=1e-1)
    assert iwdata.settings.get_current_std_tol() == 1e-1

    # Reset to defaults
    iwdata.reset_settings()
    # Check that it's back to the default value (1e-2 based on user's changes)
    assert iwdata.settings.get_current_std_tol() == 1e-2


def test_settings_invalid_parameter():
    """Test that invalid parameters raise ValueError."""
    with pytest.raises(ValueError, match="Invalid setting"):
        iwdata.update_settings(invalid_setting=1.0)


def test_settings_dict_conversion():
    """Test converting settings to/from dictionary."""
    # Get current settings as dict
    settings_dict = iwdata.get_settings().to_dict()

    # Check that all expected keys are present
    expected_keys = {
        "current_std_tol",
        "voltage_std_tol",
        "power_std_tol",
        "rest_tol",
        "eis_tol",
        "zero_current_percent_tol",
        "eis_tolerance",
        "sign_tolerance",
    }
    assert set(settings_dict.keys()) == expected_keys

    # Test loading from dict
    modified_dict = settings_dict.copy()
    modified_dict["current_std_tol"] = 5e-3

    iwdata.get_settings().from_dict(modified_dict)
    assert iwdata.settings.get_current_std_tol() == 5e-3

    # Restore original
    iwdata.get_settings().from_dict(settings_dict)
