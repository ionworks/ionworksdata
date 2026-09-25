"""
Settings module for ionworksdata.

This module provides configurable settings for various tolerance parameters
used throughout the ionworksdata package. Users can modify these settings
to adjust the behavior of algorithms and analysis functions.
"""

from __future__ import annotations

from typing import Any


class Settings:
    """
    Configuration settings for ionworksdata.

    This class holds all configurable parameters used throughout the package.
    Users can modify these settings to customize the behavior of various
    algorithms and analysis functions.
    """

    def __init__(self):
        # Step identification tolerances
        self.current_std_tol: float = 1e-2
        """Tolerance for standard deviation of current below which a step is considered constant current."""

        self.voltage_std_tol: float = 1e-2
        """Tolerance for standard deviation of voltage below which a step is considered constant voltage."""

        self.power_std_tol: float = 1e-2
        """Tolerance for standard deviation of power below which a step is considered constant power."""

        self.rest_tol: float = 1e-3
        """Tolerance for absolute value of current below which a step is considered a rest step."""

        self.eis_tol: float = 1e-8
        """Tolerance for absolute value of frequency below which a step is considered an EIS step."""

        # Transform tolerances
        self.zero_current_percent_tol: float = 1e-2
        """Tolerance for considering current as zero when using current sign method (as percentage of max current)."""

        self.eis_tolerance: float = 1e-8
        """Tolerance for considering a frequency as an EIS step in transform operations."""

        self.sign_tolerance: float = 1e-2
        """General tolerance for sign operations."""

    def update(self, **kwargs: Any) -> None:
        """
        Update settings with new values.

        Parameters
        ----------
        **kwargs
            Keyword arguments where keys are setting names and values are the new values.
            Only valid setting names will be updated.

        Raises
        ------
        ValueError
            If an invalid setting name is provided.
        """
        valid_settings = {
            "current_std_tol",
            "voltage_std_tol",
            "power_std_tol",
            "rest_tol",
            "eis_tol",
            "zero_current_percent_tol",
            "eis_tolerance",
            "sign_tolerance",
        }

        for key, value in kwargs.items():
            if key not in valid_settings:
                raise ValueError(
                    f"Invalid setting: {key}. Valid settings are: {sorted(valid_settings)}"
                )
            setattr(self, key, value)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert settings to a dictionary.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing all current settings.
        """
        return {
            "current_std_tol": self.current_std_tol,
            "voltage_std_tol": self.voltage_std_tol,
            "power_std_tol": self.power_std_tol,
            "rest_tol": self.rest_tol,
            "eis_tol": self.eis_tol,
            "zero_current_percent_tol": self.zero_current_percent_tol,
            "eis_tolerance": self.eis_tolerance,
            "sign_tolerance": self.sign_tolerance,
        }

    def from_dict(self, settings_dict: dict[str, Any]) -> None:
        """
        Load settings from a dictionary.

        Parameters
        ----------
        settings_dict : dict[str, Any]
            Dictionary containing settings to load.
        """
        self.update(**settings_dict)

    def reset_to_defaults(self) -> None:
        """Reset all settings to their default values."""
        self.__init__()

    def __repr__(self) -> str:
        """String representation of the settings."""
        settings_str = "Settings(\n"
        for key, value in self.to_dict().items():
            settings_str += f"    {key}={value},\n"
        settings_str += ")"
        return settings_str


# Global settings instance
_settings = Settings()


def get_settings() -> Settings:
    """
    Get the global settings instance.

    Returns
    -------
    Settings
        The global settings instance.
    """
    return _settings


def update_settings(**kwargs: Any) -> None:
    """
    Update the global settings with new values.

    Parameters
    ----------
    **kwargs
        Keyword arguments where keys are setting names and values are the new values.
        Only valid setting names will be updated.

    Raises
    ------
    ValueError
        If an invalid setting name is provided.
    """
    _settings.update(**kwargs)


def reset_settings() -> None:
    """Reset the global settings to their default values."""
    _settings.reset_to_defaults()


# Convenience functions for accessing specific settings
def get_current_std_tol() -> float:
    """Get the current standard deviation tolerance setting."""
    return _settings.current_std_tol


def get_voltage_std_tol() -> float:
    """Get the voltage standard deviation tolerance setting."""
    return _settings.voltage_std_tol


def get_power_std_tol() -> float:
    """Get the power standard deviation tolerance setting."""
    return _settings.power_std_tol


def get_rest_tol() -> float:
    """Get the rest tolerance setting."""
    return _settings.rest_tol


def get_eis_tol() -> float:
    """Get the EIS tolerance setting."""
    return _settings.eis_tol


def get_zero_current_percent_tol() -> float:
    """Get the zero current percentage tolerance setting."""
    return _settings.zero_current_percent_tol


def get_eis_tolerance() -> float:
    """Get the EIS tolerance setting for transform operations."""
    return _settings.eis_tolerance


def get_sign_tolerance() -> float:
    """Get the sign tolerance setting."""
    return _settings.sign_tolerance
