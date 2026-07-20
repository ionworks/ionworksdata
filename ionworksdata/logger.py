from __future__ import annotations

import logging
import sys

import numpy as np

# Compatibility patch for NumPy 2.0: alias np.trapz to np.trapezoid
# This fixes PyBaMM which uses np.trapz internally
try:
    _ = np.trapz
except AttributeError:
    try:
        np.trapz = np.trapezoid
    except AttributeError:
        pass


def get_log_level_func(value_to_log: int):
    def func(self, message: str, *args, **kws) -> None:
        if self.isEnabledFor(value_to_log):
            self._log(value_to_log, message, args, **kws)

    return func


# Additional levels inspired by verboselogs
new_levels = {"SPAM": 5, "VERBOSE": 15, "NOTICE": 25, "SUCCESS": 35}
for level, value in new_levels.items():
    logging.addLevelName(value, level)
    setattr(logging.Logger, level.lower(), get_log_level_func(value))


FORMAT = (
    "%(asctime)s.%(msecs)03d - [%(levelname)s] %(module)s.%(funcName)s(%(lineno)d): "
    "%(message)s"
)
LOG_FORMATTER = logging.Formatter(datefmt="%Y-%m-%d %H:%M:%S", fmt=FORMAT)


_log_level: str | int = "ERROR"
_pybamm_log_synced = False


def set_logging_level(log_level: str | int) -> None:
    global _log_level, _pybamm_log_synced
    _log_level = log_level
    _pybamm_log_synced = False
    _sync_pybamm_log_level()
    logger.setLevel(log_level)


def _sync_pybamm_log_level() -> None:
    """Apply the stored log level to pybamm if it's loaded.

    Called automatically by :func:`_import_pybamm` so that pybamm's logger
    stays in sync without an eager module-level import.
    """
    global _pybamm_log_synced
    if _pybamm_log_synced:
        return
    if "pybamm" not in sys.modules:
        return
    try:
        sys.modules["pybamm"].set_logging_level(_log_level)
        _pybamm_log_synced = True
    except (ImportError, AttributeError):
        pass


def _import_pybamm():
    """Import pybamm and sync its log level with ionworksdata's.

    Use ``pybamm = _import_pybamm()`` instead of a bare ``import pybamm``
    inside ionworksdata to ensure the log level is applied on first load.
    """
    import pybamm

    _sync_pybamm_log_level()
    return pybamm


def _get_new_logger(name: str, filename: str | None = None) -> logging.Logger:
    new_logger = logging.getLogger(name)
    if filename is None:
        handler = logging.StreamHandler()
    else:
        handler = logging.FileHandler(filename)
    handler.setFormatter(LOG_FORMATTER)
    new_logger.addHandler(handler)
    return new_logger


# Only the function for getting a new logger with filename not None is exposed
def get_new_logger(name: str, filename: str) -> logging.Logger:
    if filename is None:
        raise ValueError("filename must be specified")
    return _get_new_logger(name, filename)


# Create a custom logger
logger = _get_new_logger(__name__)
logger.setLevel("ERROR")
