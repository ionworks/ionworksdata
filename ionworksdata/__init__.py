# ruff: noqa: E402
import warnings

warnings.warn(
    "ionworksdata is deprecated and will receive no further releases. Its functionality now lives in the 'ionworks' package: https://packages.docs.ionworks.com/ionworks/migration.html",
    FutureWarning,
    stacklevel=2,
)

from . import steps, transform, util, read, cycle_metrics, write
from .analysis import AnalysisLoader
from .load import (
    DataLoader,
    OCPDataLoader,
    set_cache_enabled,
    set_cache_directory,
    get_cache_directory,
    set_cache_ttl,
    get_cache_ttl,
    clear_cache,
    window_rows,
)
from .piecewise_linear_timeseries import (
    PiecewiseLinearTimeseries,
    piecewise_linear_timeseries,
)
from .logger import logger, set_logging_level, get_new_logger
from .settings import get_settings, update_settings, reset_settings

try:
    from importlib.metadata import version

    __version__ = version("ionworksdata")
except Exception:
    __version__ = "0.0.0"
