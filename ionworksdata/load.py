from __future__ import annotations

import hashlib
import pickle
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import pybamm
from scipy.interpolate import interp1d
from scipy.signal import find_peaks, savgol_filter

from .logger import logger
from .util import monotonic_time_offset

# Compatibility patch for NumPy 2.0: alias np.trapz to np.trapezoid
# This fixes PyBaMM which uses np.trapz internally
try:
    _ = np.trapz
except AttributeError:
    try:
        np.trapz = np.trapezoid
    except AttributeError:
        pass


# =============================================================================
# Cache configuration
# =============================================================================
_CACHE_CONFIG = {
    "enabled": True,
    "directory": Path.home() / ".ionworksdata_cache",
    "ttl_seconds": 3600,  # 1 hour default TTL
}


def set_cache_enabled(enabled: bool) -> None:
    """Enable or disable caching for from_db calls."""
    _CACHE_CONFIG["enabled"] = enabled


def set_cache_directory(directory: str | Path) -> None:
    """Set the cache directory for from_db calls."""
    _CACHE_CONFIG["directory"] = Path(directory)


def get_cache_directory() -> Path:
    """Get the current cache directory."""
    return _CACHE_CONFIG["directory"]


def set_cache_ttl(ttl_seconds: int | None) -> None:
    """
    Set the cache time-to-live (TTL) in seconds.

    Parameters
    ----------
    ttl_seconds : int | None
        The time in seconds before cached data is considered stale.
        Set to None to disable TTL (cache never expires).
        Default is 3600 (1 hour).
    """
    _CACHE_CONFIG["ttl_seconds"] = ttl_seconds


def get_cache_ttl() -> int | None:
    """
    Get the current cache TTL in seconds.

    Returns
    -------
    int | None
        The TTL in seconds, or None if TTL is disabled.
    """
    return _CACHE_CONFIG["ttl_seconds"]


def clear_cache() -> int:
    """
    Clear all cached data.

    Returns
    -------
    int
        Number of cache files deleted.
    """
    cache_dir = _CACHE_CONFIG["directory"]
    if not cache_dir.exists():
        return 0

    count = 0
    for cache_file in cache_dir.glob("*.pkl"):
        cache_file.unlink()
        count += 1
    return count


def _get_cache_path(measurement_id: str) -> Path:
    """Get the cache file path for a measurement ID."""
    # Use hash to handle any special characters in measurement_id
    hash_key = hashlib.md5(measurement_id.encode()).hexdigest()[:16]
    safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in measurement_id)
    return _CACHE_CONFIG["directory"] / f"{safe_id}_{hash_key}.pkl"


def _load_from_cache(measurement_id: str) -> dict | None:
    """
    Load measurement data from cache if available and not expired.

    Returns
    -------
    dict | None
        Cached data dict with 'time_series' and 'steps' keys, or None if not cached
        or if the cache has expired.
    """
    import time

    if not _CACHE_CONFIG["enabled"]:
        return None

    cache_path = _get_cache_path(measurement_id)
    if cache_path.exists():
        # Check if cache has expired based on TTL
        ttl_seconds = _CACHE_CONFIG["ttl_seconds"]
        if ttl_seconds is not None:
            file_age = time.time() - cache_path.stat().st_mtime
            if file_age > ttl_seconds:
                # Cache has expired, delete it
                cache_path.unlink(missing_ok=True)
                return None

        try:
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        except Exception:
            # If cache is corrupted, delete it
            cache_path.unlink(missing_ok=True)
            return None
    return None


def _save_to_cache(measurement_id: str, data: dict) -> None:
    """Save measurement data to cache."""
    if not _CACHE_CONFIG["enabled"]:
        return

    cache_dir = _CACHE_CONFIG["directory"]
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_path = _get_cache_path(measurement_id)
    try:
        with open(cache_path, "wb") as f:
            pickle.dump(data, f)
    except Exception:
        # Silently fail if we can't write cache
        pass


def first_step_from_cycle(cycle: int) -> str:
    """Generate SQL query to get the first step in a given cycle."""
    return f'SELECT * FROM steps WHERE "Cycle count" = {cycle} ORDER BY "Step count" LIMIT 1'


def last_step_from_cycle(cycle: int) -> str:
    """Generate SQL query to get the last step in a given cycle."""
    return f'SELECT * FROM steps WHERE "Cycle count" = {cycle} ORDER BY "Step count" DESC LIMIT 1'


class GenericDataLoader:
    def __init__(
        self,
        data: pd.DataFrame | pl.DataFrame,
        filters: dict | None = None,
        interpolate: float | np.ndarray | None = None,
    ):
        # Convert to pandas for internal processing. This class's methods rely on:
        # - pandas interpolation methods (e.g., df.interpolate())
        # - scipy.interpolate functions that expect pandas DataFrames
        # - Boolean indexing and filtering patterns optimized for pandas
        # - Backward compatibility with existing user code that expects pandas interface
        if isinstance(data, pl.DataFrame):
            data = data.to_pandas()
        self.data = data
        if filters is not None:
            self.data = self.filter_data(self.data, filters)
        if interpolate is not None:
            self.data = self.interpolate_data(self.data, interpolate)

    def __getitem__(self, key: str) -> pd.Series:
        return self.data[key]

    def copy(self) -> GenericDataLoader:
        raise NotImplementedError(
            "Copy method not implemented for GenericDataLoader. "
            "This should be implemented in the subclass."
        )

    @staticmethod
    def filter_data(data: pd.DataFrame, filters: dict) -> pd.DataFrame:
        """
        Process the data using a specified filter function.

        Parameters
        ----------
        data : pd.DataFrame
            The raw data to be filtered.
        filters : dict
            The filter function to use. Currently supported: "savgol"

        Returns
        -------
        pd.DataFrame
            The filtered data.

        Raises
        ------
        ValueError
            If an unknown filter function is specified.
        """
        filtered_data = data.copy()
        for variable, kwargs in filters.items():
            match kwargs["filter_type"]:
                case "savgol":
                    filtered_data[variable] = savgol_filter(
                        data[variable], **kwargs["parameters"]
                    )
                case _:
                    raise ValueError(
                        f"Unknown filter function: {kwargs['filter_type']}"
                    )
        return filtered_data

    @staticmethod
    def interpolate_data(
        data: pd.DataFrame | pl.DataFrame,
        knots: float | np.ndarray,
        x_column: str = "Time [s]",
    ) -> pd.DataFrame:
        """
        Interpolate the data using np.interp

        Parameters
        ----------
        knots : float | np.ndarray
            The knots at which to interpolate the data. If a float is provided,
            the data is interpolated at regular intervals of that size. If an
            array is provided, the data is interpolated at the specified knots.

        data : pd.DataFrame | pl.DataFrame
            The data to interpolate. Must contain x_column.
        x_column : str, optional
            The column to use as the x-axis for interpolation. Defaults to "Time [s]".

        Returns
        -------
        pd.DataFrame
            The interpolated data.
        """
        # Keep both pandas and Polars versions:
        # - Polars for fast min/max aggregations
        # - Pandas for scipy.interpolate functions which require pandas DataFrames
        if isinstance(data, pl.DataFrame):
            data_pl = data
            data_pd = data.to_pandas()
        else:
            data_pl = pl.from_pandas(data)
            data_pd = data
        if isinstance(knots, float):
            x_min = data_pl[x_column].min()
            x_max = data_pl[x_column].max()
            knots = np.arange(x_min, x_max, knots)

        # Interpolation still uses numpy (no Polars equivalent)
        interpolated_data = {}
        x_values = data_pd[x_column].values
        for variable in data_pd.columns:
            if variable == x_column:
                continue
            interpolated_data[variable] = np.interp(
                knots, x_values, data_pd[variable].values
            )
        interpolated_data[x_column] = knots
        return pd.DataFrame(interpolated_data)


# TODO: add plotting methods, capacity attributes, those annoying sorting things in MSMR,
# maybe some interpolation methods, maybe a pybamm function parameter generator
class OCPDataLoader(GenericDataLoader):
    def __init__(
        self,
        data: pd.DataFrame | pl.DataFrame | dict,
        **kwargs,
    ):
        """
        Initialize an OCPDataLoader object to process OCP data.

        Parameters
        ----------
        data : pd.DataFrame | pl.DataFrame | dict
            Dataframe containing the OCP data. Can be Pandas or Polars DataFrame.
        options : dict, optional
            A dictionary containing optional parameters:

            - capacity_column : str
                The name of the column to use as the capacity. If not provided,
                auto-detects from common capacity column names.
            - sort : bool
                If True, sort the data so voltage decreases and capacity increases.
                Default is False.
            - remove_duplicates : bool
                If True, remove duplicate capacity and voltage values. Default is False.
            - remove_extremes : bool
                If True, remove data points at the extremes where the second derivative
                of voltage with respect to capacity is zero. Default is False.
            - filters : dict
                Dictionary containing filter configuration. Keys are variable names,
                values are dicts with "filter_type" (e.g., "savgol") and "parameters".
            - interpolate : float | np.ndarray
                The knots at which to interpolate the data, in seconds. If a float,
                data is interpolated at regular intervals. If an array, at specified knots.
        """
        # Support passing options as a dict, with kwargs taking precedence
        options = kwargs.pop("options", None) or {}
        merged = {**options, **kwargs}  # kwargs override options

        filters = merged.get("filters")
        interpolate = merged.get("interpolate")

        # Convert to pandas for column existence checking and value extraction
        # (uses pandas Series and DataFrame methods for backward compatibility)
        if isinstance(data, dict):
            data = pd.DataFrame(data)
        if isinstance(data, pl.DataFrame):
            data = data.to_pandas()
        self._filters = filters
        self._interpolate = interpolate

        # Handle voltage column aliasing
        potential_ocp_column_names = [
            "Voltage [V]",
            "OCP [V]",
            "OCV [V]",
            "Open-circuit potential [V]",
            "Open-circuit voltage [V]",
        ]
        if "Voltage [V]" not in data.columns:
            for column in potential_ocp_column_names:
                if column in data.columns:
                    data["Voltage [V]"] = data[column]
                    break
            else:
                raise ValueError(
                    "OCP data must contain one of the following columns: "
                    + ", ".join(potential_ocp_column_names)
                )

        # Handle capacity column aliasing
        capacity_column = merged.get("capacity_column")
        self._capacity_column = capacity_column
        potential_capacity_column_names = [
            "Capacity [A.h]",
            "Discharge capacity [A.h]",
            "Charge capacity [A.h]",
            "Capacity [mA.h.cm-2]",
            "Discharge capacity [mA.h.cm-2]",
            "Charge capacity [mA.h.cm-2]",
        ]
        if capacity_column is not None:
            # User specified which column to use
            if capacity_column in data.columns:
                data["Capacity [A.h]"] = data[capacity_column]
            else:
                raise ValueError(
                    f"Specified capacity_column '{capacity_column}' not found in data. "
                    f"Available columns: {list(data.columns)}"
                )
        elif "Capacity [A.h]" not in data.columns:
            # Auto-detect from available columns
            for column in potential_capacity_column_names:
                if column in data.columns:
                    data["Capacity [A.h]"] = data[column]
                    break

        # Apply preprocessing options
        if merged.get("sort", False):
            data = self._sort_capacity_and_ocp(data)
        if merged.get("remove_duplicates", False):
            data = self._remove_duplicate_ocp(data)
        if merged.get("remove_extremes", False):
            data = self._remove_ocp_extremes(data)

        # Store preprocessing options for serialization
        self._sort = merged.get("sort", False)
        self._remove_duplicates = merged.get("remove_duplicates", False)
        self._remove_extremes = merged.get("remove_extremes", False)

        super().__init__(data, filters, interpolate)

    @staticmethod
    def _remove_duplicate_ocp(data, capacity_column_name="Capacity [A.h]"):
        """
        Remove any duplicate capacity and voltage values.

        Parameters
        ----------
        data : pd.DataFrame
            The data to process.
        capacity_column_name : str, optional
            The name of the capacity column. Default is "Capacity [A.h]".

        Returns
        -------
        pd.DataFrame
            The data with duplicates removed.
        """
        data = data.drop_duplicates(subset=[capacity_column_name])
        data = data.drop_duplicates(subset=["Voltage [V]"])
        return data

    @staticmethod
    def _sort_capacity_and_ocp(data):
        """
        For OCP data, make sure that the capacity is always increasing and the voltage is
        always decreasing.

        Parameters
        ----------
        data : pd.DataFrame
            Pandas dataframe containing the data, should include one column with the voltage
            and one column with the capacity.

        Returns
        -------
        pd.DataFrame
            The sorted data.
        """
        # make sure voltage is decreasing, flip the whole data if not
        V = data["Voltage [V]"].values
        if V[-1] > V[0]:
            data = data.iloc[::-1].reset_index(drop=True)

        # Find the capacity column (could be in A.h, mAh.cm-2, mA.h.cm-2, etc)
        capacity_column_names = [col for col in data if col.startswith("Capacity [")]
        if len(capacity_column_names) == 0:
            raise ValueError("No capacity column found")
        elif len(capacity_column_names) > 1:
            raise ValueError(
                f"Multiple capacity columns found: {capacity_column_names}"
            )
        else:
            capacity_column_name = capacity_column_names[0]

        # Now that voltage is decreasing, if capacity is still decreasing, that means
        # capacity was measured in reverse, so flip it
        Q = data[capacity_column_name].values
        if Q[0] > Q[-1]:
            Q = Q.max() - Q

        # Make sure capacity starts at zero
        Q -= Q.min()
        data[capacity_column_name] = Q

        data = OCPDataLoader._remove_duplicate_ocp(data, capacity_column_name)

        return data

    @staticmethod
    def _remove_ocp_extremes(data):
        """
        Remove any data points at the start and end of the OCP curve where the second
        derivative of the voltage with respect to capacity is zero.

        This is to remove any data points where the voltage is not changing smoothly,
        which can have an outsized effect on the fit. For example, in some datasets,
        the OCP at the extremes is linearly extrapolated from the first and last few
        data points (zero second derivative), which can cause a bad fit in the MSMR model.

        Parameters
        ----------
        data : pd.DataFrame
            Pandas dataframe containing the data, should include one column with the voltage
            in Volts ("Voltage [V]") and one column with the capacity in Amp-hours
            ("Capacity [A.h]").

        Returns
        -------
        pd.DataFrame
            Pandas dataframe containing the trimmed data.
        """
        q = data["Capacity [A.h]"].values
        U = data["Voltage [V]"].values
        d2UdQ2 = np.gradient(np.gradient(U, q), q)
        # find the first and last points where the second derivative is positive
        first_positive, last_positive = np.where(abs(d2UdQ2) > 1e-10)[0][[0, -1]]
        data_trimmed = data.iloc[first_positive : last_positive + 1]
        return data_trimmed

    def calculate_dUdQ_cutoff(
        self,
        method: str = "explicit",
        show_plot: bool = False,
        options: dict | None = None,
    ) -> float:
        """
        Calculate the cut-off for dUdQ based on the data.

        Parameters
        ----------
        method : str, optional
            Method to use for calculating the cut-off. Options are:
            - "explicit" (default): Uses explicit method based on data range
            - "quantile": Uses quantile-based method
            - "peaks": Uses peak detection method
        show_plot : bool, optional
            Whether to show a plot of the dUdQ values with the cut-off. Default is False.
        options : dict, optional
            Dictionary of options to pass to the method. Default is None.

        Returns
        -------
        float
            Cut-off for dUdQ
        """
        xlabel = "Capacity [A.h]"
        ylabel = "Voltage [V]"

        q = self.data[xlabel]
        U = self.data[ylabel]
        dUdQ = abs(np.gradient(U, q))

        return self._calculate_differential_cutoff(
            q, U, dUdQ, method, show_plot, xlabel, ylabel, options
        )

    def calculate_dQdU_cutoff(
        self,
        method: str = "explicit",
        show_plot: bool = False,
        options: dict | None = None,
    ) -> float:
        """
        Calculate the cut-off for dQdU based on the data.

        Parameters
        ----------
        method : str, optional
            Method to use for calculating the cut-off. Options are:
            - "explicit" (default): Uses explicit method based on data range
            - "quantile": Uses quantile-based method
            - "peaks": Uses peak detection method
        show_plot : bool, optional
            Whether to show a plot of the dQdU values with the cut-off. Default is False.
        options : dict, optional
            Dictionary of options to pass to the method. Default is None.

        Returns
        -------
        float
            Cut-off for dQdU
        """
        xlabel = "Voltage [V]"
        ylabel = "Capacity [A.h]"

        U = self.data[xlabel]
        q = self.data[ylabel]
        dQdU = abs(np.gradient(q, U))

        return self._calculate_differential_cutoff(
            U, q, dQdU, method, show_plot, xlabel, ylabel, options
        )

    def _calculate_differential_cutoff(
        self,
        x,
        y,
        dydx,
        method: str = "explicit",
        show_plot: bool = False,
        xlabel: str | None = None,
        ylabel: str | None = None,
        options: dict | None = None,
    ) -> float:
        """
        Calculate the cut-off for dydx based on the data.

        Parameters
        ----------
        x : array_like
            x values
        y : array_like
            y values
        dydx : array_like
            dy/dx values
        method : str, optional
            Method to use for calculating the cut-off.
        show_plot : bool, optional
            Whether to show a plot of the dydx values with the cut-off.
        xlabel : str, optional
            Label for x-axis in plot.
        ylabel : str, optional
            Label for y-axis in plot.
        options : dict, optional
            Dictionary of options to pass to the method.

        Returns
        -------
        float
            Cut-off for dydx
        """
        options = options or {}
        if method == "explicit":
            cutoff = self._calculate_differential_cutoff_explicit(
                x, y, dydx, show_plot=show_plot, xlabel=xlabel, ylabel=ylabel, **options
            )
        elif method == "quantile":
            cutoff = self._calculate_differential_cutoff_quantile(
                x, y, dydx, show_plot=show_plot, xlabel=xlabel, ylabel=ylabel, **options
            )
        elif method == "peaks":
            cutoff = self._calculate_differential_cutoff_peaks(
                x, y, dydx, show_plot=show_plot, xlabel=xlabel, ylabel=ylabel, **options
            )
        else:
            raise ValueError(f"Method {method} not recognized")

        return cutoff

    def _calculate_differential_cutoff_explicit(
        self,
        x,
        y,
        dydx,
        show_plot: bool = False,
        xlabel: str | None = None,
        ylabel: str | None = None,
        lower_ratio: float = 0.1,
        upper_ratio: float = 0.9,
        scale: float = 1.1,
    ) -> float:
        """
        Calculate the cut-off for dydx using an explicit method.
        The cut-off is defined as scale times the maximum dydx between the min and max points.
        """
        # Find the index of the min and max points
        x = np.array(x)
        x_max = x.max()
        x_min = x.min()

        x_scaled = (x - x_min) / (x_max - x_min)
        xmin_idx = np.argmin(np.abs(x_scaled - lower_ratio))
        xmax_idx = np.argmin(np.abs(x_scaled - upper_ratio))

        # Flip the indices if they are in decreasing order
        if xmin_idx > xmax_idx:
            xmin_idx, xmax_idx = xmax_idx, xmin_idx

        # The cutoff is scale times the maximum dydx between the min and max points
        dydx_cutoff = dydx[xmin_idx:xmax_idx].max() * scale

        if show_plot:
            fig, ax = self._plot_differential_cutoff(
                x, dydx, dydx_cutoff, xlabel, ylabel
            )
            plt.show()

        return dydx_cutoff

    def _calculate_differential_cutoff_quantile(
        self,
        x,
        y,
        dydx,
        show_plot: bool = False,
        xlabel: str | None = None,
        ylabel: str | None = None,
        quantile: float = 0.8,
        scale: float = 2,
    ) -> float:
        """
        Calculate the cut-off for dydx using a quantile method.
        The cut-off is defined as the scale times the value of the quantile.
        """
        x = np.array(x)
        x_interp = np.linspace(x.min(), x.max(), 1000)
        dydx_interp = interp1d(x, dydx, kind="linear")(x_interp)

        quantile_value = np.percentile(dydx_interp, quantile * 100)
        dydx_cutoff = scale * quantile_value

        if show_plot:
            fig, ax = self._plot_differential_cutoff(
                x_interp, dydx_interp, dydx_cutoff, xlabel, ylabel
            )
            ax.axhline(
                quantile_value,
                color="tab:blue",
                linestyle="--",
                label="Quantile",
                zorder=10,
            )
            ax.axhline(
                dydx_cutoff, color="tab:red", linestyle="--", label="Cut-off", zorder=10
            )
            ax.legend()
            plt.show()

        return dydx_cutoff

    def _calculate_differential_cutoff_peaks(
        self,
        x,
        y,
        dydx,
        show_plot: bool = False,
        xlabel: str | None = None,
        ylabel: str | None = None,
        scale: float = 1.1,
        **kwargs,
    ) -> float:
        """
        Calculate the cut-off for dydx using a peak method.
        The cut-off is defined as the scale times the maximum peak in the dydx curve.
        """
        x = np.array(x)
        x_interp = np.linspace(x.min(), x.max(), 1000)
        dydx_interp = interp1d(x, dydx, kind="linear")(x_interp)

        # Find the peaks in the dydx curve using scipy.signal
        peaks, _ = find_peaks(dydx_interp, **kwargs)

        # The cutoff is scale times the maximum peak in the dydx curve
        dydx_cutoff = dydx_interp[peaks].max() * scale

        if show_plot:
            fig, ax = self._plot_differential_cutoff(
                x_interp, dydx_interp, dydx_cutoff, xlabel, ylabel
            )
            plt.show()

        return dydx_cutoff

    def _plot_differential_cutoff(self, x, dydx, dydx_cutoff, xlabel=None, ylabel=None):
        """Plot the differential with cutoff line."""
        if xlabel is None:
            xlabel = "x"
        if ylabel is None:
            ylabel = "dy/dx"

        fig, ax = plt.subplots()
        ax.plot(x, dydx)
        ax.axhline(dydx_cutoff, color="tab:red", linestyle="--")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, dydx_cutoff * 1.5)
        return fig, ax

    def to_config(self) -> dict:
        """
        Convert the OCPDataLoader back to parser configuration format.

        Returns
        -------
        dict
            Configuration dictionary that can recreate this OCPDataLoader.
            If loaded from database, returns {"data": "db:measurement_id", ...}.
            Otherwise, returns {"data": {...}, ...}.
        """
        # If loaded from database, use db reference instead of embedding data
        if hasattr(self, "_measurement_id") and self._measurement_id is not None:
            config = {
                "data": f"db:{self._measurement_id}",
            }
        else:
            config = {
                "data": self.data.to_dict(orient="list"),
            }

        # Add options if any were provided
        options = {}
        if hasattr(self, "_filters") and self._filters is not None:
            options["filters"] = self._filters
        if hasattr(self, "_interpolate") and self._interpolate is not None:
            # Convert numpy array to list for JSON serialization
            if isinstance(self._interpolate, np.ndarray):
                options["interpolate"] = self._interpolate.tolist()
            else:
                options["interpolate"] = self._interpolate
        if hasattr(self, "_capacity_column") and self._capacity_column is not None:
            options["capacity_column"] = self._capacity_column
        if hasattr(self, "_sort") and self._sort:
            options["sort"] = self._sort
        if hasattr(self, "_remove_duplicates") and self._remove_duplicates:
            options["remove_duplicates"] = self._remove_duplicates
        if hasattr(self, "_remove_extremes") and self._remove_extremes:
            options["remove_extremes"] = self._remove_extremes

        if options:
            config["options"] = options

        return config

    def copy(self) -> OCPDataLoader:
        """
        Create a copy of the OCPDataLoader instance.

        Returns
        -------
        OCPDataLoader
            A new instance with copied data.
        """
        return OCPDataLoader(data=self.data.copy())

    @classmethod
    def from_db(
        cls,
        measurement_id: str,
        options: dict | None = None,
        use_cache: bool = True,
    ) -> OCPDataLoader:
        """
        Load OCP data from the Ionworks database.

        Parameters
        ----------
        measurement_id : str
            The ID of the measurement to load from the database.
        options : dict | None, optional
            Options to pass to the OCPDataLoader constructor.
        use_cache : bool, optional
            If True (default), use local file cache to avoid repeated API calls.
            Set to False to force a fresh load from the database.

        Returns
        -------
        OCPDataLoader
            An OCPDataLoader instance with the loaded data.
        """
        # Try loading from cache first
        cached_data = None
        if use_cache:
            cached_data = _load_from_cache(measurement_id)

        if cached_data is not None:
            time_series = cached_data.get("time_series")
        else:
            from ionworks import Ionworks

            client = Ionworks()
            measurement_detail = client.cell_measurement.detail(measurement_id)
            time_series = measurement_detail.time_series

            # Save to cache for future use
            if use_cache:
                _save_to_cache(measurement_id, {"time_series": time_series})

        options = options or {}
        instance = cls(time_series, **options)

        # Store measurement_id for reverse parsing
        instance._measurement_id = measurement_id  # noqa: SLF001
        return instance


class DataLoader(GenericDataLoader):
    def __init__(
        self,
        time_series: pd.DataFrame | pl.DataFrame | dict,
        steps: pd.DataFrame | pl.DataFrame | dict,
        **kwargs,
    ):
        """
        Initialize a DataLoader object to process data for a specific test.

        Parameters
        ----------
        time_series : pd.DataFrame | pl.DataFrame | dict
            Dataframe containing the time series data. Can be Pandas or Polars DataFrame.
        steps : pd.DataFrame | pl.DataFrame | dict
            Dataframe containing the step information. Can be Pandas or Polars DataFrame.
        options : dict, optional
            A dictionary containing optional parameters:

            - first_step : str | int
                The first step to include in the data. Can be an int (direct step index)
                or a str (SQL query to select one step from the "steps" table).
            - last_step : str | int
                The last step to include in the data. Can be an int (direct step index)
                or a str (SQL query to select one step from the "steps" table).
            - filters : dict
                Dictionary containing filter configuration. Keys are variable names,
                values are dicts with "filter_type" (e.g., "savgol") and "parameters".
            - interpolate : float | np.ndarray
                The knots at which to interpolate the data, in seconds. If a float,
                data is interpolated at regular intervals. If an array, at specified knots.
        """
        # Support passing options as a dict, with kwargs taking precedence
        options = kwargs.pop("options", None) or {}
        merged = {**options, **kwargs}  # kwargs override options

        # Extract options from merged dict
        first_step = merged.get("first_step")
        last_step = merged.get("last_step")
        first_step_dict = merged.get("first_step_dict")
        last_step_dict = merged.get("last_step_dict")
        filters = merged.get("filters")
        interpolate = merged.get("interpolate")

        # Convert to pandas for PiecewiseLinearTimeseries class methods which rely on:
        # - pandas .iloc indexing for row access
        # - pandas filtering and boolean indexing patterns
        # - scipy interpolation functions expecting pandas DataFrames
        # - Backward compatibility with existing code expecting pandas interface
        if isinstance(time_series, pl.DataFrame):
            time_series = time_series.to_pandas()
        else:
            time_series = pd.DataFrame(time_series)

        if isinstance(steps, pl.DataFrame):
            steps = steps.to_pandas()
        else:
            steps = pd.DataFrame(steps)

        # Store original unfiltered data and parameters for reverse parsing
        self._original_time_series = time_series.copy()
        self._original_steps = steps.copy()
        self._filters = filters
        self._interpolate = interpolate

        # Handle parameter conflicts and deprecation warnings
        if first_step is not None and first_step_dict is not None:
            raise ValueError("Cannot specify both first_step and first_step_dict")
        if last_step is not None and last_step_dict is not None:
            raise ValueError("Cannot specify both last_step and last_step_dict")

        # Use deprecated parameters if new ones not provided
        if first_step_dict is not None:
            warnings.warn(
                "first_step_dict is deprecated and will be removed in a future version of ionworkspipeline. "
                "Use first_step instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            first_step = first_step_dict
        if last_step_dict is not None:
            warnings.warn(
                "last_step_dict is deprecated and will be removed in a future version of ionworkspipeline. "
                "Use last_step instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            last_step = last_step_dict

        # Store the final first_step and last_step for reverse parsing
        self._first_step = first_step
        self._last_step = last_step

        # Determine the first and last steps to include
        first_step_idx = self._get_step(first_step, steps, first=True)
        last_step_idx = self._get_step(last_step, steps, first=False)

        # Use Polars for efficient row slicing, then convert back to pandas
        # (PiecewiseLinearTimeseries stores steps as pandas for .iloc access)
        steps_pl = pl.from_pandas(steps)
        self.steps = steps_pl.slice(
            first_step_idx, last_step_idx - first_step_idx + 1
        ).to_pandas()

        if first_step_idx == 0:
            self.initial_voltage = self.steps.iloc[0]["Start voltage [V]"]
        else:
            self.initial_voltage = steps.iloc[first_step_idx - 1]["End voltage [V]"]

        start_idx = int(self.steps["Start index"].iloc[0])
        end_idx = int(self.steps["End index"].iloc[-1]) + 1
        self._start_idx = start_idx
        self._end_idx = end_idx

        # Load and filter the data - use pandas iloc to preserve index behavior
        data = time_series.iloc[start_idx:end_idx]
        super().__init__(data, filters, interpolate)

    def to_config(self, filter_data: bool = True) -> dict:
        """
        Convert the DataLoader back to parser configuration format.

        Parameters
        ----------
        filter_data : bool, optional
            If True (default), saves the filtered data (self.data, self.steps) rather
            than the original unfiltered data. This ensures that when the config is
            re-parsed, the same filtered data is used regardless of whether it's parsed
            as DataLoader or OCPDataLoader.
            If False, saves the original unfiltered data along with first_step/last_step
            options to allow re-filtering.

        Returns
        -------
        dict
            Configuration dictionary that can recreate this DataLoader.
            If loaded from database, returns {"data": "db:measurement_id", ...}.
            Otherwise, returns {"data": {"time_series": ..., "steps": ...}, ...}.
        """
        # If loaded from database, use db reference instead of embedding data
        if hasattr(self, "_measurement_id") and self._measurement_id is not None:
            config = {
                "data": f"db:{self._measurement_id}",
            }
            # Add options for database-loaded data
            options = {}
            if hasattr(self, "_first_step") and self._first_step is not None:
                options["first_step"] = self._first_step
            if hasattr(self, "_last_step") and self._last_step is not None:
                options["last_step"] = self._last_step
            if hasattr(self, "_filters") and self._filters is not None:
                options["filters"] = self._filters
            if hasattr(self, "_interpolate") and self._interpolate is not None:
                if isinstance(self._interpolate, np.ndarray):
                    options["interpolate"] = self._interpolate.tolist()
                else:
                    options["interpolate"] = self._interpolate
            if options:
                config["options"] = options
        elif filter_data:
            # Save the filtered data with reset indices so it can be loaded standalone
            # Reset time_series index to 0-based
            time_series_reset = self.data.reset_index(drop=True)

            # Adjust steps indices to match the reset time_series
            # The new start index is relative to the filtered data (0-based)
            steps_reset = self.steps.copy()
            if len(steps_reset) > 0:
                original_start = int(steps_reset["Start index"].iloc[0])
                steps_reset["Start index"] = steps_reset["Start index"] - original_start
                steps_reset["End index"] = steps_reset["End index"] - original_start

            config = {
                "data": {
                    "time_series": time_series_reset.to_dict(orient="list"),
                    "steps": steps_reset.to_dict(orient="list"),
                },
            }

            # Add options for filters and interpolation (but NOT first_step/last_step
            # since the data is already filtered)
            options = {}
            if hasattr(self, "_filters") and self._filters is not None:
                options["filters"] = self._filters
            if hasattr(self, "_interpolate") and self._interpolate is not None:
                # Convert numpy array to list for JSON serialization
                if isinstance(self._interpolate, np.ndarray):
                    options["interpolate"] = self._interpolate.tolist()
                else:
                    options["interpolate"] = self._interpolate

            if options:
                config["options"] = options
        else:
            # Save the original unfiltered data with filtering options
            if hasattr(self, "_original_time_series") and hasattr(
                self, "_original_steps"
            ):
                config = {
                    "data": {
                        "time_series": self._original_time_series.to_dict(
                            orient="list"
                        ),
                        "steps": self._original_steps.to_dict(orient="list"),
                    },
                }
            else:
                # Fallback for DataLoaders created before this change
                config = {
                    "data": {
                        "time_series": self.data.to_dict(orient="list"),
                        "steps": self.steps.to_dict(orient="list"),
                    },
                }

            # Add all options including first_step/last_step
            options = {}
            if hasattr(self, "_first_step") and self._first_step is not None:
                options["first_step"] = self._first_step
            if hasattr(self, "_last_step") and self._last_step is not None:
                options["last_step"] = self._last_step
            if hasattr(self, "_filters") and self._filters is not None:
                options["filters"] = self._filters
            if hasattr(self, "_interpolate") and self._interpolate is not None:
                # Convert numpy array to list for JSON serialization
                if isinstance(self._interpolate, np.ndarray):
                    options["interpolate"] = self._interpolate.tolist()
                else:
                    options["interpolate"] = self._interpolate

            if options:
                config["options"] = options

        return config

    @property
    def start_idx(self) -> int:
        """Get the start index for the data."""
        return self._start_idx

    @start_idx.setter
    def start_idx(self, value: int) -> None:
        """Set the start index for the data."""
        self._start_idx = value

    @property
    def end_idx(self) -> int:
        """Get the end index for the data."""
        return self._end_idx

    @end_idx.setter
    def end_idx(self, value: int) -> None:
        """Set the end index for the data."""
        self._end_idx = value

    @staticmethod
    def _get_step_from_cycle(
        cycle: int, steps: pd.DataFrame | pl.DataFrame, first: bool
    ) -> int:
        # Convert to Polars for faster filtering
        steps_pl = steps if isinstance(steps, pl.DataFrame) else pl.from_pandas(steps)
        steps_per_cycle = steps_pl.filter(pl.col("Cycle count") == cycle)
        if steps_per_cycle.height == 0:
            raise ValueError(f"No steps found for cycle {cycle}")
        if first:
            return steps_per_cycle["Step count"][0]
        return steps_per_cycle["Step count"][-1]

    @staticmethod
    def _get_step_from_step(
        step: int, steps: pd.DataFrame | pl.DataFrame, first: bool
    ) -> int:
        return step

    @staticmethod
    def _get_step_from(
        step_dict: dict,
        all_steps: pd.DataFrame | pl.DataFrame,
        first: bool,
    ) -> int:
        match list(step_dict.keys())[0]:
            case "cycle":
                return DataLoader._get_step_from_cycle(
                    list(step_dict.values())[0],
                    all_steps,
                    first,
                )
            case "step":
                return DataLoader._get_step_from_step(
                    list(step_dict.values())[0],
                    all_steps,
                    first,
                )
            case _:
                raise ValueError(f"Unknown step type: {list(step_dict.keys())[0]}")

    @staticmethod
    def _get_step(
        step_param: str | int | dict | None,
        all_steps: pd.DataFrame | pl.DataFrame,
        first: bool,
    ) -> int:
        if step_param is None:
            return 0 if first else len(all_steps) - 1

        if isinstance(step_param, int):
            return step_param

        if isinstance(step_param, str):
            # Treat as SQL query
            try:
                # Convert to Polars for SQL context
                steps_pl = (
                    all_steps
                    if isinstance(all_steps, pl.DataFrame)
                    else pl.from_pandas(all_steps)
                )

                # Execute SQL query using Polars SQL context
                ctx = pl.SQLContext(steps=steps_pl)
                result = ctx.execute(step_param).collect()

                # Validate exactly 1 row returned
                if len(result) == 0:
                    raise ValueError(f"SQL query returned no results: {step_param}")
                if len(result) > 1:
                    raise ValueError(
                        f"SQL query returned {len(result)} results, expected exactly 1: {step_param}"
                    )

                # Extract and return the "Step count" value
                step_count = result["Step count"][0]
                return step_count

            except Exception as e:
                raise ValueError(
                    f"Error executing SQL query '{step_param}': {e}"
                ) from e

        if isinstance(step_param, dict):
            # Handle deprecated dict format
            warnings.warn(
                "Using dict format for step selection is deprecated and will be removed in a future version of ionworkspipeline. "
                "Use string SQL queries or integer step indices instead.",
                DeprecationWarning,
                stacklevel=3,
            )
            return DataLoader._get_step_from(step_param, all_steps, first)

        raise TypeError(f"Unsupported step parameter type: {type(step_param)}")

    def generate_experiment(self, use_cv: bool = False) -> pybamm.Experiment:
        """
        Generate a PyBaMM experiment from the loaded step information.

        This method creates a PyBaMM Experiment object by converting each step in the test
        data into an appropriate PyBaMM step object based on its type.

        Parameters
        ----------
        use_cv : bool, optional
            If True, use PyBaMM's Voltage step for CV steps. If False (default),
            use a current interpolant based on measured current values instead.

        Returns
        -------
        pybamm.Experiment
            A PyBaMM Experiment object containing all the steps from the test,
            suitable for simulation with PyBaMM models.

        Raises
        ------
        ValueError
            If an unknown step type is encountered in the data.
        """
        steps = []
        for _, step in self.steps.iterrows():
            duration = step["Duration [s]"]
            step_type = step["Step type"]
            if duration <= np.nextafter(0, 1):
                continue
            match step_type:
                case "Constant current discharge" | "Constant current charge":
                    mean_current = step["Mean current [A]"]
                    step = pybamm.step.Current(mean_current, duration=duration)
                case "Constant voltage discharge" | "Constant voltage charge":
                    # by default use a current interpolant, if specified use a cv step
                    if use_cv:
                        mean_voltage = step["Mean voltage [V]"]
                        step = pybamm.step.Voltage(mean_voltage, duration=duration)
                    else:
                        step = self._create_current_interpolant_step(step, duration)
                case "Rest":
                    step = pybamm.step.Current(0, duration=duration)
                case "Constant power discharge" | "Constant power charge":
                    mean_power = step["Mean power [W]"]
                    step = pybamm.step.Power(mean_power, duration=duration)
                case "EIS":
                    raise NotImplementedError(
                        "EIS steps are not yet implemented directly in PyBaMM experiments. Please simulate EIS using the pybamm-eis package."
                    )
                case _:
                    logger.warning(
                        f"Unknown step type: {step_type}, falling back to current interpolant ",
                    )
                    step = self._create_current_interpolant_step(step, duration)

            steps.append(step)
        return pybamm.Experiment(steps)

    def generate_interpolant(self) -> pybamm.Interpolant:
        """
        Generate a PyBaMM interpolant from the loaded step information.

        Parameters
        ----------

        Returns
        -------
        pybamm.Interpolant
            A PyBaMM interpolant object containing all the steps from the test,
            suitable for simulation with PyBaMM models.

        Raises
        ------
            If an unknown step type is encountered in the data.
        """
        ts = []
        cs = []
        for _, step in self.steps.iterrows():
            if (
                "constant current" in step["Step type"].lower()
                or step["Step type"] == "Rest"
            ):
                first_t = step["Start time [s]"]
                last_t = step["End time [s]"]
                ts.append(np.array([first_t, last_t]))
                cs.append(
                    np.array([step["Mean current [A]"], step["Mean current [A]"]])
                )
            else:
                times, currents = self._get_times_and_currents(step)
                ts.append(times)
                cs.append(currents)
        ts = np.concatenate(ts)
        cs = np.concatenate(cs)
        ts = monotonic_time_offset(ts, 0.0, offset_initial_time=False)
        return pybamm.Interpolant(ts, cs, pybamm.t)

    def plot_data(self, show: bool = False) -> tuple[plt.Figure, plt.Axes]:
        """
        Plot voltage vs time data from the loaded experiment.

        Creates a simple matplotlib plot showing voltage over time for the
        entire loaded dataset. This provides a quick visualization of the
        voltage profile during the experiment.

        Parameters
        ----------
        show : bool, optional
            If True, displays the plot immediately using plt.show().
            If False (default), just creates the plot objects for further
            customization or saving.

        Returns
        -------
        tuple[plt.Figure, plt.Axes]
            A tuple containing the matplotlib Figure and Axes objects,
            allowing for further customization of the plot if needed.
        """
        fig, ax = plt.subplots(3, 1, sharex=True)
        ax[0].plot(self.data["Time [s]"], self.data["Voltage [V]"])
        ax[0].set_ylabel("Voltage [V]")
        ax[1].plot(self.data["Time [s]"], self.data["Current [A]"])
        ax[1].set_ylabel("Current [A]")
        ax[2].plot(self.data["Time [s]"], self.data["Temperature [degC]"])
        ax[2].set_xlabel("Time [s]")
        ax[2].set_ylabel("Temperature [degC]")
        if show:
            plt.show()
        return fig, ax

    def _get_times_and_currents(
        self, step: pd.DataFrame | pl.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        start_idx = int(step["Start index"]) - self._start_idx
        end_idx = int(step["End index"]) + 1 - self._start_idx
        current = self.data["Current [A]"].iloc[start_idx:end_idx].array
        time = self.data["Time [s]"].iloc[start_idx:end_idx].array
        return time, current

    def _create_current_interpolant_step(
        self, step: pd.DataFrame | pl.DataFrame, duration: float
    ) -> pybamm.step.BaseStep:
        time, current = self._get_times_and_currents(step)
        time = monotonic_time_offset(time, 0.0, offset_initial_time=False)
        arr = np.transpose(np.vstack((time, current)))
        return pybamm.step.Current(arr, duration=duration)

    @classmethod
    def from_local(
        cls,
        data_path: str,
        options: dict | None = None,
        use_polars: bool = False,
    ) -> DataLoader:
        """
        Load data from local filesystem.

        Parameters
        ----------
        data_path : str
            Path to the directory containing time_series.csv and steps.csv files.
        options : dict | None, optional
            Options to pass to the DataLoader constructor (e.g., first_step, last_step,
            filters, interpolate).
        use_polars : bool, optional
            If True, read data using Polars. Default is False (uses Pandas).

        Returns
        -------
        DataLoader
            A DataLoader instance with the loaded data.
        """
        if use_polars:
            time_series = pl.read_csv(Path(data_path) / "time_series.csv")
            steps = pl.read_csv(Path(data_path) / "steps.csv")
        else:
            time_series = pd.read_csv(Path(data_path) / "time_series.csv")
            steps = pd.read_csv(Path(data_path) / "steps.csv")
        options = options or {}
        return cls(time_series, steps, **options)

    @classmethod
    def from_db(
        cls,
        measurement_id: str,
        options: dict | None = None,
        use_cache: bool = True,
    ) -> DataLoader:
        """
        Load data from the Ionworks database.

        Parameters
        ----------
        measurement_id : str
            The ID of the measurement to load from the database.
        options : dict | None, optional
            Options to pass to the DataLoader constructor.
        use_cache : bool, optional
            If True (default), use local file cache to avoid repeated API calls.
            Set to False to force a fresh load from the database.

        Returns
        -------
        DataLoader
            A DataLoader instance with the loaded data.
        """
        # Try loading from cache first
        cached_data = None
        if use_cache:
            cached_data = _load_from_cache(measurement_id)

        if cached_data is not None:
            time_series = cached_data.get("time_series")
            steps = cached_data.get("steps")
        else:
            from ionworks import Ionworks

            client = Ionworks()
            measurement_detail = client.cell_measurement.detail(measurement_id)
            time_series = measurement_detail.time_series
            steps = measurement_detail.steps

            # Save to cache for future use
            if use_cache:
                _save_to_cache(
                    measurement_id, {"time_series": time_series, "steps": steps}
                )

        options = options or {}
        instance = cls(
            time_series,
            steps,
            **options,
        )

        # Store measurement_id for reverse parsing
        instance._measurement_id = measurement_id  # noqa: SLF001
        return instance

    @classmethod
    def from_processed_data(
        cls,
        data: pd.DataFrame | pl.DataFrame,
        steps: pd.DataFrame | pl.DataFrame,
        initial_voltage: float,
        start_idx: int,
        end_idx: int,
    ) -> DataLoader:
        """
        Create a DataLoader from already-processed data.

        This method bypasses the normal constructor logic and creates a DataLoader
        directly from processed data and steps, avoiding slicing issues.

        Parameters
        ----------
        data : pd.DataFrame | pl.DataFrame
            The processed time series data
        steps : pd.DataFrame | pl.DataFrame
            The processed steps data
        initial_voltage : float
            The initial voltage value
        start_idx : int
            The start index for the data
        end_idx : int
            The end index for the data

        Returns
        -------
        DataLoader
            A new DataLoader instance with the provided data
        """
        # Create a new instance without calling __init__
        instance = cls.__new__(cls)

        # Set up the basic attributes
        instance.data = data.copy()
        instance.steps = steps.copy()
        instance.initial_voltage = initial_voltage
        instance.start_idx = start_idx
        instance.end_idx = end_idx

        # Initialize the GenericDataLoader part (no filters/interpolation)
        super(DataLoader, instance).__init__(instance.data)

        return instance

    def copy(self) -> DataLoader:
        """
        Create a copy of the DataLoader instance.

        Returns
        -------
        DataLoader
            A new instance with copied data and steps.
        """
        return DataLoader.from_processed_data(
            data=self.data.copy(),
            steps=self.steps.copy(),
            initial_voltage=self.initial_voltage,
            start_idx=self._start_idx,
            end_idx=self._end_idx,
        )
