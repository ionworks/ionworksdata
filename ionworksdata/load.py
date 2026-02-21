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


class DataLoader(GenericDataLoader):
    """
    Unified data loader for time-series and OCP data.

    Handles two modes:
    - **With steps**: loads time-series data with step information for simulation,
      experiment generation, etc.
    - **Without steps**: loads simple tabular data (e.g. OCP curves with Capacity
      and Voltage columns).

    Post-load preprocessing is configured via the ``transforms`` dict option.

    Parameters
    ----------
    time_series : pd.DataFrame | pl.DataFrame | dict
        The data to load. Can be a Pandas/Polars DataFrame or a dict.
    steps : pd.DataFrame | pl.DataFrame | dict | None, optional
        Step information. When None, the loader operates in simple (no-steps) mode.
    **kwargs
        Options passed directly or via an ``options`` dict. Supported keys:

        When steps are provided:
            - first_step, last_step : str | int
            - first_step_dict, last_step_dict : dict (deprecated)

        When steps are None:
            - capacity_column : str

        Always:
            - transforms : dict with any of:
                - gitt_to_ocp : bool
                - sort : bool
                - remove_duplicates : bool
                - remove_extremes : bool
                - filters : dict
                - interpolate : float | np.ndarray
    """

    def __init__(
        self,
        time_series: pd.DataFrame | pl.DataFrame | dict,
        steps: pd.DataFrame | pl.DataFrame | dict | None = None,
        **kwargs,
    ):
        options = kwargs.pop("options", None) or {}
        merged = {**options, **kwargs}

        transforms = dict(merged.get("transforms") or {})
        # Backward compat: top-level filters/interpolate migrate into transforms
        if "filters" in merged and "filters" not in transforms:
            transforms["filters"] = merged["filters"]
        if "interpolate" in merged and "interpolate" not in transforms:
            transforms["interpolate"] = merged["interpolate"]
        self._transforms = transforms
        self._measurement_id = None

        if steps is not None:
            data = self._init_with_steps(time_series, steps, merged)
        else:
            data = self._init_without_steps(time_series, merged)

        super().__init__(data)
        self._apply_transforms()

    # ------------------------------------------------------------------
    # Initialization paths
    # ------------------------------------------------------------------

    def _init_with_steps(self, time_series, steps, options):
        if isinstance(time_series, pl.DataFrame):
            time_series = time_series.to_pandas()
        else:
            time_series = pd.DataFrame(time_series)
        if isinstance(steps, pl.DataFrame):
            steps = steps.to_pandas()
        else:
            steps = pd.DataFrame(steps)

        self._original_time_series = time_series.copy()
        self._original_steps = steps.copy()

        first_step = options.get("first_step")
        last_step = options.get("last_step")
        first_step_dict = options.get("first_step_dict")
        last_step_dict = options.get("last_step_dict")

        if first_step is not None and first_step_dict is not None:
            raise ValueError("Cannot specify both first_step and first_step_dict")
        if last_step is not None and last_step_dict is not None:
            raise ValueError("Cannot specify both last_step and last_step_dict")

        if first_step_dict is not None:
            warnings.warn(
                "first_step_dict is deprecated. Use first_step instead.",
                DeprecationWarning,
                stacklevel=3,
            )
            first_step = first_step_dict
        if last_step_dict is not None:
            warnings.warn(
                "last_step_dict is deprecated. Use last_step instead.",
                DeprecationWarning,
                stacklevel=3,
            )
            last_step = last_step_dict

        self._first_step = first_step
        self._last_step = last_step

        first_step_idx = self._get_step(first_step, steps, first=True)
        last_step_idx = self._get_step(last_step, steps, first=False)

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

        return time_series.iloc[start_idx:end_idx]

    def _init_without_steps(self, time_series, options):
        if isinstance(time_series, dict):
            time_series = pd.DataFrame(time_series)
        if isinstance(time_series, pl.DataFrame):
            time_series = time_series.to_pandas()

        self.steps = None
        self._original_time_series = None
        self._original_steps = None
        self._first_step = None
        self._last_step = None
        self._start_idx = 0
        self._end_idx = len(time_series)
        self.initial_voltage = None

        # Voltage column aliasing
        potential_ocp_column_names = [
            "Voltage [V]",
            "OCP [V]",
            "OCV [V]",
            "Open-circuit potential [V]",
            "Open-circuit voltage [V]",
        ]
        if "Voltage [V]" not in time_series.columns:
            for column in potential_ocp_column_names:
                if column in time_series.columns:
                    time_series["Voltage [V]"] = time_series[column]
                    break
            else:
                raise ValueError(
                    "Data must contain one of the following columns: "
                    + ", ".join(potential_ocp_column_names)
                )

        # Capacity column aliasing
        capacity_column = options.get("capacity_column")
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
            if capacity_column in time_series.columns:
                time_series["Capacity [A.h]"] = time_series[capacity_column]
            else:
                raise ValueError(
                    f"Specified capacity_column '{capacity_column}' not found in data. "
                    f"Available columns: {list(time_series.columns)}"
                )
        elif "Capacity [A.h]" not in time_series.columns:
            for column in potential_capacity_column_names:
                if column in time_series.columns:
                    time_series["Capacity [A.h]"] = time_series[column]
                    break

        return time_series

    # ------------------------------------------------------------------
    # Transforms
    # ------------------------------------------------------------------

    def _apply_transforms(self):
        transforms = self._transforms
        if transforms.get("gitt_to_ocp"):
            self._transform_gitt_to_ocp()
        if transforms.get("sort"):
            self.data = self._sort_capacity_and_ocp(self.data)
        if transforms.get("remove_duplicates"):
            self.data = self._remove_duplicate_ocp(self.data)
        if transforms.get("remove_extremes"):
            self.data = self._remove_ocp_extremes(self.data)
        filters = transforms.get("filters")
        if filters:
            self.data = self.filter_data(self.data, filters)
        interpolate = transforms.get("interpolate")
        if interpolate is not None:
            self.data = self.interpolate_data(self.data, interpolate)

    def _transform_gitt_to_ocp(self):
        """Extract OCP from GITT rest steps: take the last data point of each rest."""
        if self.steps is None:
            raise ValueError("gitt_to_ocp requires steps data")

        gitt_rest = self.steps[
            (self.steps["Label"] == "GITT") & (self.steps["Step type"] == "Rest")
        ]
        if len(gitt_rest) == 0:
            raise ValueError("No GITT rest steps found in data")

        ocp_points = []
        for _, step in gitt_rest.iterrows():
            end_idx = int(step["End index"]) - self._start_idx
            row = self.data.iloc[end_idx]
            discharge = row.get("Discharge capacity [A.h]", 0)
            charge = row.get("Charge capacity [A.h]", 0)
            capacity = abs(discharge - charge)
            ocp_points.append({
                "Capacity [A.h]": capacity,
                "Voltage [V]": row["Voltage [V]"],
            })

        ocp_df = pd.DataFrame(ocp_points)
        ocp_df = ocp_df.sort_values("Capacity [A.h]").reset_index(drop=True)
        ocp_df["Capacity [A.h]"] -= ocp_df["Capacity [A.h]"].iloc[0]

        self.data = ocp_df
        self.steps = None

    @staticmethod
    def _remove_duplicate_ocp(data, capacity_column_name="Capacity [A.h]"):
        """Remove any duplicate capacity and voltage values."""
        data = data.drop_duplicates(subset=[capacity_column_name])
        data = data.drop_duplicates(subset=["Voltage [V]"])
        return data

    @staticmethod
    def _sort_capacity_and_ocp(data):
        """Sort OCP data so voltage is decreasing and capacity is increasing."""
        V = data["Voltage [V]"].values
        if V[-1] > V[0]:
            data = data.iloc[::-1].reset_index(drop=True)

        capacity_column_names = [col for col in data if col.startswith("Capacity [")]
        if len(capacity_column_names) == 0:
            raise ValueError("No capacity column found")
        elif len(capacity_column_names) > 1:
            raise ValueError(
                f"Multiple capacity columns found: {capacity_column_names}"
            )
        capacity_column_name = capacity_column_names[0]

        Q = data[capacity_column_name].values
        if Q[0] > Q[-1]:
            Q = Q.max() - Q
        Q -= Q.min()
        data[capacity_column_name] = Q

        data = DataLoader._remove_duplicate_ocp(data, capacity_column_name)
        return data

    @staticmethod
    def _remove_ocp_extremes(data):
        """Remove data at extremes where the second derivative of voltage vs
        capacity is zero."""
        q = data["Capacity [A.h]"].values
        U = data["Voltage [V]"].values
        d2UdQ2 = np.gradient(np.gradient(U, q), q)
        first_positive, last_positive = np.where(abs(d2UdQ2) > 1e-10)[0][[0, -1]]
        return data.iloc[first_positive : last_positive + 1]

    # ------------------------------------------------------------------
    # Differential cutoff methods
    # ------------------------------------------------------------------

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
            Whether to show a plot of the dUdQ values with the cut-off.
        options : dict, optional
            Dictionary of options to pass to the method.

        Returns
        -------
        float
            Cut-off for dUdQ
        """
        q = self.data["Capacity [A.h]"]
        U = self.data["Voltage [V]"]
        dUdQ = abs(np.gradient(U, q))
        return self._calculate_differential_cutoff(
            q, U, dUdQ, method, show_plot, "Capacity [A.h]", "Voltage [V]", options
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
            Whether to show a plot of the dQdU values with the cut-off.
        options : dict, optional
            Dictionary of options to pass to the method.

        Returns
        -------
        float
            Cut-off for dQdU
        """
        U = self.data["Voltage [V]"]
        q = self.data["Capacity [A.h]"]
        dQdU = abs(np.gradient(q, U))
        return self._calculate_differential_cutoff(
            U, q, dQdU, method, show_plot, "Voltage [V]", "Capacity [A.h]", options
        )

    def _calculate_differential_cutoff(
        self, x, y, dydx, method="explicit", show_plot=False,
        xlabel=None, ylabel=None, options=None,
    ) -> float:
        options = options or {}
        if method == "explicit":
            return self._calculate_differential_cutoff_explicit(
                x, y, dydx, show_plot=show_plot, xlabel=xlabel, ylabel=ylabel, **options
            )
        elif method == "quantile":
            return self._calculate_differential_cutoff_quantile(
                x, y, dydx, show_plot=show_plot, xlabel=xlabel, ylabel=ylabel, **options
            )
        elif method == "peaks":
            return self._calculate_differential_cutoff_peaks(
                x, y, dydx, show_plot=show_plot, xlabel=xlabel, ylabel=ylabel, **options
            )
        raise ValueError(f"Method {method} not recognized")

    def _calculate_differential_cutoff_explicit(
        self, x, y, dydx, show_plot=False, xlabel=None, ylabel=None,
        lower_ratio=0.1, upper_ratio=0.9, scale=1.1,
    ) -> float:
        x = np.array(x)
        x_scaled = (x - x.min()) / (x.max() - x.min())
        xmin_idx = np.argmin(np.abs(x_scaled - lower_ratio))
        xmax_idx = np.argmin(np.abs(x_scaled - upper_ratio))
        if xmin_idx > xmax_idx:
            xmin_idx, xmax_idx = xmax_idx, xmin_idx
        dydx_cutoff = dydx[xmin_idx:xmax_idx].max() * scale
        if show_plot:
            self._plot_differential_cutoff(x, dydx, dydx_cutoff, xlabel, ylabel)
            plt.show()
        return dydx_cutoff

    def _calculate_differential_cutoff_quantile(
        self, x, y, dydx, show_plot=False, xlabel=None, ylabel=None,
        quantile=0.8, scale=2,
    ) -> float:
        x = np.array(x)
        x_interp = np.linspace(x.min(), x.max(), 1000)
        dydx_interp = interp1d(x, dydx, kind="linear")(x_interp)
        quantile_value = np.percentile(dydx_interp, quantile * 100)
        dydx_cutoff = scale * quantile_value
        if show_plot:
            fig, ax = self._plot_differential_cutoff(
                x_interp, dydx_interp, dydx_cutoff, xlabel, ylabel
            )
            ax.axhline(quantile_value, color="tab:blue", linestyle="--",
                        label="Quantile", zorder=10)
            ax.axhline(dydx_cutoff, color="tab:red", linestyle="--",
                        label="Cut-off", zorder=10)
            ax.legend()
            plt.show()
        return dydx_cutoff

    def _calculate_differential_cutoff_peaks(
        self, x, y, dydx, show_plot=False, xlabel=None, ylabel=None,
        scale=1.1, **kwargs,
    ) -> float:
        x = np.array(x)
        x_interp = np.linspace(x.min(), x.max(), 1000)
        dydx_interp = interp1d(x, dydx, kind="linear")(x_interp)
        peaks, _ = find_peaks(dydx_interp, **kwargs)
        dydx_cutoff = dydx_interp[peaks].max() * scale
        if show_plot:
            self._plot_differential_cutoff(
                x_interp, dydx_interp, dydx_cutoff, xlabel, ylabel
            )
            plt.show()
        return dydx_cutoff

    def _plot_differential_cutoff(self, x, dydx, dydx_cutoff, xlabel=None, ylabel=None):
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

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_config(self, filter_data: bool = True) -> dict:
        """
        Convert the DataLoader back to parser configuration format.

        Parameters
        ----------
        filter_data : bool, optional
            If True (default) and steps are present, saves the filtered data
            rather than the original unfiltered data.

        Returns
        -------
        dict
            Configuration dictionary that can recreate this DataLoader.
        """
        if hasattr(self, "_measurement_id") and self._measurement_id is not None:
            config = {"data": f"db:{self._measurement_id}"}
            opts = self._build_options_for_config()
            if opts:
                config["options"] = opts
            return config

        if self.steps is None:
            config = {"data": self.data.to_dict(orient="list")}
            opts = self._build_options_for_config()
            if opts:
                config["options"] = opts
            return config

        # With-steps path
        if filter_data:
            time_series_reset = self.data.reset_index(drop=True)
            steps_reset = self.steps.copy()
            if len(steps_reset) > 0:
                original_start = int(steps_reset["Start index"].iloc[0])
                steps_reset["Start index"] -= original_start
                steps_reset["End index"] -= original_start
            config = {
                "data": {
                    "time_series": time_series_reset.to_dict(orient="list"),
                    "steps": steps_reset.to_dict(orient="list"),
                },
            }
        else:
            if self._original_time_series is not None:
                config = {
                    "data": {
                        "time_series": self._original_time_series.to_dict(orient="list"),
                        "steps": self._original_steps.to_dict(orient="list"),
                    },
                }
            else:
                config = {
                    "data": {
                        "time_series": self.data.to_dict(orient="list"),
                        "steps": self.steps.to_dict(orient="list"),
                    },
                }

        opts = self._build_options_for_config(include_step_options=not filter_data)
        if opts:
            config["options"] = opts
        return config

    def _build_options_for_config(self, include_step_options=True):
        opts = {}
        if include_step_options:
            if getattr(self, "_first_step", None) is not None:
                opts["first_step"] = self._first_step
            if getattr(self, "_last_step", None) is not None:
                opts["last_step"] = self._last_step
        if getattr(self, "_capacity_column", None) is not None:
            opts["capacity_column"] = self._capacity_column
        if self._transforms:
            serializable_transforms = {}
            for k, v in self._transforms.items():
                if isinstance(v, np.ndarray):
                    serializable_transforms[k] = v.tolist()
                else:
                    serializable_transforms[k] = v
            opts["transforms"] = serializable_transforms
        return opts

    # ------------------------------------------------------------------
    # Properties (with-steps mode)
    # ------------------------------------------------------------------

    @property
    def start_idx(self) -> int:
        return self._start_idx

    @start_idx.setter
    def start_idx(self, value: int) -> None:
        self._start_idx = value

    @property
    def end_idx(self) -> int:
        return self._end_idx

    @end_idx.setter
    def end_idx(self, value: int) -> None:
        self._end_idx = value

    # ------------------------------------------------------------------
    # Step selection helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_step_from_cycle(cycle, steps, first):
        steps_pl = steps if isinstance(steps, pl.DataFrame) else pl.from_pandas(steps)
        steps_per_cycle = steps_pl.filter(pl.col("Cycle count") == cycle)
        if steps_per_cycle.height == 0:
            raise ValueError(f"No steps found for cycle {cycle}")
        return steps_per_cycle["Step count"][0 if first else -1]

    @staticmethod
    def _get_step_from_step(step, steps, first):
        return step

    @staticmethod
    def _get_step_from(step_dict, all_steps, first):
        match list(step_dict.keys())[0]:
            case "cycle":
                return DataLoader._get_step_from_cycle(
                    list(step_dict.values())[0], all_steps, first,
                )
            case "step":
                return DataLoader._get_step_from_step(
                    list(step_dict.values())[0], all_steps, first,
                )
            case _:
                raise ValueError(f"Unknown step type: {list(step_dict.keys())[0]}")

    @staticmethod
    def _get_step(step_param, all_steps, first):
        if step_param is None:
            return 0 if first else len(all_steps) - 1
        if isinstance(step_param, int):
            return step_param
        if isinstance(step_param, str):
            try:
                steps_pl = (
                    all_steps
                    if isinstance(all_steps, pl.DataFrame)
                    else pl.from_pandas(all_steps)
                )
                ctx = pl.SQLContext(steps=steps_pl)
                result = ctx.execute(step_param).collect()
                if len(result) == 0:
                    raise ValueError(f"SQL query returned no results: {step_param}")
                if len(result) > 1:
                    raise ValueError(
                        f"SQL query returned {len(result)} results, expected exactly 1: {step_param}"
                    )
                return result["Step count"][0]
            except Exception as e:
                raise ValueError(
                    f"Error executing SQL query '{step_param}': {e}"
                ) from e
        if isinstance(step_param, dict):
            warnings.warn(
                "Using dict format for step selection is deprecated. "
                "Use string SQL queries or integer step indices instead.",
                DeprecationWarning,
                stacklevel=3,
            )
            return DataLoader._get_step_from(step_param, all_steps, first)
        raise TypeError(f"Unsupported step parameter type: {type(step_param)}")

    # ------------------------------------------------------------------
    # Experiment / interpolant generation (requires steps)
    # ------------------------------------------------------------------

    def generate_experiment(self, use_cv: bool = False) -> pybamm.Experiment:
        """Generate a PyBaMM experiment from the loaded step information."""
        if self.steps is None:
            raise ValueError("generate_experiment requires steps data")
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
                        "EIS steps are not yet implemented directly in PyBaMM "
                        "experiments. Please simulate EIS using the pybamm-eis package."
                    )
                case _:
                    logger.warning(
                        f"Unknown step type: {step_type}, "
                        "falling back to current interpolant",
                    )
                    step = self._create_current_interpolant_step(step, duration)
            steps.append(step)
        return pybamm.Experiment(steps)

    def generate_interpolant(self) -> pybamm.Interpolant:
        """Generate a PyBaMM interpolant from the loaded step information."""
        if self.steps is None:
            raise ValueError("generate_interpolant requires steps data")
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
        """Plot voltage vs time data from the loaded experiment."""
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

    def _get_times_and_currents(self, step):
        start_idx = int(step["Start index"]) - self._start_idx
        end_idx = int(step["End index"]) + 1 - self._start_idx
        current = self.data["Current [A]"].iloc[start_idx:end_idx].array
        time = self.data["Time [s]"].iloc[start_idx:end_idx].array
        return time, current

    def _create_current_interpolant_step(self, step, duration):
        time, current = self._get_times_and_currents(step)
        time = monotonic_time_offset(time, 0.0, offset_initial_time=False)
        arr = np.transpose(np.vstack((time, current)))
        return pybamm.step.Current(arr, duration=duration)

    # ------------------------------------------------------------------
    # Class methods
    # ------------------------------------------------------------------

    @classmethod
    def from_local(cls, data_path, options=None, use_polars=False):
        """
        Load data from local filesystem.

        Parameters
        ----------
        data_path : str
            Path to the directory containing time_series.csv and optionally
            steps.csv files.
        options : dict | None, optional
            Options to pass to the DataLoader constructor.
        use_polars : bool, optional
            If True, read data using Polars. Default is False (uses Pandas).

        Returns
        -------
        DataLoader
        """
        read_fn = pl.read_csv if use_polars else pd.read_csv
        time_series = read_fn(Path(data_path) / "time_series.csv")
        steps_path = Path(data_path) / "steps.csv"
        steps = read_fn(steps_path) if steps_path.exists() else None
        options = options or {}
        return cls(time_series, steps, **options)

    @classmethod
    def from_db(cls, measurement_id, options=None, use_cache=True):
        """
        Load data from the Ionworks database.

        Loads steps if available, otherwise creates a steps-less DataLoader.

        Parameters
        ----------
        measurement_id : str
            The ID of the measurement to load from the database.
        options : dict | None, optional
            Options to pass to the DataLoader constructor.
        use_cache : bool, optional
            If True (default), use local file cache to avoid repeated API calls.

        Returns
        -------
        DataLoader
        """
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
            steps = getattr(measurement_detail, "steps", None)

            cache_payload = {"time_series": time_series}
            if steps is not None:
                cache_payload["steps"] = steps
            if use_cache:
                _save_to_cache(measurement_id, cache_payload)

        options = options or {}
        instance = cls(time_series, steps, **options)
        instance._measurement_id = measurement_id  # noqa: SLF001
        return instance

    @classmethod
    def from_processed_data(
        cls, data, steps, initial_voltage, start_idx, end_idx,
    ):
        """
        Create a DataLoader from already-processed data, bypassing __init__.

        Parameters
        ----------
        data : pd.DataFrame | pl.DataFrame
            The processed time series data.
        steps : pd.DataFrame | pl.DataFrame | None
            The processed steps data (or None).
        initial_voltage : float
            The initial voltage value.
        start_idx : int
            The start index for the data.
        end_idx : int
            The end index for the data.

        Returns
        -------
        DataLoader
        """
        instance = cls.__new__(cls)
        instance.data = data.copy()
        if steps is not None:
            instance.steps = steps.copy()
        else:
            instance.steps = None
        instance.initial_voltage = initial_voltage
        instance.start_idx = start_idx
        instance.end_idx = end_idx
        instance._transforms = {}
        instance._measurement_id = None
        instance._capacity_column = None
        instance._first_step = None
        instance._last_step = None
        instance._original_time_series = None
        instance._original_steps = None
        super(DataLoader, instance).__init__(instance.data)
        return instance

    def copy(self) -> DataLoader:
        """Create a copy of the DataLoader instance."""
        return DataLoader.from_processed_data(
            data=self.data.copy(),
            steps=self.steps.copy() if self.steps is not None else None,
            initial_voltage=self.initial_voltage,
            start_idx=self._start_idx,
            end_idx=self._end_idx,
        )


class OCPDataLoader(DataLoader):
    """Deprecated: use DataLoader instead."""

    def __init__(self, data, **kwargs):
        warnings.warn(
            "OCPDataLoader is deprecated. Use DataLoader(data) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        # Map old flat options into transforms
        options = kwargs.pop("options", None) or {}
        merged = {**options, **kwargs}
        transforms = dict(merged.pop("transforms", None) or {})
        for key in ("sort", "remove_duplicates", "remove_extremes",
                     "filters", "interpolate"):
            if key in merged and key not in transforms:
                transforms[key] = merged.pop(key)
        if transforms:
            merged["transforms"] = transforms
        super().__init__(data, steps=None, **merged)

    @classmethod
    def from_db(cls, measurement_id, options=None, use_cache=True):
        warnings.warn(
            "OCPDataLoader.from_db is deprecated. Use DataLoader.from_db instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return DataLoader.from_db(
            measurement_id, options=options, use_cache=use_cache
        )
