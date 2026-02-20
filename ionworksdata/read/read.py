from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import iwutil
import pandas as pd
import polars as pl

import ionworksdata as iwdata
from ionworksdata.logger import logger
from ionworksdata.read.detect import detect_reader
from ionworks.validators import (
    MeasurementValidationError,
    validate_measurement_data,
)


def _validate_argument_order(
    filename: str | Path, reader: str | None, function_name: str
) -> None:
    """
    Validate that filename and reader are in the correct order.

    Parameters
    ----------
    filename : str | Path
        The first argument (should be filename).
    reader : str | None
        The second argument (should be reader name or None).
    function_name : str
        Name of the function for error message.

    Raises
    ------
    ValueError
        If arguments appear to be in the wrong order.
    """
    # Check if first argument looks like a reader name
    if isinstance(filename, str):
        known_readers = BaseReader.get_reader_types().keys()
        filename_lower = filename.lower()
        # Check if it matches a known reader name
        if filename_lower in known_readers:
            # Check if second argument looks like a filename
            if reader is not None:
                reader_str = str(reader)
                # Check if it looks like a file path (has extension or contains path separators)
                path_obj = Path(reader_str)
                if (
                    path_obj.suffix
                    or "/" in reader_str
                    or "\\" in reader_str
                    or Path(reader_str).exists()
                ):
                    # Customize error message for measurement_details
                    if function_name == "measurement_details":
                        raise ValueError(
                            f"Arguments appear to be in the wrong order for "
                            f"{function_name}. "
                            f"Expected: {function_name}(filename, measurement, reader, ...), "
                            f"but got: {function_name}(reader='{filename}', "
                            f"filename='{reader}', ...). "
                            f"Please use: {function_name}('{reader}', measurement, '{filename}', ...)"
                        )
                    else:
                        raise ValueError(
                            f"Arguments appear to be in the wrong order for "
                            f"{function_name}. "
                            f"Expected: {function_name}(filename, reader, ...), "
                            f"but got: {function_name}(reader='{filename}', "
                            f"filename='{reader}'). "
                            f"Please use: {function_name}('{reader}', '{filename}', ...)"
                        )


class BaseReader:
    name: str = "Unknown reader"
    default_options: dict[str, Any] = {}

    # Columns that should always be treated as numeric (Float64).
    # Used by _coerce_numeric_columns and standard_data_processing.
    ALWAYS_NUMERIC_COLUMNS = [
        "Time [s]",
        "Time [h]",
        "Current [A]",
        "Current [mA]",
        "Current [mA.cm-2]",
        "Voltage [V]",
        "Temperature [degC]",
        "Frequency [Hz]",
        "Capacity [A.h]",
        "Energy [W.h]",
        "Charge capacity [A.h]",
        "Discharge capacity [A.h]",
        "Charge energy [W.h]",
        "Discharge energy [W.h]",
    ]

    @staticmethod
    def _coerce_numeric(df: pl.DataFrame, col: str) -> pl.DataFrame:
        """
        Coerce a column to Float64, handling both string and numeric types.

        For string columns, removes thousand separators (commas) before parsing.
        For numeric columns, simply casts to Float64.

        Parameters
        ----------
        df : pl.DataFrame
            Input dataframe.
        col : str
            Column name to coerce.

        Returns
        -------
        pl.DataFrame
            Dataframe with column coerced to Float64 if it exists.
        """
        if col not in df.columns:
            return df
        dtype = df.schema[col]
        if dtype == pl.Utf8:
            # String column - remove thousand separators and parse
            return df.with_columns(
                pl.col(col).str.replace_all(",", "").cast(pl.Float64, strict=False)
            )
        elif dtype != pl.Float64:
            # Numeric column (Int, UInt, Float32) - cast to Float64
            return df.with_columns(pl.col(col).cast(pl.Float64, strict=False))
        return df

    def _coerce_numeric_columns(
        self, df: pl.DataFrame, columns: list[str] | None = None
    ) -> pl.DataFrame:
        """
        Coerce multiple columns to Float64.

        Parameters
        ----------
        df : pl.DataFrame
            Input dataframe.
        columns : list[str] | None
            List of column names to coerce. If None, uses ALWAYS_NUMERIC_COLUMNS.

        Returns
        -------
        pl.DataFrame
            Dataframe with specified columns coerced to Float64.
        """
        columns = columns if columns is not None else self.ALWAYS_NUMERIC_COLUMNS
        for col in columns:
            df = self._coerce_numeric(df, col)
        return df

    @classmethod
    def get_reader_types(cls) -> dict[str, type[BaseReader]]:
        def get_all_subclasses(klass: type) -> list[type]:
            """Recursively get all subclasses of a class."""
            subclasses = list(klass.__subclasses__())
            for subclass in list(subclasses):
                subclasses.extend(get_all_subclasses(subclass))
            return subclasses

        return {c.get_name(): c for c in get_all_subclasses(cls)}

    @classmethod
    def get_reader_object(cls, name: str) -> BaseReader:
        try:
            reader_object = cls.get_reader_types()[name.lower()]()
            return reader_object
        except KeyError as e:
            m = f"Unsupported reader type: {name}. Supported reader types: {list(cls.get_reader_types().keys())}"
            raise ValueError(m) from e

    @classmethod
    def get_name(cls) -> str:
        return cls.name.lower()

    def run(
        self,
        filename: str | Path,
        extra_column_mappings: dict[str, str] | None = None,
        options: dict[str, str] | None = None,
    ) -> pl.DataFrame:
        raise NotImplementedError

    def standard_data_processing(
        self,
        data: pl.DataFrame,
        columns_keep: list[str] | None = None,
    ) -> pl.DataFrame:
        """
        Standard data processing for all files. Skips NaNs in current and voltage,
        converts all numeric columns to float, resets "Time [s]" to start at zero,
        offsets duplicate time values, and only keeps the required columns.

        Parameters
        ----------
        data : pl.DataFrame
            The data to be processed.
        columns_keep : list[str] | None, optional
            List of columns to keep from the data. Default is None.

        Returns
        -------
        pl.DataFrame
            The processed data with standardized columns and formatting.
        """
        subset_cols = [
            c
            for c in ["Voltage [V]", "Current [A]", "Current [mA.cm-2]"]
            if c in data.columns
        ]
        if subset_cols:
            data = data.drop_nulls(subset=subset_cols)

        # Coerce known numeric columns to Float64 (handles both string and int columns)
        data = self._coerce_numeric_columns(data)

        # Cast any remaining numeric dtypes to float (for columns not in ALWAYS_NUMERIC_COLUMNS)
        NUMERIC_DTYPES = {
            pl.Int8,
            pl.Int16,
            pl.Int32,
            pl.Int64,
            pl.UInt8,
            pl.UInt16,
            pl.UInt32,
            pl.UInt64,
            pl.Float32,
            pl.Float64,
        }
        for col, dtype in data.schema.items():
            if (
                dtype in NUMERIC_DTYPES
                and col not in self.ALWAYS_NUMERIC_COLUMNS
                and col not in ("Step from cycler", "Cycle from cycler")
            ):
                data = data.with_columns(pl.col(col).cast(pl.Float64, strict=False))

        # Ensure integer columns are ints if present
        for col in ["Step from cycler", "Cycle from cycler"]:
            if col in data.columns:
                data = data.with_columns(pl.col(col).cast(pl.Int64))

        if columns_keep:
            columns_keep = [col for col in columns_keep if col in data.columns]
            data = data.select(columns_keep)

        data = iwdata.transform.reset_time(data)
        if "Time [s]" in data.columns:
            data = iwdata.transform.offset_duplicate_times(data)

        step_options: dict[str, Any] = {}
        if "Current [A]" not in data.columns:
            if "Current [mA.cm-2]" in data.columns:
                step_options.update({"current units": "density"})
            else:
                raise RuntimeError("Could not identify current units")
        data = iwdata.transform.set_positive_current_for_discharge(
            data, options=step_options
        )

        return data

    def read_start_time(
        self,
        filename: str | Path,
        extra_column_mappings: dict[str, str] | None = None,
        options: dict[str, str] | None = None,
    ) -> Any:
        raise NotImplementedError


def time_series(
    filename: str | Path,
    reader: str | None = None,
    extra_column_mappings: dict[str, str] | None = None,
    extra_constant_columns: dict[str, float] | None = None,
    options: dict[str, str] | None = None,
    save_dir: str | Path | None = None,
) -> pl.DataFrame:
    """
    Read the time series data from cycler file into a dataframe with standardized columns.

    Parameters
    ----------
    filename : str or Path
        The path to the cycler file to read.
    reader : str | None, optional
        The name of the reader to use. See subclasses of `iwdata.read.BaseReader`.
        If not provided, the reader will be automatically detected from the file.
    extra_column_mappings : dict, optional
        A dictionary of extra column mappings. The keys are the original column names and
        the values are the new column names.
    extra_constant_columns : dict, optional
        A dictionary of extra constant columns. The keys are the column names and the
        values are the constant values to fill those columns with.
    options : dict, optional
        A dictionary of options to pass to the reader. See the specific reader's
        documentation for available options.
    save_dir : str or Path, optional
        The directory to save the time series data to. If not provided, the data will
        not be saved.

    Returns
    -------
    pl.DataFrame
        The processed time series data with standardized columns:
        - "Time [s]" : Time in seconds
        - "Current [A]" : Current in amperes
        - "Voltage [V]" : Voltage in volts
        - "Temperature [degC]" : Temperature in degrees Celsius (optional)
        - "Step from cycler" : Step number from cycling data
        - "Cycle from cycler" : Cycle number from cycling data
        - "Frequency [Hz]" : Frequency in hertz (optional)
        - "Step count" : Cumulative step count
        - "Cycle count" : Cumulative cycle count (defaults to 0 if no cycle information is available)
        - "Discharge capacity [A.h]" : Discharge capacity in ampere-hours
        - "Charge capacity [A.h]" : Charge capacity in ampere-hours
        - "Discharge energy [W.h]" : Discharge energy in watt-hours
        - "Charge energy [W.h]" : Charge energy in watt-hours
        Additional columns may be present if specified in extra_column_mappings or
        extra_constant_columns.
    """
    # Validate argument order
    _validate_argument_order(filename, reader, "time_series")

    # Auto-detect reader if not provided
    if reader is None:
        reader = detect_reader(filename)

    # Read the raw data into standard format
    reader_object = BaseReader.get_reader_object(reader)
    data = reader_object.run(
        filename,
        extra_column_mappings=extra_column_mappings,
        options=options,
    )

    # Add constant columns
    if extra_constant_columns:
        for col, value in extra_constant_columns.items():
            data = data.with_columns(pl.lit(value).alias(col))

    # Add step count
    # If "Step from cycler" is provided, use it to calculate step count
    # Otherwise, create a temporary step column, calculate step count, then remove it
    if "Step from cycler" in data.columns:
        step_options = {"step column": "Step from cycler"}
        data = iwdata.transform.set_step_count(data, options=step_options)
    else:
        # Create temporary step column using current sign method
        data = iwdata.transform.set_cumulative_step_number(
            data, options={"method": "current sign"}
        )
        # Calculate step count using the temporary column
        step_options = {"step column": "Step number"}
        data = iwdata.transform.set_step_count(data, options=step_options)
        # Remove the temporary step column
        data = data.drop("Step number")

    # Add cycle count (always set, defaults to 0 if no cycle from cycler)
    data = iwdata.transform.set_cycle_count(data)

    # Add capacity and energy columns
    data = iwdata.transform.set_capacity(data, options=options)
    data = iwdata.transform.set_energy(data, options=options)

    if save_dir:
        save_path = Path(save_dir) if isinstance(save_dir, str) else save_dir
        iwutil.save.csv(data, save_path / "time_series.csv")

    return data


def time_series_and_steps(
    filename: str | Path,
    reader: str | None = None,
    extra_column_mappings: dict[str, str] | None = None,
    extra_constant_columns: dict[str, float] | None = None,
    options: dict[str, Any] | None = None,
    save_dir: str | Path | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Read the time series data from cycler file into a dataframe using :func:`ionworksdata.read.time_series`
    and then label the steps. The steps dataframe is created using :func:`ionworksdata.steps.summarize`.
    The steps output always includes a "Cycle count" column (defaults to 0 if no cycle information is available)
    and a "Cycle from cycler" column (only if provided in the input data).

    When validation is enabled, runs the same validation as the Ionworks API so that
    data which passes here will pass API validation on upload. Control via the
    ``options`` dict: ``validate`` (default True) and ``validate_strict`` (default False).

    Parameters
    ----------
    filename : str or Path
        The path to the cycler file to read.
    reader : str | None, optional
        The name of the reader to use. See subclasses of `iwdata.read.BaseReader`.
        If not provided, the reader will be automatically detected from the file.
    extra_column_mappings : dict, optional
        A dictionary of extra column mappings. The keys are the original column names and
        the values are the new column names.
    extra_constant_columns : dict, optional
        A dictionary of extra constant columns. The keys are the column names and the values
        are the constant values to assign.
    options : dict, optional
        Options for the reader and for validation. Reader-specific keys are passed through
        to the reader. Additionally:
        - ``validate``: bool, if True validate data before returning (default: True).
        - ``validate_strict``: bool, if True use strict validation (default: False).
    save_dir : str or Path, optional
        The directory to save the time series and steps data to. If not provided, the data will
        not be saved.

    Returns
    -------
    data : pl.DataFrame
        The processed time series data with standardized column names and units. See
        :func:`ionworksdata.read.time_series` for details.
    steps : pl.DataFrame
        The step summary data containing step types, cycle numbers, and other metadata.

    Raises
    ------
    MeasurementValidationError
        If validation is enabled and the data fails API-matching validation.
    """
    # Validate argument order
    _validate_argument_order(filename, reader, "time_series_and_steps")

    opts = options or {}
    reader_options = {
        k: v for k, v in opts.items() if k not in ("validate", "validate_strict")
    }
    should_validate = opts.get("validate", True)
    validate_strict = opts.get("validate_strict", False)

    data = time_series(
        filename,
        reader,
        extra_column_mappings,
        extra_constant_columns,
        reader_options,
        save_dir,
    )

    # Label the steps using "Step count" as the filter column
    if "Step count" not in data.columns:
        raise ValueError(
            "No 'Step count' column found in data. Cannot create steps dataframe. "
            "This column is automatically added by time_series()."
        )

    steps = iwdata.steps.summarize(data)
    if not isinstance(steps, pl.DataFrame):
        steps = pl.from_pandas(steps)

    if should_validate:
        try:
            validate_measurement_data(data, strict=validate_strict)
        except MeasurementValidationError as e:
            # Check if it's a current sign convention error - if so, auto-fix
            if "Current sign convention error" in str(e):
                logger.info("Detected wrong current sign convention, auto-fixing...")
                # Flip current sign
                data = data.with_columns((-pl.col("Current [A]")).alias("Current [A]"))
                # Recalculate capacity and energy with correct sign
                data = iwdata.transform.set_capacity(data, options=reader_options)
                data = iwdata.transform.set_energy(data, options=reader_options)
                # Regenerate steps with fixed data
                steps = iwdata.steps.summarize(data)
                if not isinstance(steps, pl.DataFrame):
                    steps = pl.from_pandas(steps)
                # Re-validate
                validate_measurement_data(data, strict=validate_strict)
            else:
                raise

    if save_dir:
        save_path = Path(save_dir) if isinstance(save_dir, str) else save_dir
        iwutil.save.csv(steps, save_path / "steps.csv")

    return data, steps


def keep_required_columns(
    data: pl.DataFrame,
    extra_columns: list[str] | None = None,
) -> pl.DataFrame:
    """
    Returns a new dataframe with only required columns and any extra columns specified.

    Parameters
    ----------
    data : pl.DataFrame
        The time series dataframe.
    extra_columns : list[str] | None, optional
        List of extra columns to keep. Default is None.

    Returns
    -------
    pl.DataFrame
        A new dataframe containing only the required columns:
        - "Time [s]"
        - "Current [A]"
        - "Voltage [V]"
        - "Temperature [degC]"
        - "Frequency [Hz]"
        - "Step count"
        - "Cycle count"
        - "Discharge capacity [A.h]"
        - "Charge capacity [A.h]"
        - "Discharge energy [W.h]"
        - "Charge energy [W.h]"
        And any extra columns specified in extra_columns.
    """
    extra_columns = extra_columns or []
    # Note: "Step from cycler" and "Cycle from cycler" are not included here
    # as they belong in the steps table, not the time series table.
    columns_to_keep = [
        "Time [s]",
        "Current [A]",
        "Voltage [V]",
        "Temperature [degC]",
        "Frequency [Hz]",
        "Step count",
        "Cycle count",
        "Discharge capacity [A.h]",
        "Charge capacity [A.h]",
        "Discharge energy [W.h]",
        "Charge energy [W.h]",
    ]
    columns_to_keep.extend([col for col in extra_columns if col not in columns_to_keep])

    return data.select([col for col in columns_to_keep if col in data.columns])


def start_time(
    filename: str | Path,
    reader: str | None = None,
    extra_column_mappings: dict[str, str] | None = None,
    options: dict[str, str] | None = None,
) -> Any:
    """
    Read the start time from the cycler file.

    Parameters
    ----------
    filename : str or Path
        The path to the cycler file to read.
    reader : str | None, optional
        The name of the reader to use. See subclasses of `iwdata.read.BaseReader`.
        If not provided, the reader will be automatically detected from the file.
    extra_column_mappings : dict[str, str] | None, optional
        Dictionary of additional column mappings to use when reading the file.
        The keys are the original column names and the values are the new column
        names. Default is None.
    options : dict[str, str] | None, optional
        A dictionary of options to pass to the reader. See the reader's documentation
        for the available options. Default is None.

    Returns
    -------
    datetime
        The start time of the cycler file as a timezone-aware datetime object.
        Returns None if the reader cannot determine the start time.
    """
    # Validate argument order
    _validate_argument_order(filename, reader, "start_time")

    # Auto-detect reader if not provided
    if reader is None:
        from ionworksdata.read.detect import detect_reader

        reader = detect_reader(filename)

    reader_object = BaseReader.get_reader_object(reader)

    if reader_object.get_name() == "csv":
        warnings.warn(
            "CSV reader does not support reading start time from file",
            stacklevel=2,
        )
        return None
    return reader_object.read_start_time(filename, extra_column_mappings, options)


def _read_ocp_measurement(
    filename: str | Path,
    measurement: dict[str, str],
    extra_column_mappings: dict[str, str] | None = None,
    extra_constant_columns: dict[str, float] | None = None,
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Read OCP (open-circuit potential) data and return a measurement dict.

    This is a simplified path that only requires ``Voltage [V]`` (and
    optionally ``Capacity [A.h]``) in the source data.  Synthetic ``Step
    count`` and ``Cycle count`` columns are added automatically.

    Parameters
    ----------
    filename : str | Path
        Path to a CSV file containing the OCP data.
    measurement : dict
        Measurement metadata dictionary (updated in-place).
    extra_column_mappings : dict, optional
        Maps raw column names to standard names, e.g.
        ``{"SOC": "Capacity [A.h]", "OCV": "Voltage [V]"}``.
    extra_constant_columns : dict, optional
        Constant-valued columns to add to the time series.
    options : dict, optional
        ``validate`` (bool, default True) and ``validate_strict`` (bool,
        default False) control validation behaviour.

    Returns
    -------
    dict[str, Any]
        ``{"time_series": pl.DataFrame, "steps": pl.DataFrame,
        "measurement": dict}``
    """
    data = pl.read_csv(filename)

    if extra_column_mappings:
        rename_map = {
            old: new
            for old, new in extra_column_mappings.items()
            if old in data.columns
        }
        if rename_map:
            data = data.rename(rename_map)

    if "Voltage [V]" not in data.columns:
        raise ValueError(
            "OCP data must contain a 'Voltage [V]' column (after applying "
            f"extra_column_mappings). Available columns: {data.columns}"
        )

    if extra_constant_columns:
        for col, value in extra_constant_columns.items():
            data = data.with_columns(pl.lit(value).alias(col))

    data = data.with_columns(
        [
            pl.lit(0).cast(pl.Int64).alias("Step count"),
            pl.lit(0).cast(pl.Int64).alias("Cycle count"),
        ]
    )

    cols_to_keep = ["Voltage [V]", "Step count", "Cycle count"]
    if "Capacity [A.h]" in data.columns:
        cols_to_keep.append("Capacity [A.h]")
    if extra_column_mappings:
        for mapped in extra_column_mappings.values():
            if mapped in data.columns and mapped not in cols_to_keep:
                cols_to_keep.append(mapped)
    if extra_constant_columns:
        for col in extra_constant_columns:
            if col not in cols_to_keep:
                cols_to_keep.append(col)

    data = data.select([c for c in cols_to_keep if c in data.columns])

    # Coerce numeric columns
    for col in data.columns:
        if data.schema[col] != pl.Float64 and col not in ("Step count", "Cycle count"):
            data = data.with_columns(pl.col(col).cast(pl.Float64, strict=False))

    steps = iwdata.steps.ocp_steps(data)

    opts = options or {}
    should_validate = opts.get("validate", True)
    validate_strict = opts.get("validate_strict", False)
    if should_validate:
        validate_measurement_data(data, strict=validate_strict, data_type="ocp")

    measurement["data_type"] = "ocp"
    measurement["step_labels_validated"] = False

    return {
        "measurement": measurement,
        "steps": steps,
        "time_series": data,
    }


def measurement_details(
    filename: str | Path,
    measurement: dict[str, str],
    reader: str | None = None,
    extra_column_mappings: dict[str, str] | None = None,
    extra_constant_columns: dict[str, float] | None = None,
    options: dict[str, Any] | None = None,
    labels: list[dict[str, Any]] | None = None,
    keep_only_required_columns: bool = True,
    data_type: str | None = None,
) -> dict[str, Any]:
    """
    Read the time series data from cycler file into a dataframe using :func:`ionworksdata.read.time_series_and_steps`
    and then keep only the required columns in the time series using :func:`ionworksdata.read.keep_required_columns`.
    The cycler name and test start time are added to the measurement dictionary. Then return a dictionary with the time
    series data, the steps data, and the measurement dictionary.

    Parameters
    ----------
    filename : str | Path
        The path to the cycler file to read.
    measurement : dict[str, str]
        A dictionary containing the measurement information (e.g. protocol name,
        test name, etc.). This is updated inplace to include the cycler name and test start time.
    reader : str | None, optional
        The name of the reader to use. See subclasses of `iwdata.read.BaseReader`.
        If not provided, the reader will be automatically detected from the file.
    extra_column_mappings : dict[str, str] | None, optional
        A dictionary of extra column mappings. The keys are the original column names and
        the values are the new column names. Default is None.
    extra_constant_columns : dict[str, float] | None, optional
        A dictionary of extra constant columns. The keys are the column names and the values
        are the constant values. Default is None.
    options : dict[str, str] | None, optional
        A dictionary of options to pass to the reader. See the reader's documentation
        for the available options. Default is None.
    labels : list[dict[str, Any]] | None, optional
        A list of dictionaries containing the labels to add to the steps table.
        The keys are the label names and the values are the label options. If not provided,
        the default labels are added, which are cycling, pulse (charge and discharge), and EIS.
        If None, then the options dictionary must contain a "cell_metadata" key, which has a
        "Nominal cell capacity [A.h]" key, which is used to add the cycling and pulse labels.
        Default is None.
    keep_only_required_columns : bool, optional
        If True, only the required columns are kept in the time series. Default is True.
        See :func:`ionworksdata.read.keep_required_columns` for the required columns.
    data_type : str | None, optional
        The type of data in the file. Use ``"ocp"`` for open-circuit potential
        data, which uses a simplified processing path that only requires
        ``Voltage [V]`` (and optionally ``Capacity [A.h]``). Default is
        ``None`` (standard cycler data).

    Returns
    -------
    dict[str, Any]
        A dictionary containing:
        - "time_series": pl.DataFrame with the time series data
        - "steps": pl.DataFrame with the steps data
        - "measurement": dict with the updated measurement information
    """
    if data_type == "ocp":
        return _read_ocp_measurement(
            filename,
            measurement,
            extra_column_mappings=extra_column_mappings,
            extra_constant_columns=extra_constant_columns,
            options=options,
        )

    # Validate argument order (skip measurement parameter)
    _validate_argument_order(filename, reader, "measurement_details")

    # Auto-detect reader if not provided
    if reader is None:
        reader = detect_reader(filename)

    # Additional check: if reader looks like a file path, it's wrong
    if reader is not None:
        reader_str = str(reader)
        if (
            Path(reader_str).suffix
            or "/" in reader_str
            or "\\" in reader_str
            or Path(reader_str).exists()
        ):
            try:
                BaseReader.get_reader_object(reader)
            except ValueError:
                # Reader is a file path, check if measurement is actually the reader name
                if isinstance(measurement, str):
                    known_readers = BaseReader.get_reader_types().keys()
                    if measurement.lower() in known_readers:
                        raise ValueError(
                            f"Arguments appear to be in the wrong order for "
                            f"measurement_details. "
                            f"Expected: measurement_details(filename, measurement, reader, ...), "
                            f"but got: measurement_details(measurement='{measurement}', "
                            f"reader='{reader}', ...). "
                            f"Please use: measurement_details('{reader}', measurement, '{measurement}', ...)"
                        ) from None

    # Emit CSV start-time warning when applicable (compat with pandas tests)
    if BaseReader.get_reader_object(reader).get_name() == "csv":
        import warnings

        warnings.warn(
            "CSV reader does not support reading start time from file",
            stacklevel=2,
        )
    # Get the time series and steps
    data, steps = time_series_and_steps(
        filename,
        reader,
        extra_column_mappings,
        extra_constant_columns,
        options,
    )
    # Keep only the required columns in the time series
    if keep_only_required_columns:
        extra_columns: list[str] = []
        if extra_column_mappings:
            extra_columns.extend(extra_column_mappings.values())
        if extra_constant_columns:
            extra_columns.extend(extra_constant_columns.keys())
        data = keep_required_columns(data, extra_columns=extra_columns)

    # Add labels to the steps table
    options = options or {}
    cell_metadata_raw: dict | str | None = options.get("cell_metadata", {})
    if (
        not isinstance(cell_metadata_raw, dict)
        or "Nominal cell capacity [A.h]" not in cell_metadata_raw
    ):
        logger.warning(
            "No 'Nominal cell capacity [A.h]' found in cell_metadata dictionary. "
            "Unable to add labels to the steps table.",
        )
        step_labels_validated = False
    else:
        nominal_capacity = cell_metadata_raw["Nominal cell capacity [A.h]"]
        cell_metadata: dict = {"Nominal cell capacity [A.h]": nominal_capacity}
        default_labels = [
            {"Cycling": {"cell_metadata": cell_metadata}},
            {
                "Pulse": {
                    "cell_metadata": cell_metadata,
                    "current direction": "discharge",
                }
            },
            {"Pulse": {"cell_metadata": cell_metadata, "current direction": "charge"}},
            {"EIS": {}},
        ]
        labels_to_apply: list[dict[str, Any]] = labels or default_labels
        for label in labels_to_apply:
            for label_name, label_options in label.items():
                # Label methods require pandas (see comments in label files for details)
                # Convert to pandas, apply labeling, then convert back to Polars
                if isinstance(steps, pl.DataFrame):
                    steps_pd = steps.to_pandas()
                else:
                    steps_pd = steps
                label_name_lower = label_name.lower()
                if label_name_lower == "cycling":
                    steps_pd = iwdata.steps.label_cycling(
                        steps_pd, options=label_options
                    )
                elif label_name_lower == "pulse":
                    steps_pd = iwdata.steps.label_pulse(steps_pd, options=label_options)
                elif label_name_lower == "eis":
                    steps_pd = iwdata.steps.label_eis(steps_pd, options=label_options)
                else:
                    raise ValueError(f"Unknown label type: {label_name}")
                steps = pl.from_pandas(steps_pd)
        # Check that the steps labels are valid
        validations: list[bool] = []
        # Convert to pandas for validation (validate_steps requires pandas)
        steps_pd_for_validate = (
            steps.to_pandas() if isinstance(steps, pl.DataFrame) else steps
        )
        for label_name in steps_pd_for_validate["Label"].unique():
            if pd.isna(label_name):
                continue
            this_valid = iwdata.steps.validate(steps_pd_for_validate, label_name)
            validations.append(this_valid)
        step_labels_validated = all(validations) if validations else False

    # Populate the measurement dictionary
    measurement["step_labels_validated"] = step_labels_validated
    # Add the cycler name
    measurement["cycler"] = reader
    # Add the start time
    test_start_time = start_time(filename, reader, extra_column_mappings, options)
    if test_start_time is not None:
        measurement["start_time"] = test_start_time.isoformat()

    measurement_details_result = {
        "measurement": measurement,
        "steps": steps,
        "time_series": data,
    }
    return measurement_details_result
