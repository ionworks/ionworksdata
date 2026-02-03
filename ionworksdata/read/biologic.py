from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any

import iwutil
import polars as pl
import pytz

import ionworksdata as iwdata

from .read import BaseReader


class Biologic(BaseReader):
    """
    Reader for Biologic files (.mpt and .txt formats).

    The file format (separator, skiprows) is auto-detected based on extension:
    - .mpt files use tab separator
    - .txt files auto-detect separator (tab or comma)
    """

    name: str = "Biologic"
    default_options: dict[str, Any] = {
        "file_encoding": "ISO-8859-1",
        "timezone": "UTC",
        "cell_metadata": {},
    }

    @staticmethod
    def _get_file_args(
        filename: str | Path, options: dict[str, str] | None = None
    ) -> tuple[int, str]:
        """
        Get file arguments for reading a Biologic file.

        Parameters
        ----------
        filename : str | Path
            Path to the Biologic file.
        options : dict[str, str] | None
            Options dict with file_encoding key.

        Returns
        -------
        tuple[int, str]
            Tuple of (skiprows, sep).
        """
        encoding = options["file_encoding"]
        ext = Path(filename).suffix.lower()

        with open(filename, encoding=encoding) as f:
            lines = f.readlines()

        # Determine separator based on file type
        if ext == ".mpt":
            sep = "\t"
        else:
            # Auto-detect for other file types
            sep = "\t" if any("\t" in line for line in lines[:20]) else ","

        # Determine skiprows
        # Try to find line with "Nb header lines : int" and extract the int
        header_line = next((line for line in lines if "Nb header lines" in line), None)
        if header_line is not None:
            match = re.search(r"Nb header lines\s*:\s*(\d+)", header_line)
            if match:
                return int(match.group(1)) - 1, sep
            # If header_line found but regex doesn't match, fall through
        # Fallback to looking for the line that contains "mode"
        for i, row in enumerate(lines):
            if "mode" in row or "freq/Hz" in row:
                return i, sep
        # If neither method worked, raise an error
        raise ValueError("Could not find header row in Biologic file")

    @staticmethod
    def _get_column_renamings() -> dict[str, str]:
        """
        Get standard column renaming mappings for Biologic files.

        Returns
        -------
        dict[str, str]
            Dictionary mapping original column names to standardized names.
        """
        return {
            "Ecell/V": "Voltage [V]",
            "Ewe/V": "Voltage [V]",
            "<Ewe>/V": "Voltage [V]",
            "I/mA": "Current [mA]",
            "<I>/mA": "Current [mA]",
            "time/s": "Time [s]",
            "Ns": "Step from cycler",
            "Cycle number": "Cycle from cycler",
            "cycle number": "Cycle from cycler",
        }

    def run(
        self,
        filename: str | Path,
        extra_column_mappings: dict[str, str] | None = None,
        options: dict[str, str] | None = None,
    ) -> pl.DataFrame:
        """
        Read and process data from a BioLogic file.

        The following column mappings are applied by default:

            - "Ecell/V" -> "Voltage [V]"
            - "Ewe/V" -> "Voltage [V]"
            - "I/mA" -> "Current [mA]"
            - "<I>/mA" -> "Current [mA]"
            - "time/s" -> "Time [s]"
            - "Ns" -> "Step from cycler"
            - "Cycle number" -> "Cycle from cycler"

        Additional column mappings can be provided via extra_column_mappings.

        Parameters
        ----------
        filename : str
            Path to the BioLogic file to be read (.mpt or .txt).
        extra_column_mappings : dict of str to str, optional
            Dictionary of additional column mappings. Keys are original column
            names, values are the new column names.
        options : dict of str to str, optional
            Dictionary of options for reading the BioLogic file. Options are:

            - file_encoding: str, optional
                Encoding format for the file. Default is "ISO-8859-1".
            - timezone: str, optional
                Timezone for timestamps. Default is "UTC".

        Returns
        -------
        pl.DataFrame
            Processed data with standardized column names and units.
        """
        options = iwutil.check_and_combine_options(self.default_options, options)
        skiprows, sep = self._get_file_args(filename, options)

        # Force numeric columns to be read as Float64 to avoid type inference issues
        # where initial integer-like values (e.g., "0") cause the column to be read
        # as Int64, truncating subsequent decimal values
        schema_overrides = {
            "Ecell/V": pl.Float64,
            "Ewe/V": pl.Float64,
            "<Ewe>/V": pl.Float64,
            "I/mA": pl.Float64,
            "<I>/mA": pl.Float64,
            "time/s": pl.Float64,
        }

        # Load data and rename columns
        data = pl.read_csv(
            filename,
            encoding=options["file_encoding"],
            separator=sep,
            skip_rows=skiprows,
            schema_overrides=schema_overrides,
        )

        column_renamings = self._get_column_renamings()
        column_renamings.update(extra_column_mappings or {})
        iwdata.util.check_for_duplicates(column_renamings, data)

        # Filter renamings to only include columns that exist
        existing_renames = {
            k: v for k, v in column_renamings.items() if k in data.columns
        }
        if existing_renames:
            data = data.rename(existing_renames)

        # Convert current to amps
        if "Current [mA]" in data.columns:
            data = data.with_columns(
                (pl.col("Current [mA]") / 1000.0).alias("Current [A]")
            )
            data = data.drop("Current [mA]")

        # Keep only the columns we care about
        columns_keep = list(
            set(column_renamings.values()) - {"Current [mA]"} | {"Current [A]"}
        )
        return self.standard_data_processing(data, columns_keep=columns_keep)

    def read_start_time(
        self,
        filename: str | Path,
        extra_column_mappings: dict[str, str] | None = None,
        options: dict[str, str] | None = None,
    ) -> datetime | None:
        """
        Read the start time from a BioLogic file.

        Parameters
        ----------
        filename : str
            Path to the BioLogic file to be read (.mpt or .txt).
        options : dict of str to str, optional
            Dictionary of options for reading the BioLogic file.

        Returns
        -------
        datetime | None
            The start time of the BioLogic file, or None if not found.
        """
        options = iwutil.check_and_combine_options(self.default_options, options)
        skiprows, sep = self._get_file_args(filename, options)

        # Try to load the date column
        data = pl.read_csv(
            filename,
            encoding=options["file_encoding"],
            separator=sep,
            skip_rows=skiprows,
        )
        try:
            start_datetime = datetime.strptime(data["Date"][0], "%Y-%m-%d %H:%M:%S")
            timezone = options.get("timezone", "UTC")
            if isinstance(timezone, str):
                timezone = pytz.timezone(timezone)
            else:
                raise ValueError(f"Invalid timezone: {timezone}")
            start_datetime = start_datetime.replace(tzinfo=timezone)
            return iwdata.util.check_and_convert_datetime(start_datetime)
        except (KeyError, pl.exceptions.ColumnNotFoundError):
            return None


class BiologicMPT(Biologic):
    """
    Reader for Biologic MPT files.

    This is an alias for the Biologic reader - both handle .mpt and .txt formats
    automatically based on file extension.
    """

    name: str = "Biologic MPT"


def biologic(
    filename: str | Path,
    extra_column_mappings: dict[str, str] | None = None,
    options: dict[str, str] | None = None,
) -> pl.DataFrame:
    return Biologic().run(
        filename, extra_column_mappings=extra_column_mappings, options=options
    )


def biologic_mpt(
    filename: str | Path,
    extra_column_mappings: dict[str, str] | None = None,
    options: dict[str, str] | None = None,
) -> pl.DataFrame:
    return BiologicMPT().run(
        filename, extra_column_mappings=extra_column_mappings, options=options
    )
