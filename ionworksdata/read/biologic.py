from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re
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
            "Temperature/°C": "Temperature [degC]",
            "Temperature/Â°C": "Temperature [degC]",
            "Temperature/degC": "Temperature [degC]",
            "Ns": "Step from cycler",
            "Cycle number": "Cycle from cycler",
            "cycle number": "Cycle from cycler",
            "freq/Hz": "Frequency [Hz]",
            "Re(Z)/Ohm": "Z_Re [Ohm]",
            "-Im(Z)/Ohm": "Z_Im [Ohm]",
            "|Z|/Ohm": "Z_Mod [Ohm]",
            "Phase(Z)/deg": "Z_Phase [deg]",
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
            - "Temperature/°C" -> "Temperature [degC]"
            - "Ns" -> "Step from cycler"
            - "Cycle number" -> "Cycle from cycler"
            - "freq/Hz" -> "Frequency [Hz]"
            - "Re(Z)/Ohm" -> "Z_Re [Ohm]"
            - "-Im(Z)/Ohm" -> "Z_Im [Ohm]"
            - "|Z|/Ohm" -> "Z_Mod [Ohm]"
            - "Phase(Z)/deg" -> "Z_Phase [deg]"

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

        # Read headers first to determine which columns exist
        # This is necessary because polars 1.x mishandles schema_overrides
        # for non-existent columns
        header_df = pl.read_csv(
            filename,
            encoding=options["file_encoding"],
            separator=sep,
            skip_rows=skiprows,
            n_rows=0,
            infer_schema_length=0,
            truncate_ragged_lines=True,
        )

        # BioLogic data is entirely numeric, so force all columns to Float64 to
        # avoid type inference issues where initial integer-like values (e.g., "0")
        # cause a column to be inferred as Int64, then fail on later float values.
        schema_overrides = {col: pl.Float64 for col in header_df.columns}
        schema_overrides.pop("Date", None)

        # Load data and rename columns
        data = pl.read_csv(
            filename,
            encoding=options["file_encoding"],
            separator=sep,
            skip_rows=skiprows,
            schema_overrides=schema_overrides,
            truncate_ragged_lines=True,
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

        # BioLogic reports "-Im(Z)/Ohm" which is the negated imaginary part.
        # Our convention is that Z_Im [Ohm] stores the raw imaginary part
        # (negative for capacitive behavior), so we negate the values.
        if "Z_Im [Ohm]" in data.columns:
            data = data.with_columns((-pl.col("Z_Im [Ohm]")).alias("Z_Im [Ohm]"))

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

        header_df = pl.read_csv(
            filename,
            encoding=options["file_encoding"],
            separator=sep,
            skip_rows=skiprows,
            n_rows=0,
            infer_schema_length=0,
            truncate_ragged_lines=True,
        )
        schema_overrides = {col: pl.Float64 for col in header_df.columns}
        schema_overrides.pop("Date", None)

        data = pl.read_csv(
            filename,
            encoding=options["file_encoding"],
            separator=sep,
            skip_rows=skiprows,
            schema_overrides=schema_overrides,
            truncate_ragged_lines=True,
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


class BiologicCSV(BaseReader):
    """Reader for BioLogic plain CSV exports.

    BioLogic users often export cycling data to plain CSV instead of the native
    .mpt/.mpr format. These CSVs have different column names (e.g. ``Ecell_V``,
    ``I_mA``, ``cycleNumber``) that are not recognised by the generic CSV reader
    or the standard BioLogic reader.
    """

    name: str = "Biologic CSV"
    default_options: dict[str, Any] = {
        "cell_metadata": {},
    }

    #: Map raw BioLogic CSV columns to ionworks standard names.  Values that
    #: require unit conversion are handled separately in ``run()``.
    COLUMN_MAP: dict[str, str] = {
        "time_s": "Time [s]",
        "Ecell_V": "Voltage [V]",
        "Ewe_V": "Voltage [V]",
        "I_mA": "Current [mA]",
        "EnergyCharge_W_h": "Charge energy [W.h]",
        "QCharge_mA_h": "Charge capacity [mA.h]",
        "EnergyDischarge_W_h": "Discharge energy [W.h]",
        "QDischarge_mA_h": "Discharge capacity [mA.h]",
        "Temperature__C": "Temperature [degC]",
        "cycleNumber": "Cycle from cycler",
        "Ns": "Step from cycler",
    }

    #: Columns whose values are in milli-units and need ÷ 1000.
    MILLI_COLUMNS: dict[str, str] = {
        "Current [mA]": "Current [A]",
        "Charge capacity [mA.h]": "Charge capacity [A.h]",
        "Discharge capacity [mA.h]": "Discharge capacity [A.h]",
    }

    @classmethod
    def sniff(cls, first_line: str) -> bool:
        """Return True if *first_line* looks like a BioLogic plain CSV header."""
        return "Ecell_V" in first_line or "Ewe_V" in first_line

    def run(
        self,
        filename: str | Path,
        extra_column_mappings: dict[str, str] | None = None,
        options: dict[str, str] | None = None,
    ) -> pl.DataFrame:
        """Read a BioLogic plain CSV export and return standardised data.

        Parameters
        ----------
        filename : str | Path
            Path to the CSV file.
        extra_column_mappings : dict[str, str] | None, optional
            Additional column mappings on top of the built-in map.
        options : dict[str, str] | None, optional
            Reader options (currently only ``cell_metadata``).

        Returns
        -------
        pl.DataFrame
            Processed data with standard ionworks column names and SI units.
        """
        options = iwutil.check_and_combine_options(self.default_options, options)
        data = pl.read_csv(filename, infer_schema_length=10000)

        # Build the full renaming dict
        column_renamings = dict(self.COLUMN_MAP)
        column_renamings.update(extra_column_mappings or {})
        iwdata.util.check_for_duplicates(column_renamings, data)

        # Deduplicate renames: if multiple source columns map to the same
        # target (e.g. Ecell_V and Ewe_V both → Voltage [V]), keep only the
        # first one present to avoid a Polars DuplicateError.
        seen_targets: set[str] = set()
        existing_renames: dict[str, str] = {}
        for k, v in column_renamings.items():
            if k in data.columns and v not in seen_targets:
                existing_renames[k] = v
                seen_targets.add(v)
        if existing_renames:
            data = data.rename(existing_renames)

        # Batch-convert milli-unit columns to SI in a single pass
        milli_exprs = [
            (pl.col(milli_col) / 1000.0).alias(si_col)
            for milli_col, si_col in self.MILLI_COLUMNS.items()
            if milli_col in data.columns
        ]
        milli_to_drop = [mc for mc in self.MILLI_COLUMNS if mc in data.columns]
        if milli_exprs:
            data = data.with_columns(milli_exprs).drop(milli_to_drop)

        # Only include SI columns that were actually produced
        converted_si = {self.MILLI_COLUMNS[mc] for mc in milli_to_drop}
        columns_keep = list(
            (set(column_renamings.values()) - set(self.MILLI_COLUMNS.keys()))
            | converted_si
        )
        return self.standard_data_processing(data, columns_keep=columns_keep)

    def read_start_time(
        self,
        filename: str | Path,
        extra_column_mappings: dict[str, str] | None = None,
        options: dict[str, str] | None = None,
    ) -> None:
        """BioLogic plain CSV exports do not contain a start-time header."""
        return None


def biologic_csv(
    filename: str | Path,
    extra_column_mappings: dict[str, str] | None = None,
    options: dict[str, str] | None = None,
) -> pl.DataFrame:
    """Convenience function for :class:`BiologicCSV`."""
    return BiologicCSV().run(
        filename, extra_column_mappings=extra_column_mappings, options=options
    )


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
