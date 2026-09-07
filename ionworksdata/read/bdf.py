"""BDF (Battery Data Format) reader.

Reads files conforming to the Battery Data Alliance BDF spec
(https://battery-data-alliance.github.io/battery-data-format/) and returns
them in the standard ionworksdata time-series format.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal
import warnings

import polars as pl

from ionworksdata.read.read import BaseReader

BDF_FORMAT = Literal["csv", "csv_gz", "parquet"]

# Machine-readable name, preferred label, ionworksdata column.
BDF_COLUMN_MAP: list[tuple[str, str, str]] = [
    ("test_time_second", "Test Time", "Time [s]"),
    ("voltage_volt", "Voltage", "Voltage [V]"),
    ("current_ampere", "Current", "Current [A]"),
    ("cycle_count", "Cycle Count", "Cycle from cycler"),
    ("step_count", "Step Count", "Step from cycler"),
    ("ambient_temperature_celsius", "Ambient Temperature", "Temperature [degC]"),
    (
        "charging_capacity_ampere_hour",
        "Charging Capacity",
        "Charge capacity [A.h]",
    ),
    (
        "discharging_capacity_ampere_hour",
        "Discharging Capacity",
        "Discharge capacity [A.h]",
    ),
    ("charging_energy_watt_hour", "Charging Energy", "Charge energy [W.h]"),
    (
        "discharging_energy_watt_hour",
        "Discharging Energy",
        "Discharge energy [W.h]",
    ),
]

# Canonical definitions live in read.detect (imported before this module) so
# that detection and reading stay in sync automatically.
from ionworksdata.read.detect import (  # noqa: E402
    BDF_EXTENSIONS,
    BDF_REQUIRED_LABELS,
    BDF_REQUIRED_MACHINE,
)

__all__ = [
    "BDF",
    "BDF_COLUMN_MAP",
    "BDF_EXTENSIONS",
    "BDF_REQUIRED_LABELS",
    "BDF_REQUIRED_MACHINE",
    "bdf",
    "detect_bdf_format",
]


def detect_bdf_format(filename: Path) -> BDF_FORMAT:
    """Return the on-disk format for a BDF file based on its extension.

    Uses ``Path.suffixes`` so compound extensions (``.bdf.gz``, ``.bdf.parquet``,
    ``.csv.gz``) are recognised correctly. ``Path.suffix`` alone returns only
    ``.gz``, which would be indistinguishable from an arbitrary gzip file.

    Parameters
    ----------
    filename : Path
        Path whose extension determines the format.

    Returns
    -------
    {"csv", "csv_gz", "parquet"}
        The detected format tag.

    Raises
    ------
    ValueError
        If the extension is not a supported BDF variant.
    """
    suffixes = [s.lower() for s in filename.suffixes]
    tail1 = suffixes[-1:] if suffixes else []
    tail2 = suffixes[-2:] if len(suffixes) >= 2 else []

    if tail2 == [".bdf", ".parquet"] or tail1 == [".parquet"]:
        return "parquet"
    if tail2 in ([".csv", ".gz"], [".bdf", ".gz"]):
        return "csv_gz"
    if tail1 in ([".csv"], [".bdf"]):
        return "csv"

    raise ValueError(
        f"Unsupported BDF extension for file: {filename}. "
        "Expected one of: .csv, .bdf, .csv.gz, .bdf.gz, .parquet, .bdf.parquet."
    )


def _load_bdf_dataframe(filename: Path) -> pl.DataFrame:
    """Load raw BDF file contents into a polars DataFrame based on extension."""
    fmt = detect_bdf_format(filename)
    if fmt == "parquet":
        return pl.read_parquet(filename)
    # Polars read_csv handles gzip transparently via magic-byte detection.
    return pl.read_csv(filename)


class BDF(BaseReader):
    """Reader for Battery Data Alliance BDF files.

    Supports CSV (``.csv`` / ``.bdf``), gzipped CSV (``.csv.gz`` / ``.bdf.gz``),
    and parquet (``.parquet`` / ``.bdf.parquet``). Recognises both the BDF
    machine-readable names (e.g. ``test_time_second``) and preferred labels
    (e.g. ``Test Time``) in the file header.

    The reader normalises current to ionworksdata's convention (discharge
    positive, charge negative) as part of standard processing. BDF does not
    mandate a sign convention, so third-party files that follow the opposite
    IEC charge-positive convention are flipped automatically.
    """

    name: str = "BDF"
    default_options: dict[str, Any] = {}

    def run(
        self,
        filename: str | Path,
        extra_column_mappings: dict[str, str] | None = None,
        options: dict[str, str] | None = None,
    ) -> pl.DataFrame:
        """Read a BDF file and return an ionworksdata time-series DataFrame.

        Parameters
        ----------
        filename : str | Path
            Path to the BDF file. Supported extensions are ``.csv``, ``.bdf``,
            ``.csv.gz``, ``.bdf.gz``, ``.parquet``, and ``.bdf.parquet``.
        extra_column_mappings : dict[str, str] | None, optional
            Extra header-to-ionworksdata-column mappings applied before the
            standard BDF mapping. Useful for non-standard BDF extensions.
        options : dict[str, str] | None, optional
            Unused; accepted for API compatibility with other readers.

        Returns
        -------
        pl.DataFrame
            Processed DataFrame with standard ionworksdata columns plus any
            unmapped BDF columns kept under their original names.

        Raises
        ------
        ValueError
            If one of the three BDF-required columns is missing from the file.
        """
        filename = Path(filename)
        extra_column_mappings = extra_column_mappings or {}

        data = _load_bdf_dataframe(filename)
        if extra_column_mappings:
            data = data.rename(
                {k: v for k, v in extra_column_mappings.items() if k in data.columns}
            )

        # Build the rename map from BDF headers → ionworksdata names.
        rename_map: dict[str, str] = {}
        for machine, label, ionworks in BDF_COLUMN_MAP:
            if machine in data.columns:
                rename_map[machine] = ionworks
            elif label in data.columns:
                rename_map[label] = ionworks
        if rename_map:
            data = data.rename(rename_map)

        required = {"Time [s]", "Voltage [V]", "Current [A]"}
        if not required.issubset(data.columns):
            missing = required - set(data.columns)
            raise ValueError(
                f"BDF file {filename} is missing required columns: "
                f"{sorted(missing)}. Expected BDF names "
                f"{list(BDF_REQUIRED_MACHINE)} or preferred labels "
                f"{list(BDF_REQUIRED_LABELS)}."
            )

        # Unmapped BDF / user columns must survive standard_data_processing;
        # append them to columns_keep explicitly.
        mapped_ionworks = {ionworks for _, _, ionworks in BDF_COLUMN_MAP}
        mapped_present = [c for c in data.columns if c in mapped_ionworks]
        passthrough = [c for c in data.columns if c not in mapped_ionworks]
        columns_keep = mapped_present + passthrough

        return self.standard_data_processing(data, columns_keep=columns_keep)

    def read_start_time(
        self,
        filename: str | Path,
        extra_column_mappings: dict[str, str] | None = None,
        options: dict[str, str] | None = None,
    ) -> None:
        """BDF has no standardised in-file start time; returns ``None``."""
        warnings.warn(
            "BDF reader does not support reading start time from file",
            stacklevel=2,
        )
        return None


def bdf(
    filename: str | Path,
    extra_column_mappings: dict[str, str] | None = None,
    options: dict[str, str] | None = None,
) -> pl.DataFrame:
    """Convenience wrapper around :meth:`BDF.run`."""
    return BDF().run(
        filename, extra_column_mappings=extra_column_mappings, options=options
    )
