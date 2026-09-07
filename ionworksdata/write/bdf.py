"""BDF (Battery Data Format) writer."""

from __future__ import annotations

import gzip
from pathlib import Path

import polars as pl

from ionworksdata.read.bdf import BDF_COLUMN_MAP, detect_bdf_format

# ionworksdata column → (machine-readable, preferred label)
_IONWORKS_TO_BDF: dict[str, tuple[str, str]] = {
    ionworks: (machine, label) for machine, label, ionworks in BDF_COLUMN_MAP
}

_REQUIRED_IONWORKS = ("Time [s]", "Voltage [V]", "Current [A]")


def bdf(
    data: pl.DataFrame,
    filename: str | Path,
    use_machine_readable_names: bool = False,
) -> None:
    """Write an ionworksdata time-series DataFrame to a BDF file.

    Columns recognised by the standard BDF mapping are renamed to their BDF
    names; other columns are passed through under their ionworksdata names so
    that a round-trip through :func:`ionworksdata.read.bdf` preserves them.
    The discharge-positive / charge-negative sign convention used by
    ionworksdata is preserved on write; callers wanting a different convention
    should flip the current column before calling this function.

    Parameters
    ----------
    data : pl.DataFrame
        Time-series DataFrame. Must contain ``Time [s]``, ``Voltage [V]``, and
        ``Current [A]``.
    filename : str | Path
        Output path. Extension selects the format: ``.csv`` / ``.bdf`` write
        plain CSV, ``.csv.gz`` / ``.bdf.gz`` write gzipped CSV, and
        ``.parquet`` / ``.bdf.parquet`` write parquet.
    use_machine_readable_names : bool, optional
        If True, write BDF machine-readable names (e.g. ``test_time_second``)
        as column headers. Default False (preferred labels such as
        ``Test Time``).

    Raises
    ------
    ValueError
        If any of ``Time [s]``, ``Voltage [V]``, or ``Current [A]`` is absent
        from ``data``, or if ``filename`` has an unsupported extension.
    """
    filename = Path(filename)
    missing = [c for c in _REQUIRED_IONWORKS if c not in data.columns]
    if missing:
        raise ValueError(
            f"Cannot write BDF file: input DataFrame is missing required "
            f"columns {missing}."
        )

    rename_map: dict[str, str] = {}
    for col in data.columns:
        if col in _IONWORKS_TO_BDF:
            machine, label = _IONWORKS_TO_BDF[col]
            rename_map[col] = machine if use_machine_readable_names else label
    renamed = data.rename(rename_map)

    fmt = detect_bdf_format(filename)
    if fmt == "csv":
        renamed.write_csv(filename)
    elif fmt == "csv_gz":
        csv_bytes = renamed.write_csv().encode("utf-8")
        with gzip.open(filename, "wb") as handle:
            handle.write(csv_bytes)
    elif fmt == "parquet":
        renamed.write_parquet(filename)
    else:
        raise AssertionError(f"Unexpected BDF format: {fmt}")
