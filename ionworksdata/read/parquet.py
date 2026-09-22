from __future__ import annotations

from pathlib import Path
from typing import Any
import warnings

import iwutil
import polars as pl

from .csv import (
    build_columns_keep,
    detect_canonical,
    synthesize_current_a_from_ma,
)
from .read import BaseReader


class Parquet(BaseReader):
    name: str = "parquet"
    default_options: dict[str, Any] = {
        "cell_metadata": {},
    }

    def run(
        self,
        filename: str | Path,
        extra_column_mappings: dict[str, str] | None = None,
        options: dict[str, str] | None = None,
    ) -> pl.DataFrame:
        """Read a parquet cycler file and return a Polars DataFrame in the
        canonical column convention.

        Mirrors the CSV reader's column-detection strategy but skips the
        text-parsing concerns: parquet is strongly-typed and has unambiguous
        column names, so no quote/separator/encoding logic is needed.

        Parameters
        ----------
        filename : str | Path
            Path to the parquet file to read.
        extra_column_mappings : dict[str, str] | None, optional
            Mapping from the file's column names to canonical names (e.g.
            ``{"current (mA)": "Current [mA]"}``). Applied before alias
            detection so the caller's renames take precedence. Default is
            None.
        options : dict[str, str] | None, optional
            Reader options. Currently only ``cell_metadata`` is recognized.

        Returns
        -------
        pl.DataFrame
            Processed data with standardized column names and units.
        """
        options = iwutil.check_and_combine_options(self.default_options, options or {})
        extra_column_mappings = extra_column_mappings or {}

        data = pl.read_parquet(filename)
        # Caller-supplied renames first — polars raises on missing keys, so
        # filter to columns actually present before calling ``rename``.
        rename = {k: v for k, v in extra_column_mappings.items() if k in data.columns}
        if rename:
            data = data.rename(rename)
        data = detect_canonical(data, set(extra_column_mappings.values()))
        data = synthesize_current_a_from_ma(data)
        columns_keep = build_columns_keep(data.columns, extra_column_mappings)
        return self.standard_data_processing(data, columns_keep=columns_keep)

    def read_start_time(
        self,
        filename: str | Path,
        extra_column_mappings: dict[str, str] | None = None,
        options: dict[str, str] | None = None,
    ) -> Any:
        """Return the first timestamp value if a timestamp column is present;
        otherwise emit a warning and return None.
        """
        lf = pl.scan_parquet(filename)
        schema_cols = lf.collect_schema().names()
        for cand in ("timestamp", "DateTime", "Date Time", "Date"):
            if cand in schema_cols:
                return lf.select(cand).head(1).collect().item()
        warnings.warn(
            "Parquet reader could not find a timestamp column",
            stacklevel=2,
        )
        return None


def parquet(
    filename: str | Path,
    extra_column_mappings: dict[str, str] | None = None,
    options: dict[str, str] | None = None,
) -> pl.DataFrame:
    return Parquet().run(
        filename,
        extra_column_mappings=extra_column_mappings,
        options=options,
    )
