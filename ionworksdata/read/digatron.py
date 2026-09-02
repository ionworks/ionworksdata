"""Reader for Digatron CSV exports.

File format
    Header names carry their unit after a ``#`` delimiter, and that unit's
    capitalisation is not consistent within one file ("AhCha#AH" beside
    "AhDch#Ah"), so header lookup is case-insensitive. Timestamps are ISO-8601
    with a UTC offset, in two spellings: fractional seconds appear only when
    non-zero.

Time
    ``Time [s]`` is derived from ``Timestamp``, never from ``Program Duration#s``.
    Observed exports declare seconds in that header but write milliseconds, so
    trusting the ``#s`` suffix stretches the time axis a thousandfold. A duration
    column that disagrees with the wall clock is reported rather than used.

Current sign convention
    The vendor writes positive for charge. ``Status`` carries CHA/DCH per row, and
    ``set_positive_current_for_discharge`` reads those labels to reach the ionworks
    convention of positive = discharge.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import iwutil
import polars as pl

import ionworksdata as iwdata
from ionworksdata.logger import logger

from .read import BaseReader

# "AhCha#AH" sits alongside "AhDch#Ah", so lookups lowercase the header.
UNIT_DELIMITER = "#"

SIGNATURE_COLUMNS = ("timestamp", "voltage#v", "current#a")

# Tolerance on a "#s" column's span vs the wall clock. Some exports write
# milliseconds under a seconds header, stretching time a thousandfold.
DURATION_TOLERANCE = 0.01


class Digatron(BaseReader):
    name: str = "Digatron"
    default_options: dict[str, Any] = {
        "cell_metadata": {},
        "file_encoding": "utf-8",
        "strict_duration_units": False,
    }

    # Two timestamp spellings occur in a single file: fractional seconds are
    # written only when non-zero.
    _timestamp_formats = (
        "%Y-%m-%d %H:%M:%S%.f%:z",
        "%Y-%m-%d %H:%M:%S%:z",
    )

    _column_renamings = {
        "voltage#v": "Voltage [V]",
        "current#a": "Current [A]",
        "t1#degc": "Temperature [degC]",
        "tenv#degc": "Ambient temperature [degC]",
        "step": "Step from cycler",
        "cycle": "Cycle from cycler",
        "status": "Status",
        "timestamp": "Timestamp",
        "ahcha#ah": "Charge capacity [A.h]",
        "ahdch#ah": "Discharge capacity [A.h]",
        "whcha#wh": "Charge energy [W.h]",
        "whdch#wh": "Discharge energy [W.h]",
    }

    @classmethod
    def sniff(cls, header: str) -> bool:
        """Report whether a CSV header line looks like a Digatron export.

        Parameters
        ----------
        header : str
            The first line of the file.

        Returns
        -------
        bool
            True when every signature column is present.
        """
        columns = {c.strip().lower() for c in header.split(",")}
        return all(c in columns for c in SIGNATURE_COLUMNS)

    @staticmethod
    def _resolve_columns(columns: list[str]) -> dict[str, str]:
        """Map each lowercased header name to the name as written in the file."""
        return {col.lower(): col for col in columns}

    @classmethod
    def _parse_timestamp(cls, data: pl.DataFrame, column: str) -> pl.DataFrame:
        """Parse the timestamp column into a UTC datetime named ``Timestamp``."""
        parsed = pl.coalesce(
            *[
                pl.col(column).str.strptime(pl.Datetime, format=fmt, strict=False)
                for fmt in cls._timestamp_formats
            ]
        ).alias("Timestamp")
        # A trailing newline reaches Polars as an all-null row; that is file
        # punctuation, not a sample, so drop it before judging the column.
        data = data.with_columns(parsed).filter(~pl.all_horizontal(pl.all().is_null()))
        # Refuse rather than drop: "Time [s]" is derived from this column, so a
        # silently discarded row is a hole in the axis everything else indexes on.
        unparsed = data.filter(pl.col("Timestamp").is_null())
        if unparsed.height:
            sample = unparsed.get_column(column).head(3).to_list()
            raise ValueError(
                f"Could not parse {unparsed.height} of {data.height} values in the "
                f"Digatron '{column}' column as a timestamp (e.g. {sample}). Expected "
                f"ISO-8601 with a UTC offset, e.g. "
                f"'2025-06-06 09:10:07.970000+00:00'."
            )
        return data

    @classmethod
    def _check_duration_units(
        cls, data: pl.DataFrame, resolved: dict[str, str], strict: bool
    ) -> None:
        """Check that any ``#s`` duration column really is in seconds.

        Parameters
        ----------
        data : pl.DataFrame
            Data with a parsed ``Timestamp`` column.
        resolved : dict[str, str]
            Mapping from lowercased to original header names.
        strict : bool
            Raise on a mismatch instead of logging it.

        Raises
        ------
        ValueError
            If ``strict`` and a duration column's span differs from the wall-clock
            span by more than ``DURATION_TOLERANCE``.
        """
        wall_clock_span = data.select(
            (
                pl.col("Timestamp").max().dt.epoch("us")
                - pl.col("Timestamp").min().dt.epoch("us")
            )
            / 1e6
        ).item()
        if not wall_clock_span:
            return

        candidates = [
            original
            for folded, original in resolved.items()
            # A substring match on the vendor's own naming: a dialect calling it
            # "Time#s" gets no check rather than a wrong one.
            if folded.endswith(f"{UNIT_DELIMITER}s") and "duration" in folded
        ]
        for column in candidates:
            values = data.get_column(column).cast(pl.Float64, strict=False)
            # Only a column that never decreases spans the whole file; a per-step
            # duration restarts each step, so its span says nothing about units.
            if not bool((values.diff().drop_nulls() >= 0).all()):
                continue
            # Span between the real endpoints, not the maximum: a mid-file excerpt
            # starts part-way through. Dropping nulls would move the endpoints.
            first, last = values.first(), values.last()
            if first is None or last is None:
                continue
            declared_span = last - first
            ratio = declared_span / wall_clock_span
            if abs(ratio - 1.0) <= DURATION_TOLERANCE:
                continue
            message = (
                f"Digatron column '{column}' declares seconds but spans "
                f"{declared_span:.6g} over {wall_clock_span:.6g} s of wall-clock time "
                f"from the 'Timestamp' column, a factor of {ratio:.6g}. 'Time [s]' is "
                f"derived from 'Timestamp' and is unaffected, but the file's other "
                f"declared units may be wrong too."
            )
            if strict:
                raise ValueError(message)
            logger.error(message)

    def _read(
        self,
        filename: str | Path,
        file_encoding: str,
        strict_duration_units: bool = False,
    ) -> pl.DataFrame:
        """Read the CSV and parse its timestamp, checking the duration units."""
        data = pl.read_csv(
            filename,
            encoding=file_encoding,
            infer_schema_length=10000,
            truncate_ragged_lines=True,
        )
        resolved = self._resolve_columns(data.columns)

        missing = [c for c in SIGNATURE_COLUMNS if c not in resolved]
        if missing:
            raise ValueError(
                f"Digatron file is missing required column(s): {', '.join(missing)}. "
                f"Found: {', '.join(data.columns)}."
            )

        # After parsing, which drops the all-null rows such a file consists of.
        data = self._parse_timestamp(data, resolved["timestamp"])
        if data.height == 0:
            raise ValueError(
                f"Digatron file has a valid header but no data rows: {filename}"
            )
        data = data.sort("Timestamp")
        self._check_duration_units(data, resolved, strict_duration_units)
        return data

    def _apply_column_renamings(
        self,
        data: pl.DataFrame,
        extra_column_mappings: dict[str, str] | None = None,
    ) -> tuple[pl.DataFrame, list[str]]:
        """Rename the file's columns to canonical names, ignoring unit capitalisation."""
        resolved = self._resolve_columns(data.columns)
        renamings = {
            resolved[folded]: canonical
            for folded, canonical in self._column_renamings.items()
            if folded in resolved
        }
        renamings.update(extra_column_mappings or {})
        present = iwdata.util.resolve_renamings(
            renamings, data, priority=extra_column_mappings
        )
        data = data.rename(present)
        return data, list(present.values())

    def run(
        self,
        filename: str | Path,
        extra_column_mappings: dict[str, str] | None = None,
        options: dict[str, str] | None = None,
    ) -> pl.DataFrame:
        """
        Read and process data from a Digatron CSV file.

        Parameters
        ----------
        filename : str | Path
            Path to the Digatron CSV file to be read.
        extra_column_mappings : dict[str, str] | None, optional
            Dictionary of additional column mappings, keyed by the column name exactly
            as written in the file. Default is None. Applied on top of the defaults,
            which are matched case-insensitively because the unit's capitalisation
            varies within a single file:

            - "Voltage#V" -> "Voltage [V]"
            - "Current#A" -> "Current [A]"
            - "T1#degC" -> "Temperature [degC]"
            - "Tenv#degC" -> "Ambient temperature [degC]"
            - "Step" -> "Step from cycler"
            - "Cycle" -> "Cycle from cycler"
            - "AhCha#Ah" / "AhDch#Ah" -> "Charge/Discharge capacity [A.h]"
            - "WhCha#Wh" / "WhDch#Wh" -> "Charge/Discharge energy [W.h]"
        options : dict[str, str] | None, optional
            Dictionary of options to use when reading the Digatron file. Options are:

            - 'cell_metadata': dictionary of metadata about the cell.
            - 'file_encoding': text encoding for the CSV. Default is `'utf-8'`.
            - 'strict_duration_units': raise instead of logging when a ``#s`` duration
              column disagrees with the wall-clock elapsed time. Default is `False`,
              since ``Time [s]`` does not depend on those columns.

        Returns
        -------
        polars.DataFrame
            Processed data with standardized column names and units, with positive
            current denoting discharge.

        Raises
        ------
        ValueError
            If a required column is absent, if the timestamp cannot be parsed, or, when
            ``strict_duration_units`` is set, if a duration column declaring seconds
            disagrees with the wall-clock elapsed time.

        Notes
        -----
        ``Time [s]`` comes from ``Timestamp``, never from ``Program Duration#s``: some
        exports declare seconds in that header but write milliseconds, which would
        stretch the time axis a thousandfold.
        """
        opts: dict[str, Any] = iwutil.check_and_combine_options(
            self.default_options, options
        )
        data = self._read(
            filename, opts["file_encoding"], opts["strict_duration_units"]
        )
        data, renamed = self._apply_column_renamings(data, extra_column_mappings)

        start_epoch_us = data.select(pl.col("Timestamp").dt.epoch("us").min()).item()
        data = data.with_columns(
            ((pl.col("Timestamp").dt.epoch("us") - start_epoch_us) / 1e6).alias(
                "Time [s]"
            )
        )

        columns_keep = [col for col in renamed if col != "Timestamp"] + ["Time [s]"]
        return self.standard_data_processing(data, columns_keep=columns_keep)

    def read_start_time(
        self,
        filename: str | Path,
        extra_column_mappings: dict[str, str] | None = None,
        options: dict[str, str] | None = None,
    ) -> datetime:
        """
        Read the start time from a Digatron CSV file.

        Parameters
        ----------
        filename : str | Path
            Path to the Digatron CSV file to be read.
        extra_column_mappings : dict[str, str] | None, optional
            Unused; accepted for signature compatibility with the other readers.
        options : dict[str, str] | None, optional
            Options for reading the file. See :func:`ionworksdata.read.Digatron.run`.

        Returns
        -------
        datetime
            The timezone-aware start time of the measurement.
        """
        opts: dict[str, Any] = iwutil.check_and_combine_options(
            self.default_options, options
        )
        start_timestamp = self._scan_start_timestamp(filename, opts["file_encoding"])
        if start_timestamp is None:
            data = self._read(
                filename, opts["file_encoding"], opts["strict_duration_units"]
            )
            start_timestamp = data.select(pl.col("Timestamp").min()).item()
        return iwdata.util.check_and_convert_datetime(start_timestamp)

    @classmethod
    def _scan_start_timestamp(
        cls, filename: str | Path, file_encoding: str
    ) -> datetime | None:
        """Read only the timestamp column and return its minimum.

        Callers routinely ask for the time series and the start time of the same
        file, and these exports run to tens of MB across 22 columns. Scanning
        projects to the one column that matters instead of parsing all of them a
        second time. Returns None when the scan cannot be trusted -- a non-UTF-8
        file, since ``scan_csv`` takes no encoding, or a header this reader does not
        recognise -- so the caller falls back to the full read.
        """
        if file_encoding.lower().replace("-", "") not in {"utf8", "utf8lossy"}:
            return None
        try:
            lazy = pl.scan_csv(filename)
            column = next(
                (c for c in lazy.collect_schema().names() if c.lower() == "timestamp"),
                None,
            )
            if column is None:
                return None
            parsed = pl.coalesce(
                *[
                    pl.col(column).str.strptime(pl.Datetime, format=fmt, strict=False)
                    for fmt in cls._timestamp_formats
                ]
            )
            # `min()` ignores nulls, so count unparsed rows by the full read's rule
            # (all-null is a blank line, not a sample) and defer to it to raise.
            summary = (
                lazy.select(
                    parsed.min().alias("start"),
                    (parsed.is_null() & ~pl.all_horizontal(pl.all().is_null()))
                    .sum()
                    .alias("unparsed"),
                )
                .collect()
                .row(0)
            )
            start, unparsed = summary
            if unparsed or start is None:
                return None
            return start
        except Exception:
            return None


def digatron(
    filename: str | Path,
    extra_column_mappings: dict[str, str] | None = None,
    options: dict[str, str] | None = None,
) -> pl.DataFrame:
    return Digatron().run(
        filename, extra_column_mappings=extra_column_mappings, options=options
    )
