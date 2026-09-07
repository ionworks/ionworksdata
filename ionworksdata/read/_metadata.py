"""Shared vocabulary for the descriptive metadata readers pull from a header.

Every cycler labels the same facts differently -- Maccor's ``Procedure``,
Basytec's ``Testplan``, Neware's ``Step file`` all name the schedule -- so
per-reader keys would give ``test_setup`` a different shape per cycler.

:data:`CANONICAL_METADATA_KEYS` is the contract. A field mapping to no
canonical key is dropped rather than passed through under its native name, so
widening the vocabulary is a deliberate edit here.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re
from typing import Any

import pytz  # type: ignore[reportMissingTypeStubs]

import ionworksdata as iwdata

#: Flat key -> ``(group, field)``. Keys absent here are dropped by
#: :func:`nest_metadata`; ``start_time`` maps to no group because the API
#: models it as a column of its own.
#:
#: Named for meaning, not for one cycler's label, so Arbin's ``Creator`` and
#: Biologic's ``User`` both land on ``operator``.
CANONICAL_METADATA_KEYS: dict[str, tuple[str, str]] = {
    # The schedule that produced the file -- what links a measurement to its
    # protocol on upload.
    "procedure": ("protocol", "name"),
    "channel_number": ("test_setup", "channel_number"),
    "export_date": ("test_setup", "export_date"),
    "source_file_path": ("test_setup", "source_file_path"),
    # An id in the cycler's own id space, *not* a cross-system identifier.
    "source_file_id": ("test_setup", "source_file_id"),
    "barcode_comment": ("test_setup", "barcode_comment"),
    "cycler": ("test_setup", "cycler"),
    "cycler_serial_number": ("test_setup", "cycler_serial_number"),
    "software_version": ("test_setup", "software_version"),
    "operator": ("test_setup", "operator"),
    "test_name": ("test_setup", "test_name"),
    # The cell as the *cycler* labelled it: a lab's own cell name or battery
    # type string, which need not match the cell instance it is uploaded to.
    "cell_label": ("test_setup", "cell_label"),
    "temperature_setpoint_degc": ("test_setup", "temperature_setpoint_degc"),
    "instrument": ("test_setup", "instrument"),
    "end_time": ("test_setup", "end_time"),
    "notes": ("test_setup", "notes"),
}


def nest_metadata(flat: dict[str, Any]) -> dict[str, Any]:
    """
    Group flat header fields under the measurement fields the API defines.

    Parameters
    ----------
    flat : dict[str, Any]
        Header fields keyed by :data:`CANONICAL_METADATA_KEYS` names. Values
        that are None or empty strings are treated as absent, so a reader can
        pass through whatever it parsed without filtering first.

    Returns
    -------
    dict[str, Any]
        The same values under their ``protocol`` / ``test_setup`` groups, with
        ``start_time`` left at the top level. A group with nothing in it is
        omitted rather than created empty, and a key outside the allowlist is
        dropped -- see this module's docstring for why.
    """
    nested: dict[str, Any] = {}
    for key, value in flat.items():
        if value is None or (isinstance(value, str) and not value.strip()):
            continue
        if key == "start_time":
            nested[key] = value
            continue
        group = CANONICAL_METADATA_KEYS.get(key)
        if group is None:
            continue
        nested.setdefault(group[0], {})[group[1]] = value
    return nested


#: Skips the re-read the "first data row" fallback costs. See
#: :func:`take_known_start_time`.
START_TIME_OPTION = "known_start_time"


def take_known_start_time(
    options: dict[str, Any] | None,
) -> tuple[datetime | None, dict[str, Any] | None]:
    """
    Split the caller-supplied start time out of the reader's options.

    Parameters
    ----------
    options : dict[str, Any] | None
        Reader options, possibly carrying :data:`START_TIME_OPTION`.

    Returns
    -------
    tuple[datetime | None, dict[str, Any] | None]
        The timestamp the caller already read out of the file, and the options
        with that key removed. It is removed because every reader validates its
        options against its own ``default_options`` and rejects a key it does
        not define -- this one is consumed by ``read_metadata`` itself and
        never reaches the parse.

    Raises
    ------
    ValueError
        If the value is neither a datetime nor None. A silent wrong type would
        surface as a missing start time much later.
    """
    if not options or START_TIME_OPTION not in options:
        return None, options
    remaining = {k: v for k, v in options.items() if k != START_TIME_OPTION}
    value = options[START_TIME_OPTION]
    if value is not None and not isinstance(value, datetime):
        raise ValueError(
            f"{START_TIME_OPTION} must be a datetime, got {type(value).__name__}"
        )
    return value, remaining


def localize(value: datetime, options: dict[str, Any]) -> datetime:
    """
    Attach the configured timezone to a naive header datetime.

    Parameters
    ----------
    value : datetime
        Naive datetime as parsed from the header.
    options : dict[str, Any]
        Reader options; ``timezone`` names the zone, defaulting to ``"UTC"``.

    Returns
    -------
    datetime
        Timezone-aware datetime, normalized the way the rest of the package
        expects.

    Raises
    ------
    ValueError
        If ``timezone`` is not a string.
    """
    timezone = options.get("timezone", "UTC")
    if not isinstance(timezone, str):
        raise ValueError(f"Invalid timezone: {timezone}")
    if value.tzinfo is not None:
        return iwdata.util.check_and_convert_datetime(value)
    return iwdata.util.check_and_convert_datetime(
        pytz.timezone(timezone).localize(value)
    )


def parse_datetime(value: str, formats: tuple[str, ...]) -> datetime | None:
    """
    Parse a header timestamp against several candidate formats.

    Parameters
    ----------
    value : str
        The header's raw timestamp text.
    formats : tuple[str, ...]
        ``strptime`` formats to try, in order.

    Returns
    -------
    datetime | None
        The first format that parses, or None when none of them do -- an
        unrecognized timestamp is a field the header does not usably carry,
        not an error worth failing the whole read for.
    """
    text = value.strip()
    for fmt in formats:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def parse_labelled_lines(
    lines: list[str],
    labels: dict[str, str],
    *,
    strip_prefix: str = "",
    separator: str = ":",
) -> dict[str, str]:
    """
    Read ``<label><separator><value>`` lines out of a header block.

    The one-field-per-line shape Novonix, Basytec, Biologic and the BaSyTec
    companion file all use, so the four readers share this rather than each
    walking the lines itself.

    Parameters
    ----------
    lines : list[str]
        Header lines, in file order.
    labels : dict[str, str]
        Lowercased label (without its separator) to canonical key. A label the
        block does not carry simply yields no key.
    strip_prefix : str, optional
        Comment marker each line carries, e.g. BaSyTec's ``"~"``. Stripped
        before the label is matched. Default is no prefix.
    separator : str, optional
        Character between label and value. Default ``":"``.

    Returns
    -------
    dict[str, str]
        Canonical key to stripped value, for the labels present. The first
        occurrence wins, since a header that repeats a label (Novonix writes
        ``Protocol`` in both ``[Summary]`` and ``[Protocol]``) states it in the
        summary first.
    """
    found: dict[str, str] = {}
    for line in lines:
        text = line.strip()
        if strip_prefix:
            text = text.lstrip(strip_prefix).strip()
        label, sep, value = text.partition(separator)
        if not sep:
            continue
        key = labels.get(label.strip().lower())
        if key is None or key in found:
            continue
        value = value.strip()
        if value:
            found[key] = value
    return found


#: A cycler names an export after the channel that produced it as a numeric
#: extension; the pattern is not Maccor-specific.
_NUMERIC_SUFFIX_RE = re.compile(r"^\d+$")


def channel_from_extension(filename: str | Path) -> int | None:
    """
    Read a channel number out of a numeric filename extension.

    Maccor names an export after the channel that produced it -- ``cell.057``
    is channel 57 -- and it is the only record of the channel in a dialect
    whose header carries no field for it.

    Parameters
    ----------
    filename : str | Path
        Path to the file.

    Returns
    -------
    int | None
        The channel, or None when the extension is not numeric (``.txt``,
        ``.csv``, ``.xlsx``), since those names say nothing about a channel.
    """
    suffix = Path(filename).suffix.lstrip(".")
    return int(suffix) if _NUMERIC_SUFFIX_RE.match(suffix) else None
