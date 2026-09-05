from __future__ import annotations

import csv as csv_py
from datetime import datetime
from io import BytesIO
from pathlib import Path
import re
from typing import Any

import iwutil
import numpy as np
import polars as pl
import pytz

import ionworksdata as iwdata

from ._utils import (
    find_data_sheet,
    is_maccor_text_extension,
    read_excel_and_get_column_names,
    suppress_excel_dtype_warnings,
)
from .read import BaseReader

# Signature columns for detection, beside the reader so a new dialect is
# added in one place rather than two.
_MACCOR_COLUMNS = (
    "step",
    "test time",
    "test (sec)",
    "test time (sec)",
    "testtime",
    "prog time",
    "current (a)",
    "current",
    "amps",
    "voltage (v)",
    "voltage",
    "volts",
    "cycle",
    "cyc#",
    "cycle id",
    "logtemp001",
    "temperature (°c)",
    "status",
    "state",
    "md",
)

# Below this, the columns above are too generic to name Maccor on their own.
_MACCOR_MIN_SIGNATURE_COLUMNS = 3

_MACCOR_TIME_COLUMNS = (
    "test time",
    "test (sec)",
    "test(sec)",
    "prog time",
    "testtime",
)


_MACCOR_DATA_SHEET_REQUIRED = ("step", "test time")

# Maccor writes the time column as "TestTime" in some exports; the shared
# normalizer separates dialects on whitespace and underscores, not case runs.
_MACCOR_HEADER_ALIASES = {"testtime": "test time"}


# Units inline in the header, e.g. ``Test Time [s]``.
_HEADER_INLINE_UNIT_RE = re.compile(r"\[[^\]]+\]\s*$")

# Bracketed units are rewritten to parentheses so both dialects share one key.
_CANONICAL_BRACKET_RE = re.compile(r"\[([^\]]*)\]")
# Separator characters Maccor dialects use interchangeably between words.
_CANONICAL_SEPARATOR_RE = re.compile(r"[\s_-]+")
# Whitespace padding a unit group, e.g. ``Test (Sec)`` vs ``Test(Sec)``.
_CANONICAL_UNIT_PAD_RE = re.compile(r"\s*\(\s*([^)]*?)\s*\)")

_CANONICAL_UNIT_ALIASES: dict[str, str] = {
    "ahr": "ah",
    "whr": "wh",
    "sec": "s",
}

# Split before the case fold, which would otherwise merge "mAh" into "MAh".
_CANONICAL_UNIT_PREFIX_RE = re.compile(r"^(m|M)(?=[AW])(.+)$")
_CANONICAL_UNIT_PREFIX_MARKERS = {"m": "milli-", "M": "mega-"}


def _canonical_unit(unit: str) -> str:
    """Fold a unit group to its canonical spelling.

    Parameters
    ----------
    unit : str
        Unit text from inside the brackets, e.g. ``"AHr"`` or ``"mAHr"``.

    Returns
    -------
    str
        Canonical unit, e.g. ``"ah"`` or ``"milli-ah"``.

    Notes
    -----
    A leading milli or mega prefix is split off and rewritten to a spelled-out
    marker, and the remainder is folded through :data:`_CANONICAL_UNIT_ALIASES`
    — so ``mAHr`` reaches ``milli-ah`` via the same ``ahr`` -> ``ah`` alias
    that handles ``AHr``, without a separate row per prefixed spelling. Mega
    (``MAHr``) canonicalizes to ``mega-ah``, a distinct key that matches no
    column, so it fails as an unrecognized column rather than being silently
    rescaled by 1e-3.
    """
    stripped = unit.strip()
    if match := _CANONICAL_UNIT_PREFIX_RE.match(stripped):
        prefix, rest = match.groups()
        return _CANONICAL_UNIT_PREFIX_MARKERS[prefix] + _canonical_unit(rest)
    lowered = stripped.lower()
    return _CANONICAL_UNIT_ALIASES.get(lowered, lowered)


def _canonical_header(name: str) -> str:
    """Fold a Maccor header to its canonical form.

    Parameters
    ----------
    name : str
        Raw header string from the file, e.g. ``"Test Time [Sec]"``.

    Returns
    -------
    str
        Canonical form, e.g. ``"TestTime(s)"``.

    Notes
    -----
    Separators are *deleted*, not normalized to a space, so ``TestTime`` folds
    together with ``Test Time`` — normalizing cannot insert a space that
    spelling never had. Safe because no two distinct Maccor columns differ only
    by separator placement (``Cycle P``/``Cycle C``, ``Test Time``/``Step
    Time`` all stay apart).

    Case is preserved on purpose: only ``MD`` is mapped, and newly mapping
    ``Md`` would change the downstream current-sign correction — a behaviour
    fix, not this refactor.
    """
    canonical = _CANONICAL_BRACKET_RE.sub(r"(\1)", name.strip())
    # The unit group is lowercased, unlike the column name: unit spelling never
    # distinguishes two Maccor columns, but ``MD`` vs ``Md`` does.
    canonical = _CANONICAL_UNIT_PAD_RE.sub(
        lambda m: f"({_canonical_unit(m.group(1))})", canonical
    )
    return _CANONICAL_SEPARATOR_RE.sub("", canonical).strip()


def _canonical_map(raw: dict[str, str]) -> dict[str, str]:
    """Canonicalize the keys of a raw-header map, rejecting collisions.

    Parameters
    ----------
    raw : dict[str, str]
        Map from a representative raw header spelling to its ionworks name.

    Returns
    -------
    dict[str, str]
        The same map keyed on :func:`_canonical_header` of each key.

    Raises
    ------
    ValueError
        If two keys canonicalize together but disagree on the target, which
        would silently route one source column to the wrong ionworks column.
    """
    canonical: dict[str, str] = {}
    for name, target in raw.items():
        key = _canonical_header(name)
        existing = canonical.get(key)
        if existing is not None and existing != target:
            raise ValueError(
                f"Maccor column map collision: {name!r} canonicalizes to "
                f"{key!r}, which already maps to {existing!r}, not {target!r}."
            )
        canonical[key] = target
    return canonical


# One entry per column: a new dialect needs a row only when it *names* a column
# differently, not when it merely punctuates it differently.
_MACCOR_CANONICAL_MAP: dict[str, str] = _canonical_map(
    {
        "Voltage": "Voltage [V]",
        "Volts": "Voltage [V]",
        "Voltage (V)": "Voltage [V]",
        "Current": "Current [A]",
        "Amps": "Current [A]",
        "Current (A)": "Current [A]",
        # Intermediate, not "Current [A]": renaming without the rescale in
        # _convert_milli_units reports current 1000x too large.
        "Current (mA)": "Current [mA]",
        "mAmps": "Current [mA]",
        "Prog Time": "Time [s]",
        "Test (Sec)": "Time [s]",
        "Test Time (sec)": "Time [s]",
        "Test Time (Hr)": "Time [h]",
        "Cycle": "Cycle from cycler",
        "Cyc#": "Cycle from cycler",
        "Cycle ID": "Cycle from cycler",
        # "Cycle P" / "Cycle C" are resolved in _resolve_cycle_column (they
        # are deliberately omitted here); see that method and issue #2.
        "Step": "Step from cycler",
        "Step ID": "Step from cycler",
        "LogTemp001": "Temperature [degC]",
        "Temperature (°C)": "Temperature [degC]",
        "EVTemp (C)": "Temperature [degC]",
        "Temperature Cell (degC)": "Temperature [degC]",
        "DCIR (Ohms)": "DC resistance [Ohm]",
        "Status": "Status",
        "State": "Status",
        "MD": "Status",
        # Raw: the public documentation does not enumerate the end codes, so
        # any mapping to names would be invented.
        "ES": "End status from cycler",
        "DPT": "Timestamp",
        "DPT Time": "Timestamp",
        # Case is not folded, so Landt's "DPt-Time" needs its own row; the
        # separator fold makes it cover "DPt Time" as well.
        "DPt Time": "Timestamp",
    }
)

_MACCOR_CANONICAL_CAPACITY_MAP: dict[str, str] = _canonical_map(
    {
        "Capacity (Ah)": "Capacity [A.h]",
        "Cap. (Ah)": "Capacity [A.h]",
        "Amp-hr": "Capacity [A.h]",
        "Energy (Wh)": "Energy [W.h]",
        "Watt-hr": "Energy [W.h]",
        "Chg Capacity (Ah)": "Charge capacity [A.h]",
        "DChg Capacity (Ah)": "Discharge capacity [A.h]",
        "Chg Energy (Wh)": "Charge energy [W.h]",
        "DChg Energy (Wh)": "Discharge energy [W.h]",
        # Milli-prefixed spellings, rescaled by _convert_milli_units
        "Capacity (mAh)": "Capacity [mA.h]",
        "Cap. (mAh)": "Capacity [mA.h]",
        "mAmp-hr": "Capacity [mA.h]",
        "Energy (mWh)": "Energy [mW.h]",
        "mWatt-hr": "Energy [mW.h]",
        "Chg Capacity (mAh)": "Charge capacity [mA.h]",
        "DChg Capacity (mAh)": "Discharge capacity [mA.h]",
        "Chg Energy (mWh)": "Charge energy [mW.h]",
        "DChg Energy (mWh)": "Discharge energy [mW.h]",
    }
)

# _canonical_map only guards within one literal, but the two are merged in
# _get_column_renamings, where a clash would override instead of raising.
if _shared := _MACCOR_CANONICAL_MAP.keys() & _MACCOR_CANONICAL_CAPACITY_MAP.keys():
    raise ValueError(
        f"Maccor column map collision across the base and capacity maps: "
        f"{sorted(_shared)}. A canonical key must belong to exactly one."
    )

# The header is the only record of a coin-cell export's mA/mAHr scale, so the
# rescale has to happen on read rather than be left to the caller.
_MACCOR_MILLI_UNIT_COLUMNS: dict[str, str] = {
    "Current [mA]": "Current [A]",
    "Capacity [mA.h]": "Capacity [A.h]",
    "Charge capacity [mA.h]": "Charge capacity [A.h]",
    "Discharge capacity [mA.h]": "Discharge capacity [A.h]",
    "Energy [mW.h]": "Energy [W.h]",
    "Charge energy [mW.h]": "Charge energy [W.h]",
    "Discharge energy [mW.h]": "Discharge energy [W.h]",
}

_MACCOR_CAPACITY_TARGETS = frozenset(_MACCOR_CANONICAL_CAPACITY_MAP.values())

# Targets whose source column may hold a duration or datetime rather than a
# number, so _process_test_time_column has to sniff the values.
_MACCOR_TIME_TARGETS = frozenset({"Time [s]", "Time [h]"})

# Unmapped on purpose: it may hold seconds, a duration, or a datetime.
_MACCOR_BARE_TEST_TIME = _canonical_header("Test Time")

# "Procedure:" sits *inside* the "Filename:" field, so values end at the next
# label, not at a delimiter. A label missing here means a value runs on.
_MACCOR_HEADER_LABELS = (
    "Today's Date",
    "Date of Test",
    "Filename",
    "Procedure",
    "Comment/Barcode",
    "TestName",
    "TestDesc",
    "ActiveMaterial",
    "Description",
    "Started",
)
# Colon-optional for the rest would truncate "... Description Run" at a word.
_MACCOR_COLONLESS_LABELS = frozenset({"Today's Date"})
_MACCOR_LABEL_BOUNDARY = "|".join(
    rf"{re.escape(label)}\s*:" + ("?" if label in _MACCOR_COLONLESS_LABELS else "")
    for label in _MACCOR_HEADER_LABELS
)


def _labelled_value_re(label: str, group: str) -> re.Pattern[str]:
    """Match ``<label>: <value>`` up to the next known label or end of line."""
    return re.compile(
        rf"{re.escape(label)}\s*:\s*(?P<{group}>.+?)"
        rf"\s*(?:{_MACCOR_LABEL_BOUNDARY}|$)",
        # Dialects differ on case (``Filename`` vs ``FileName``).
        re.IGNORECASE,
    )


_MACCOR_PROCEDURE_RE = _labelled_value_re("Procedure", "procedure")
_MACCOR_SOURCE_PATH_RE = _labelled_value_re("Filename", "path")
_MACCOR_BARCODE_RE = _labelled_value_re("Comment/Barcode", "barcode")
_MACCOR_EXPORT_DATE_RE = re.compile(
    r"Today's Date\s*:?\s*(?P<date>\d{2}/\d{2}/\d{4})",
)

# Only the bracketed form carries a machine id; the bare form is a free-text note.
_MACCOR_BARCODE_ID_RE = re.compile(r"^\[(?P<id>[^\]]+)\]\s*(?P<comment>.*)$")


def _flatten_header(header_text: str) -> str:
    """Collapse a possibly multi-line header into the single line the fields assume."""
    return " ".join(header_text.split())


# The per-row dialect names its fields without the flat dialect's trailing
# colon, so the two label vocabularies do not overlap.
_MACCOR_ROW_METADATA_KEYS = {
    "procedure": "procedure",
    "filename": "source_file_path",
    "comment/barcode": "barcode",
    "channel": "channel_number",
    # This dialect's start time; the flat one writes "Date of Test".
    "started": "started",
}

# Only this dialect writes a time of day alongside the date.
_MACCOR_ROW_START_FORMATS = ("%m/%d/%Y %H:%M", "%m/%d/%Y %H:%M:%S", "%m/%d/%Y")


def _parse_row_header(header_text: str, sep: str) -> dict[str, str]:
    """
    Extract metadata from the ``<key><sep><value>`` per-row header dialect.

    Parameters
    ----------
    header_text : str
        The header rows, newline-separated.
    sep : str
        Delimiter the file uses, as detected by :meth:`Maccor._get_file_args`.

    Returns
    -------
    dict[str, str]
        Metadata keys the rows carry, before the barcode is split up.

    Notes
    -----
    Every row is padded out to the width of the data columns, and the dialect
    does not quote its values, so a value whose final character is the
    delimiter is byte-identical to one followed by padding. Trailing
    delimiters are therefore trimmed with the padding; delimiters anywhere
    else in a value are preserved.
    """
    metadata: dict[str, str] = {}
    for line in header_text.splitlines():
        key, _, rest = line.partition(sep)
        target = _MACCOR_ROW_METADATA_KEYS.get(key.strip().rstrip(":").lower())
        if target is None:
            continue
        # Trim the padding rather than splitting into fields, so a delimiter
        # inside a free-text value survives.
        value = rest.rstrip(sep + " \t").strip()
        if value:
            metadata[target] = value
    return metadata


def _parse_flat_header(flat_header: str) -> dict[str, str]:
    """
    Extract metadata from the single-line header dialect.

    Parameters
    ----------
    flat_header : str
        The header collapsed to one line by :func:`_flatten_header`.

    Returns
    -------
    dict[str, str]
        Metadata keys the line carries, before the barcode is split up.
    """
    matches = {
        "procedure": _MACCOR_PROCEDURE_RE.search(flat_header),
        "source_file_path": _MACCOR_SOURCE_PATH_RE.search(flat_header),
        "export_date": _MACCOR_EXPORT_DATE_RE.search(flat_header),
        "barcode": _MACCOR_BARCODE_RE.search(flat_header),
    }
    values = {key: match.group(1).strip() for key, match in matches.items() if match}
    return {key: value for key, value in values.items() if value}


def _parse_barcode(barcode: str) -> dict[str, Any]:
    """
    Split a ``Comment/Barcode`` value into its id and its comment.

    Maccor writes this field in two shapes: a bare free-text note, or a
    ``[<id>] <text>`` form where the id belongs to the cycler's own id space.

    The comment is kept verbatim: reading structure out of it misparses the
    ordinary notes that share the shape -- ``EC:DEC 30:70`` is an electrolyte
    spec, not a composition.
    """
    parsed: dict[str, Any] = {}
    remainder = barcode

    id_match = _MACCOR_BARCODE_ID_RE.match(remainder)
    if id_match:
        parsed["source_file_id"] = id_match.group("id").strip()
        remainder = id_match.group("comment").strip()

    parsed["barcode_comment"] = remainder
    # Either part can come out empty; absent must mean absent, not empty.
    return {key: value for key, value in parsed.items() if value}


def _localize(value: datetime, options: dict[str, str]) -> datetime:
    """Attach the configured timezone to a naive header datetime."""
    timezone = options.get("timezone", "UTC")
    if not isinstance(timezone, str):
        raise ValueError(f"Invalid timezone: {timezone}")
    return iwdata.util.check_and_convert_datetime(
        value.replace(tzinfo=pytz.timezone(timezone))
    )


def _parse_row_start_time(value: str, options: dict[str, str]) -> datetime | None:
    """Parse the row dialect's ``Started`` value, which carries a time of day."""
    for fmt in _MACCOR_ROW_START_FORMATS:
        try:
            parsed = datetime.strptime(value, fmt)
        except ValueError:
            continue
        # Localized outside the try: a bad timezone option is a configuration
        # error, not one more format that did not match.
        return _localize(parsed, options)
    return None


# The API ignores fields it does not define, so a flat `procedure` would be
# dropped on upload rather than rejected. `start_time` is a column of its own.
_MACCOR_METADATA_LAYOUT: dict[str, tuple[str, str]] = {
    "procedure": ("protocol", "name"),
    "channel_number": ("test_setup", "channel_number"),
    "export_date": ("test_setup", "export_date"),
    "source_file_path": ("test_setup", "source_file_path"),
    "source_file_id": ("test_setup", "source_file_id"),
    "barcode_comment": ("test_setup", "barcode_comment"),
}


def _nest_metadata(flat: dict[str, Any]) -> dict[str, Any]:
    """
    Group flat header fields under the measurement fields the API defines.

    Parameters
    ----------
    flat : dict[str, Any]
        Header fields as parsed, keyed by their own names.

    Returns
    -------
    dict[str, Any]
        The same values under ``protocol`` / ``test_setup``, plus any key with
        no mapping (``start_time``) left at the top level. A group with nothing
        in it is omitted rather than created empty.
    """
    nested: dict[str, Any] = {}
    for key, value in flat.items():
        group = _MACCOR_METADATA_LAYOUT.get(key)
        if group is None:
            nested[key] = value
        else:
            nested.setdefault(group[0], {})[group[1]] = value
    return nested


def _channel_from_extension(filename: str | Path) -> int | None:
    """
    Read the channel number out of a Maccor filename extension.

    Maccor names an export after the channel that produced it, as a numeric
    extension -- ``cell_042.057`` is channel 57. It is the only record of
    the channel in the flat dialect, whose header does not carry a field for it.

    Parameters
    ----------
    filename : str | Path
        Path to the Maccor file.

    Returns
    -------
    int | None
        The channel, or None when the extension is not numeric (``.txt``,
        ``.csv``, ``.xlsx``), since those names say nothing about a channel.
    """
    suffix = Path(filename).suffix.lstrip(".")
    return int(suffix) if suffix.isdecimal() else None


def _parse_maccor_header(header_text: str, sep: str) -> dict[str, Any]:
    """
    Extract the descriptive metadata a Maccor header carries alongside the test date.

    Parameters
    ----------
    header_text : str
        The header rows as read from the file.
    sep : str
        Delimiter the file uses, as detected by :meth:`Maccor._get_file_args`.
        Only the comma-delimited dialect writes a field per row, so it gates
        that parse; the flat-line parse is tried for either delimiter.

    Returns
    -------
    dict[str, Any]
        Any of ``procedure``, ``export_date``, ``source_file_path``,
        ``source_file_id``, ``barcode_comment`` and ``channel_number`` that the
        header actually contains. Fields absent from the header are omitted
        rather than being reported as empty values, because most Maccor exports
        carry only a subset of them.

        Plus ``started`` for the per-row dialect: the raw unparsed string, which
        :meth:`Maccor.read_metadata` pops and turns into ``start_time``. It is a
        handoff to that method rather than part of the metadata, since parsing
        it needs the ``timezone`` option this function does not take.

    Notes
    -----
    Values are bounded by the header's label vocabulary, which is what stops
    one running into the next field. The cost is that a free-text value
    containing a label followed by a colon loses its tail --
    ``Comment/Barcode: see Description: of cell`` yields ``see``. The header
    offers nothing better to bound on.
    """
    metadata = _parse_flat_header(_flatten_header(header_text))
    # A comma-delimited header may be either dialect, so the per-row parse fills
    # in what the flat regexes could not find rather than replacing them.
    if sep == ",":
        metadata = {**_parse_row_header(header_text, sep), **metadata}

    barcode = metadata.pop("barcode", None)
    if barcode:
        metadata.update(_parse_barcode(barcode))

    # The API models channel_number as an int; a non-numeric channel is a
    # label, so it is dropped rather than passed through as the wrong type.
    channel = metadata.pop("channel_number", None)
    if channel is not None and channel.isdecimal():
        metadata["channel_number"] = int(channel)
    return metadata


_MACCOR_LOOP_RE = re.compile(r"^Loop\s*(?P<index>\d+)$", re.IGNORECASE)
# Mirrors ionworks_ucp's own register namespace, so a register the protocol
# writer can set is one the reader can read back.
_MACCOR_REGISTER_RE = re.compile(r"^(?P<kind>VAR|FLAG)\s*(?P<index>\d+)$")

_MACCOR_PASSTHROUGH_COLUMNS = """Maccor bookkeeping carried through as optional extras.

Named with the existing ``... from cycler`` suffix, which already marks the
columns belonging to the cycler rather than to the canonical schema
(``Step from cycler``, ``Cycle from cycler``). All are optional: a file that
did not select one yields no such column rather than an all-null one, and
:func:`ionworksdata.read.keep_required_columns` drops them, so the
required-column contract is unchanged.

- ``Mode from cycler`` -- ``MD``/``State``/``Status``, one letter per row:
  ``C`` charge, ``D`` discharge, ``R`` rest, plus the ``P``/``S``/``O``
  markers Maccor writes at a step transition, which carry no current or
  capacity and are otherwise indistinguishable from a reading at zero current.
- ``End status from cycler`` -- ``ES``, the per-step end code, raw because the
  public documentation does not enumerate the codes. Blank stays null: the
  cycler recorded no code there, and 0 is a different reading.
- ``Loop N from cycler`` -- ``Loop1``..``Loop4``. Maccor emits all four
  regardless of nesting depth, so unused levels are present and zero.
- ``VARn``/``FLAGn from cycler`` -- the ``SetVar`` registers, matched by
  pattern so an arbitrary number is read. A register reads 0 before its first
  assignment, which is Maccor's default rather than a gap.
"""


def _match_pattern_columns(columns: list[str]) -> dict[str, str]:
    """Map the numbered loop-counter and register columns.

    Matched by pattern, not listed in :data:`_MACCOR_CANONICAL_MAP`, because
    the vocabulary is open-ended. Anchored, so ``VARIANCE`` is not a register.

    Parameters
    ----------
    columns : list[str]
        Raw header strings as read from the file.

    Returns
    -------
    dict[str, str]
        Map from the raw header to its ionworks name, covering only the
        numbered columns this file actually has. An export that selected no
        loop counters or registers yields an empty map, so nothing is
        materialized as an all-null column.

    Notes
    -----
    Matched on the canonical header, so ``Loop 1`` and ``Loop1`` fold together
    the way the fixed vocabulary's spellings do.

    Maccor emits all four loop counters regardless of whether the procedure
    nests that deep, so unused levels are present and zero; likewise a
    register reads 0 before its first assignment. Both are real readings
    rather than gaps.
    """
    matched: dict[str, str] = {}
    for col in columns:
        canonical = _canonical_header(col)
        if loop := _MACCOR_LOOP_RE.match(canonical):
            matched[col] = f"Loop {int(loop.group('index'))} from cycler"
        elif register := _MACCOR_REGISTER_RE.match(canonical):
            kind = register.group("kind")
            matched[col] = f"{kind}{int(register.group('index'))} from cycler"
    return matched


# "Status" drives the current-sign correction and is then dropped, so carrying
# the mode through needs a second, retained copy.
_MACCOR_MODE_COLUMN = "Mode from cycler"


_MACCOR_TARGET_COLUMNS: tuple[str, ...] = (
    "Time [s]",
    "Voltage [V]",
    "Current [A]",
    "Cycle from cycler",
    "Step from cycler",
    "Capacity [A.h]",
    "Charge capacity [A.h]",
    "Discharge capacity [A.h]",
    "Energy [W.h]",
    "Charge energy [W.h]",
    "Discharge energy [W.h]",
    "Temperature [degC]",
    # Not the EIS quartet: measured from a current step, not a frequency sweep.
    "DC resistance [Ohm]",
)


class Maccor(BaseReader):
    name: str = "Maccor"
    default_options: dict[str, Any] = {
        "file_encoding": "ISO-8859-1",
        "timezone": "UTC",
        "cell_metadata": {},
        "time_offset_fix": -1,  # Minimum time difference to enforce. -1 means raise error on non-increasing time
        "skip_capacity_columns": False,  # If True, skip capacity/energy columns and compute from current/power
    }

    @staticmethod
    def _get_file_args(
        filename: str | Path, options: dict[str, str] | None = None
    ) -> tuple[str, list[int], str, str | None, str | None, bool]:
        # Find how many header rows to skip and set the read kwargs based on the file extension
        encoding = options["file_encoding"]
        thousands = None
        is_excel = False
        ext = Path(filename).suffix.lower()

        if ext in [".xls", ".xlsx"]:
            # Excel files - return special flag
            is_excel = True
            # For Excel, we'll handle header detection separately
            return encoding, [], ",", None, None, is_excel

        with open(filename, encoding=encoding) as f:
            if ext == ".csv":
                # Detect delimiter: some Maccor .csv files are tab-separated (e.g. export)
                skiprows = None
                sep = ","
                units_row = True
                comment = "#"
                # Streamed, not readlines(): read_metadata peeks at a header
                # behind which the data rows can be gigabytes.
                for i, line in enumerate(f):
                    for candidate_sep, has_units in [(",", True), ("\t", False)]:
                        row = [c.strip() for c in line.split(candidate_sep)]
                        if "Step" in row:
                            skiprows = i
                            sep = candidate_sep
                            # No units row follows a self-describing header;
                            # consuming one would eat the first record.
                            units_row = has_units and not any(
                                _HEADER_INLINE_UNIT_RE.search(c) for c in row
                            )
                            if sep == "\t":
                                thousands = ","
                            break
                    if skiprows is not None:
                        break
                if skiprows is None:
                    raise ValueError("Could not find header row in Maccor file")
            elif is_maccor_text_extension(ext):
                units_row = False
                comment = None
                skiprows = None
                sep = "\t"
                thousands = ","
                for i, line in enumerate(f):
                    for candidate_sep, candidate_thousands in [
                        ("\t", ","),
                        (",", None),
                    ]:
                        row = [c.strip() for c in line.split(candidate_sep)]
                        if "Step" in row:
                            skiprows = i
                            sep = candidate_sep
                            thousands = candidate_thousands
                            break
                    if skiprows is not None:
                        break
                if skiprows is None:
                    raise ValueError("Could not find header row in Maccor file")
            else:
                raise ValueError(f"Unsupported file extension: {ext}")
        if units_row:
            # skip all the header rows, plus the row after the header (which contains units)
            skiprows = list(range(skiprows)) + [skiprows + 1]
        else:
            skiprows = list(range(skiprows))
        return encoding, skiprows, sep, comment, thousands, is_excel

    def run(
        self,
        filename: str | Path,
        extra_column_mappings: dict[str, str] | None = None,
        options: dict[str, str] | None = None,
    ) -> pl.DataFrame:
        """
        Read and process data from a Maccor file.

        Headers are canonicalized before lookup, so a mapping covers that
        column's bracket, separator and unit-alias variants; only case is
        significant. See :func:`_canonical_header` for the folding and
        :meth:`_get_column_renamings` for the mappings.

        "Cycle from cycler" and "Time [s]" are resolved from the data, not the
        header alone — see :meth:`_resolve_cycle_column` and
        :meth:`_process_test_time_column`. Extra mappings can be supplied via
        ``extra_column_mappings``.

        Parameters
        ----------
        filename : str | Path
            Path to the Maccor file to be read. Supports:
            - .txt files (tab-separated)
            - .csv files (comma-separated with units row)
            - .xls/.xlsx files (Excel format)
            - Files with .+3digits extension (e.g., .123, .456)
        extra_column_mappings : dict of str to str, optional
            Dictionary of additional column mappings to use when reading the Maccor file.
            The keys are the original column names and the values are the new column
            names. Default is None.
        options : dict of str to str, optional
            Dictionary of options to use when reading the Maccor file.  Options are:

            - file_encoding: str, optional
                Encoding format for the Maccor file. Default is "ISO-8859-1".
                Note: encoding is not used for Excel files.
            - timezone: str, optional
                Timezone to use for the Maccor file. Default is "UTC".
            - time_offset_fix: float, optional
                Minimum time difference to enforce between consecutive points.
                If -1 (default), raises ValueError when time decreases or duplicates.
                If >= 0, ensures all time differences are at least this value using
                vectorized operations: fixed_diff = max(diff(time), time_offset_fix),
                then reconstructs time via cumsum.
            - skip_capacity_columns: bool, optional
                If True, skip reading capacity and energy columns from the raw file.
                This forces ionworksdata to compute capacity/energy from current/power
                integration instead. Useful when raw capacity data has resets or
                other issues. Default is False.

        Returns
        -------
        pl.DataFrame
            Processed data from the Maccor file with standardized column names and units.
        """
        options = iwutil.check_and_combine_options(self.default_options, options)

        # Load data and rename columns
        encoding, skiprows, sep, comment, thousands, is_excel = self._get_file_args(
            filename, options
        )

        if is_excel:
            # Handle Excel files
            data = self._read_excel_file(filename, encoding)
        else:
            # Derive header row index and whether a units row exists from the skiprows list
            # skiprows contains all pre-header rows and optionally the units row after the header
            skip_set = set(skiprows)
            # header row is the smallest non-skipped row index
            header_index = 0
            while header_index in skip_set:
                header_index += 1
            units_row_present = (header_index + 1) in skip_set

            read_kwargs = {
                "separator": sep,
                "skip_rows": header_index,
                "truncate_ragged_lines": True,
                # dtypes will be set below after extracting header columns
            }
            if units_row_present:
                read_kwargs["skip_rows_after_header"] = 1
            # Avoid passing comment handling for broad Polars version compatibility
            # We'll rely on skip logic and header detection above.

            # Polars only supports 'utf8' and 'utf8-lossy'. If a different encoding is
            # requested, decode manually and pass a BytesIO buffer to polars.
            encoding_lower = (encoding or "utf8").lower()
            # Read header line to build a dtypes mapping that forces all columns to Utf8
            with open(filename, encoding=encoding) as f:
                # advance to header row
                for _ in range(header_index):
                    f.readline()
                header_line = f.readline()
            header_reader = csv_py.reader([header_line], delimiter=sep)
            header_cols = next(header_reader)
            dtypes_map = dict.fromkeys(header_cols, pl.Utf8)
            # Use schema_overrides (newer Polars) but stay compatible with older versions
            read_kwargs["schema_overrides"] = dtypes_map
            if "dtypes" in read_kwargs:
                read_kwargs.pop("dtypes", None)

            if encoding_lower in {"utf8", "utf-8", "utf8-lossy"}:
                # Map to polars-supported encodings
                read_kwargs["encoding"] = (
                    "utf8" if encoding_lower in {"utf8", "utf-8"} else "utf8-lossy"
                )
                data = pl.read_csv(filename, **read_kwargs)
            else:
                with open(filename, encoding=encoding) as f:
                    content = f.read()
                data = pl.read_csv(
                    BytesIO(content.encode("utf-8")), encoding="utf8", **read_kwargs
                )

        # Resolve canonical map -> this file's actual headers, then process Test Time
        column_renamings = self._match_columns(
            data.columns, self._get_column_renamings(options)
        )
        column_renamings.update(_match_pattern_columns(data.columns))
        column_renamings = self._resolve_cycle_column(data, column_renamings)
        data, column_renamings = self._process_test_time_column(data, column_renamings)

        column_renamings.update(extra_column_mappings or {})
        existing_renames = iwdata.util.resolve_renamings(
            column_renamings, data, priority=extra_column_mappings
        )
        if existing_renames:
            data = data.rename(existing_renames)

        # If STATUS column is present, drop any rows where STATUS is MSG
        # and convert to single letter format (e.g. "D" not "DCH" for discharge)
        # Note: This is done after column renaming so it catches files with
        # status columns named "State" or "MD" (which are renamed to "Status")
        if "Status" in data.columns:
            data = data.filter(pl.col("Status") != "MSG")
            data = data.with_columns(
                pl.col("Status").cast(pl.Utf8).str.slice(0, 1).alias("Status")
            )
            # After the slice, so the retained copy and the sign correction
            # read the same single-letter values.
            data = data.with_columns(
                pl.col("Status").alias(_MACCOR_MODE_COLUMN),
            )

        # If numbers were read as strings (e.g., due to thousands separators), coerce to numeric
        data = self._coerce_numeric_columns(data)

        # Parse Timestamp column and compute Time [s] if needed
        data = self._parse_timestamp_column(data)

        # Convert time to seconds
        if "Time [h]" in data.columns:
            # Ensure numeric then convert
            data = self._coerce_numeric(data, "Time [h]")
            data = data.with_columns((pl.col("Time [h]") * 3600.0).alias("Time [s]"))
            data = data.drop("Time [h]")

        # Before the two current steps below, which read "Current [A]".
        data = self._convert_milli_units(data)

        # Fix unsigned current if needed
        data = self._fix_unsigned_current(data)

        # Fix current sign convention if needed (positive current should be discharge)
        data = iwdata.transform.set_positive_current_for_discharge(
            data, options=options
        )

        # Validate and optionally fix time to be strictly increasing
        # Do this BEFORE standard_data_processing to avoid losing duplicate timestamps
        time_offset_fix = options.get("time_offset_fix", -1)
        data = self._validate_and_fix_time(data, time_offset_fix)

        # Not intersected with the frame: standard_data_processing drops absent
        # ones, and naming "Time [s]" here covers files that derive it.
        skip_capacity = options.get("skip_capacity_columns", False)
        dropped = {"Time [h]", "Status", "Timestamp"}
        columns_keep = [
            col
            for col in _MACCOR_TARGET_COLUMNS
            if not (skip_capacity and col in _MACCOR_CAPACITY_TARGETS)
        ]
        # Appended, so an explicit caller mapping outranks skip_capacity_columns,
        # which governs only our own mappings. Their order is the file's.
        already = set(columns_keep)
        columns_keep += [
            si
            for col in dict.fromkeys(column_renamings.values())
            if (si := _MACCOR_MILLI_UNIT_COLUMNS.get(col, col)) not in already
            and si not in dropped
        ]
        # Named explicitly: it is derived from "Status" rather than renamed
        # from a header, so it never appears in column_renamings.
        if _MACCOR_MODE_COLUMN in data.columns:
            columns_keep.append(_MACCOR_MODE_COLUMN)
        data = self.standard_data_processing(data, columns_keep=columns_keep)
        data = self._restore_counter_dtypes(data)

        return data

    @staticmethod
    def _restore_counter_dtypes(data: pl.DataFrame) -> pl.DataFrame:
        """Cast the counter columns back to integers.

        ``standard_data_processing`` casts every numeric column to Float64
        except the two it names explicitly, so an end code or a loop count
        arrives here as ``17.0`` rather than ``17``. Registers are left as
        floats: a SetVar holds a real number, and Maccor's own default of 0 is
        a value rather than a placeholder.

        Parameters
        ----------
        data : pl.DataFrame
            Frame as returned by ``standard_data_processing``.

        Returns
        -------
        pl.DataFrame
            The same frame with the integer-valued counters cast to Int64.
            A blank ``ES`` stays null, which Int64 represents, so no reading
            is invented for a row the cycler left empty.
        """
        counters = [
            col
            for col in data.columns
            if col == "End status from cycler" or col.startswith("Loop ")
        ]
        if not counters:
            return data
        return data.with_columns(
            pl.col(col).cast(pl.Int64, strict=False) for col in counters
        )

    @classmethod
    def _convert_milli_units(cls, data: pl.DataFrame) -> pl.DataFrame:
        """Rescale milli-prefixed columns to SI units and drop the originals.

        Columns named e.g. ``Current [mA]`` or ``Capacity [mA.h]`` are divided
        by 1000 and renamed to ``Current [A]`` / ``Capacity [A.h]``, mirroring
        the conversion the Neware and Biologic readers do.

        Parameters
        ----------
        data : pl.DataFrame
            Frame whose columns have already been renamed to ionworks names,
            possibly including milli-unit intermediates.

        Returns
        -------
        pl.DataFrame
            Frame with milli-unit columns replaced by their SI equivalents.

        Raises
        ------
        ValueError
            If a milli-unit column and its SI equivalent are both present, since
            which scale the data is on would then be a guess.
        """
        for milli_col, si_col in _MACCOR_MILLI_UNIT_COLUMNS.items():
            if milli_col not in data.columns:
                continue
            if si_col in data.columns:
                raise ValueError(
                    f"Found both {milli_col!r} and {si_col!r} in the Maccor "
                    f"file, so the scale of the data is ambiguous. Pass "
                    f"extra_column_mappings to say which column to use."
                )
            data = cls._coerce_numeric(data, milli_col)
            data = data.with_columns((pl.col(milli_col) * 1e-3).alias(si_col)).drop(
                milli_col
            )
        return data

    @staticmethod
    def _parse_timestamp_column(data: pl.DataFrame) -> pl.DataFrame:
        """
        Parse Timestamp column and compute Time [s] if needed.

        Parameters
        ----------
        data : pl.DataFrame
            Input dataframe with potential "Timestamp" column.

        Returns
        -------
        pl.DataFrame
            Dataframe with parsed timestamps and computed Time [s] if applicable.
        """
        if "Timestamp" not in data.columns:
            return data

        # Parse datetime with multiple format attempts
        data = data.with_columns(
            pl.coalesce(
                # Try MM/DD/YYYY HH:MM:SS format (common for Maccor DPT)
                pl.col("Timestamp").str.strptime(
                    pl.Datetime, format="%m/%d/%Y %H:%M:%S", strict=False
                ),
                # Try YYYY-MM-DD HH:MM:SS format
                pl.col("Timestamp").str.strptime(
                    pl.Datetime, format="%Y-%m-%d %H:%M:%S", strict=False
                ),
                # Try MM/DD/YYYY HH:MM:SS AM/PM format
                pl.col("Timestamp").str.strptime(
                    pl.Datetime, format="%m/%d/%Y %I:%M:%S %p", strict=False
                ),
            )
            .dt.replace_time_zone("UTC")
            .alias("Timestamp")
        )

        # If we don't have a numeric "Time [s]" column, compute it from Timestamp
        if "Time [s]" not in data.columns:
            # Compute Time [s] from earliest Timestamp
            start_epoch = data.select(pl.col("Timestamp").dt.epoch("s").min()).item()
            data = data.with_columns(
                (pl.col("Timestamp").dt.epoch("s") - start_epoch).alias("Time [s]")
            )

        return data

    def _validate_and_fix_time(
        self, data: pl.DataFrame, time_offset_fix: float
    ) -> pl.DataFrame:
        """
        Validate that time is strictly increasing and optionally fix it.

        Parameters
        ----------
        data : pl.DataFrame
            Input dataframe with "Time [s]" column.
        time_offset_fix : float
            Minimum time difference to enforce when fixing.
            If -1, raises ValueError. If >= 0, ensures all time differences are at least this value.

        Returns
        -------
        pl.DataFrame
            Dataframe with validated or fixed time.

        Raises
        ------
        ValueError
            If time is not strictly increasing and time_offset_fix is -1.
        """
        if "Time [s]" not in data.columns:
            return data

        # Vectorized check: compute differences between consecutive times
        time_col = data["Time [s]"]
        time_diff = time_col.diff()  # time[i] - time[i-1]

        # Check if any difference is < 0 (decreasing, excluding first row which is null)
        # Note: duplicates (diff == 0) are allowed and will be handled by standard_data_processing
        has_decreasing = (time_diff.tail(-1) < 0).any()

        if not has_decreasing:
            return data

        if time_offset_fix == -1:
            # Find first problematic index for error message (only for decreasing times)
            bad_mask = time_diff < 0
            bad_indices = data.with_row_index().filter(bad_mask)["index"].to_list()
            i = bad_indices[0]
            time_values = time_col.to_list()

            raise ValueError(
                f"Time [s] must be strictly increasing. "
                f"Found Time[{i - 1}] = {time_values[i - 1]:.6f}s > Time[{i}] = {time_values[i]:.6f}s. "
                f"Set options['time_offset_fix'] to a positive offset (in seconds) to automatically fix."
            )

        # Apply offset fix: ensure all negative differences are at least time_offset_fix
        # Only fix decreasing times (negative diff), leave duplicates (zero diff) alone
        # Efficient vectorized approach using numpy

        time_values = time_col.to_numpy()

        # Compute differences between consecutive time points
        time_diff_np = np.diff(time_values)

        # Only fix negative differences (decreasing times), leave duplicates and positive diffs alone
        fixed_diff = np.where(time_diff_np < 0, time_offset_fix, time_diff_np)

        # Reconstruct time series: start at first point, then add cumulative fixed differences
        fixed_time = np.concatenate(
            [[time_values[0]], time_values[0] + np.cumsum(fixed_diff)]
        )

        return data.with_columns(pl.Series("Time [s]", fixed_time))

    def _fix_unsigned_current(self, data: pl.DataFrame) -> pl.DataFrame:
        """
        Fix unsigned current by flipping sign during charge if needed.

        If both "D" (discharge) and "C" (charge) are in the "Status" column
        and the current is always positive, then the current isn't signed,
        so we need to flip it during charge.

        Parameters
        ----------
        data : pl.DataFrame
            Input dataframe with potential "Status" and "Current [A]" columns.

        Returns
        -------
        pl.DataFrame
            Dataframe with current sign corrected if needed.
        """
        if "Status" not in data.columns or "Current [A]" not in data.columns:
            return data

        statuses = set(data.select(pl.col("Status").unique()).to_series().to_list())
        if "D" not in statuses or "C" not in statuses:
            return data

        # Ensure numeric current
        data = self._coerce_numeric(data, "Current [A]")

        c_min = (
            data.filter(pl.col("Status") == "C")
            .select(pl.col("Current [A]").min())
            .item()
        )
        d_min = (
            data.filter(pl.col("Status") == "D")
            .select(pl.col("Current [A]").min())
            .item()
        )

        # If both charge and discharge currents are positive, current is unsigned
        if c_min is not None and d_min is not None and c_min >= 0 and d_min >= 0:
            data = data.with_columns(
                pl.when(pl.col("Status") == "C")
                .then(-pl.col("Current [A]"))
                .otherwise(pl.col("Current [A]"))
                .alias("Current [A]")
            )

        return data

    @staticmethod
    def _get_column_renamings(options: dict[str, Any] | None = None) -> dict[str, str]:
        """
        Get standard column renaming mappings for Maccor files.

        Keys are canonical headers (see :func:`_canonical_header`), not raw
        ones; :meth:`_match_columns` canonicalizes each file header before
        looking it up here.

        Parameters
        ----------
        options : dict, optional
            Options dict. If options["skip_capacity_columns"] is True,
            capacity and energy column mappings are excluded, forcing
            ionworksdata to compute them from current/power integration.

        Returns
        -------
        dict[str, str]
            Dictionary mapping canonical column names to standardized names.
        """
        renamings = dict(_MACCOR_CANONICAL_MAP)
        if not (options and options.get("skip_capacity_columns", False)):
            renamings.update(_MACCOR_CANONICAL_CAPACITY_MAP)
        return renamings

    @staticmethod
    def _match_columns(
        columns: list[str], canonical_renamings: dict[str, str]
    ) -> dict[str, str]:
        """Resolve a canonical column map against one file's actual headers.

        Parameters
        ----------
        columns : list[str]
            Raw header strings as read from the file.
        canonical_renamings : dict[str, str]
            Map keyed on canonical headers, from :meth:`_get_column_renamings`.

        Returns
        -------
        dict[str, str]
            Map from the raw header as it appears in this file to its ionworks
            name, covering only columns the file actually has.

        Notes
        -----
        Two raw headers in one file can canonicalize together (``Test Time`` and
        ``Test-Time``); both are mapped, and
        :func:`ionworksdata.util.resolve_renamings` in :meth:`run` warns and
        keeps the first, so file order picks the winner.
        """
        return {
            col: canonical_renamings[key]
            for col in columns
            if (key := _canonical_header(col)) in canonical_renamings
        }

    @staticmethod
    def _resolve_cycle_column(
        data: pl.DataFrame, column_renamings: dict[str, str]
    ) -> dict[str, str]:
        """Map the correct Maccor cycle column to ``Cycle from cycler``.

        Maccor exports up to two cycle columns: ``Cycle C`` (the cumulative
        cycle counter, always the true cycle number) and ``Cycle P`` (a
        procedure-level loop counter whose meaning depends on how the Maccor
        procedure was written — often 0, sometimes equal to ``Cycle C``).
        ``Cycle C`` is therefore preferred whenever present. ``Cycle P`` is
        used only as a fallback when ``Cycle C`` is absent, so files that
        expose only ``Cycle P`` still get a ``Cycle from cycler`` column (see
        issue #2).

        Parameters
        ----------
        data : pl.DataFrame
            Raw data whose columns determine which cycle column to map.
        column_renamings : dict[str, str]
            Column renaming dictionary to update in place.

        Returns
        -------
        dict[str, str]
            The updated ``column_renamings`` dictionary.
        """
        if "Cycle C" in data.columns:
            column_renamings["Cycle C"] = "Cycle from cycler"
        elif "Cycle P" in data.columns:
            column_renamings["Cycle P"] = "Cycle from cycler"
        return column_renamings

    @staticmethod
    def _parse_excel_duration(duration_str: str) -> float | None:
        """
        Parse Excel duration format :D:HH:MM:SS to total seconds.

        Parameters
        ----------
        duration_str : str
            Duration string in format ":D:HH:MM:SS"

        Returns
        -------
        float | None
            Total seconds, or None if parsing fails.
        """
        if not duration_str.startswith(":"):
            return None
        parts = duration_str[1:].split(":")
        if len(parts) != 4:
            return None
        try:
            days, hours, minutes, seconds = map(int, parts)
            return float(days * 86400 + hours * 3600 + minutes * 60 + seconds)
        except (ValueError, TypeError):
            return None

    _COMPACT_DURATION_RE = re.compile(r"^\s*(\d+)d\s+(\d+):(\d+):(\d+(?:\.\d+)?)\s*$")

    @classmethod
    def _parse_compact_duration(cls, duration_str: str) -> float | None:
        """
        Parse compact short-form duration ``Nd HH:MM:SS[.fff]`` to total seconds.

        Used by Maccor's compact-header exports where ``TestTime`` / ``StepTime``
        contain values like ``  0d 00:00:5.01000022888184``. Leading whitespace
        and unpadded seconds are tolerated; fractional seconds are preserved.

        Parameters
        ----------
        duration_str : str
            Duration string in format ``Nd HH:MM:SS[.fff]`` (with optional
            leading whitespace).

        Returns
        -------
        float | None
            Total seconds, or None if parsing fails.
        """
        m = cls._COMPACT_DURATION_RE.match(duration_str)
        if m is None:
            return None
        try:
            days, hours, minutes = (int(g) for g in m.groups()[:3])
            seconds = float(m.group(4))
            return days * 86400 + hours * 3600 + minutes * 60 + seconds
        except (ValueError, TypeError):
            return None

    def _process_test_time_column(
        self, data: pl.DataFrame, column_renamings: dict[str, str]
    ) -> tuple[pl.DataFrame, dict[str, str]]:
        """
        Process ``Test Time`` / ``TestTime`` columns and determine their format.

        Handles four formats per column:

        1. Excel duration (``:D:HH:MM:SS``) -> converts to seconds, maps to ``Time [s]``
        2. Compact duration (``Nd HH:MM:SS[.fff]``) -> converts to seconds, maps to ``Time [s]``
        3. Datetime strings (contains ``/`` or ``-``) -> maps to ``Timestamp``
        4. Numeric values -> leaves as-is (assumed seconds)

        Both the verbose ``Test Time`` (with space) and the compact ``TestTime``
        (no space) variants are inspected; whichever is present is processed.

        Parameters
        ----------
        data : pl.DataFrame
            Input dataframe with potential ``Test Time`` / ``TestTime`` columns.
        column_renamings : dict[str, str]
            Column renaming dictionary to update.

        Returns
        -------
        tuple[pl.DataFrame, dict[str, str]]
            Updated dataframe and column_renamings dict.
        """
        candidates = [
            col
            for col in data.columns
            if _canonical_header(col) == _MACCOR_BARE_TEST_TIME
            or column_renamings.get(col) in _MACCOR_TIME_TARGETS
        ]
        for col in candidates:
            data, column_renamings = self._process_one_test_time_column(
                data, column_renamings, col
            )
        return data, column_renamings

    def _process_one_test_time_column(
        self,
        data: pl.DataFrame,
        column_renamings: dict[str, str],
        col: str,
    ) -> tuple[pl.DataFrame, dict[str, str]]:
        if col not in data.columns:
            return data, column_renamings

        sample = data.select(pl.col(col)).filter(pl.col(col).is_not_null()).head(1)
        if sample.height == 0:
            return data, column_renamings

        val = sample.item(0, 0)

        if isinstance(val, str) and val.startswith(":"):
            data = data.with_columns(
                pl.col(col)
                .map_elements(self._parse_excel_duration, return_dtype=pl.Float64)
                .alias(col)
            )
            column_renamings[col] = "Time [s]"
        elif isinstance(val, str) and self._COMPACT_DURATION_RE.match(val):
            data = data.with_columns(
                pl.col(col)
                .map_elements(self._parse_compact_duration, return_dtype=pl.Float64)
                .alias(col)
            )
            column_renamings[col] = "Time [s]"
        elif isinstance(val, str) and ("/" in val or "-" in val):
            column_renamings[col] = "Timestamp"
        # Otherwise treat as numeric time column (might already be in seconds)

        return data, column_renamings

    @staticmethod
    def has_time_column(text: str) -> bool:
        """Return True if *text* names a Maccor time column, case-insensitively.

        Parameters
        ----------
        text : str
            A header string, or a whole header line for the CSV/text path,
            where detection scans a metadata preamble for the real header.

        Returns
        -------
        bool
            True when any known Maccor time-column spelling appears.
        """
        lowered = text.lower()
        return any(t in lowered for t in _MACCOR_TIME_COLUMNS)

    @classmethod
    def sniff_excel(cls, filename: str | Path) -> bool:
        """Return True if *filename* is a Maccor ``.xls``/``.xlsx`` export.

        The data sheet is resolved first: a Maccor workbook need not put the
        time series on the first sheet, and detection must not be narrower
        than what the reader can read.

        Parameters
        ----------
        filename : str | Path
            Path to the workbook to inspect.

        Returns
        -------
        bool
            True when the resolved sheet carries a step column alongside a
            Maccor time column, or enough Maccor-shaped columns to be
            unambiguous.
        """
        try:
            _df, column_names = read_excel_and_get_column_names(
                Path(filename), sheet_name=cls._find_data_sheet(filename)
            )
        except Exception:
            return False
        has_step = any("step" in col for col in column_names)
        has_time = any(cls.has_time_column(col) for col in column_names)
        if has_step and has_time:
            return True
        matches = sum(
            1 for col in column_names if any(mc in col for mc in _MACCOR_COLUMNS)
        )
        return matches >= _MACCOR_MIN_SIGNATURE_COLUMNS

    @staticmethod
    def _find_data_sheet(filename: str | Path) -> str | None:
        """Return the worksheet holding the Maccor time series.

        Parameters
        ----------
        filename : str | Path
            Path to the Maccor ``.xls``/``.xlsx`` workbook.

        Returns
        -------
        str | None
            Name of the data sheet. Falls back to the first sheet when no
            sheet confirms, because a Maccor export may carry a metadata row
            above the real header, putting the column names out of reach of a
            one-row probe.
        """
        return find_data_sheet(
            filename,
            required_columns=_MACCOR_DATA_SHEET_REQUIRED,
            aliases=_MACCOR_HEADER_ALIASES,
            fallback_to_first=True,
        )

    def _read_excel_file(self, filename: str | Path, encoding: str) -> pl.DataFrame:
        """
        Read Maccor data from an Excel file (.xls or .xlsx).

        Parameters
        ----------
        filename : str | Path
            Path to the Excel file.
        encoding : str
            File encoding (not used for Excel but kept for consistency).

        Returns
        -------
        pl.DataFrame
            Raw data from Excel file with header row identified.
        """
        # Suppress pandas dtype warning when reading Excel (printed to stderr)
        with suppress_excel_dtype_warnings():
            data, _ = read_excel_and_get_column_names(
                filename, sheet_name=self._find_data_sheet(filename)
            )

        return data

    def read_header(
        self, filename: str | Path, options: dict[str, str] | None = None
    ) -> str:
        """
        Read the header from a Maccor file.
        """
        options = iwutil.check_and_combine_options(self.default_options, options)
        encoding, skiprows, _, _, _, is_excel = self._get_file_args(filename, options)

        if is_excel:
            # Suppress pandas dtype warning when reading Excel (printed to stderr)
            with suppress_excel_dtype_warnings():
                df_raw = pl.read_excel(
                    filename, sheet_name=self._find_data_sheet(filename)
                )

            # Return header row as string (column names are the header)
            return "\t".join(str(col) for col in df_raw.columns)
        else:
            with open(filename, encoding=encoding) as f:
                if len(skiprows) == 1:
                    # Header is single line
                    header_text = f.readline()
                else:
                    # Header is multiple lines
                    header_text = "".join(f.readline() for _ in skiprows)
            return header_text

    def read_start_time(
        self,
        filename: str | Path,
        extra_column_mappings: dict[str, str] | None = None,
        options: dict[str, str] | None = None,
    ) -> datetime | None:
        """
        Read the start time from a Maccor file.

        Parameters
        ----------
        filename : str | Path
            Path to the Maccor file to be read. Supports:
            - .txt files (tab-separated)
            - .csv files (comma-separated with units row)
            - .xls/.xlsx files (Excel format)
            - Files with .+3digits extension (e.g., .123, .456)
        options : dict of str to str, optional
            See :func:`ionworksdata.read.Maccor.run`.

        Returns
        -------
        datetime | None
            The start time of the Maccor file, or None if not found.
        """
        options = iwutil.check_and_combine_options(self.default_options, options)
        return self._parse_start_time(self.read_header(filename, options), options)

    @classmethod
    def _parse_start_time(
        cls, header_text: str, options: dict[str, str]
    ) -> datetime | None:
        """Shared by read_start_time and read_metadata, so the header's date is
        parsed one way rather than two."""
        header_text = _flatten_header(header_text)
        if "Date of Test:" not in header_text:
            return None
        # The date sits between "Date of Test:" and "Filename:".
        date_str = header_text.split("Date of Test:")[1].split("Filename:")[0].strip()

        start_datetime = None
        for fmt in ["%d %B %Y, %I:%M:%S %p", "%m/%d/%Y"]:
            try:
                if fmt == "%m/%d/%Y":
                    # A bare date: assume midnight. Possibly unsafe for two
                    # tests on the same day.
                    date_str = date_str + " 00:00:00"
                    fmt = "%m/%d/%Y %H:%M:%S"
                start_datetime = datetime.strptime(date_str, fmt)
                break
            except ValueError:
                continue
        if start_datetime is None:
            return None

        return _localize(start_datetime, options)

    def read_metadata(
        self,
        filename: str | Path,
        options: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """
        Read the descriptive metadata from a Maccor file header.

        The ``Procedure:`` name identifies the test schedule that produced the
        file, which is what links a measurement to its protocol on upload.

        Parameters
        ----------
        filename : str | Path
            Path to the Maccor file to be read. Supports the same formats as
            :meth:`read_start_time`.
        options : dict of str to str, optional
            See :func:`ionworksdata.read.Maccor.run`.

        Returns
        -------
        dict[str, Any]
            Header fields already grouped the way the measurement API defines
            them, so the result can be merged into a measurement dict as-is:

            - ``protocol["name"]``: name of the Maccor test schedule.
            - ``test_setup["export_date"]``: ``Today's Date``, when the file
              was exported.
            - ``test_setup["source_file_path"]``: path it was exported from.
            - ``test_setup["source_file_id"]``: id from a bracketed
              ``Comment/Barcode``. This belongs to the cycler's own id space
              and is **not** a cross-system identifier -- it does not
              correspond to a recipe, cell or sample id in any other system.
            - ``test_setup["barcode_comment"]``: free-text part of
              ``Comment/Barcode``.
            - ``test_setup["channel_number"]``: the per-row dialect's
              ``Channel``, else the numeric filename extension, which is how
              Maccor names an export after the channel that produced it
              (``...057`` is channel 57).
            - ``start_time``: timezone-aware, from ``Date of Test`` or, in the
              per-row dialect, ``Started``. Top-level, because the API models
              it as a column rather than inside a group. Same value
              :meth:`read_start_time` returns for the flat dialect, so a
              caller wanting both reads the header once.

            Keys the header does not contain are omitted, so an export with no
            ``Comment/Barcode`` yields no barcode keys rather than empty ones.

            The dialects carry different fields, so an absent key may mean the
            dialect does not write it rather than that the value is missing.
            ``export_date`` and ``source_file_path`` come from the flat header
            only. ``channel_number`` comes from the per-row header or, for
            either dialect, a numeric filename extension -- so it is absent
            only for a ``.txt``/``.csv``/``.xlsx`` name.
        """
        options = iwutil.check_and_combine_options(self.default_options, options)
        _, _, sep, _, _, is_excel = self._get_file_args(filename, options)
        header_text = self.read_header(filename, options)
        # Excel headers are column names: labels without values, so a
        # flat-line parse captures the whole joined row as one value.
        parsed = {} if is_excel else _parse_maccor_header(header_text, sep)
        # The flat dialect has no channel field, so the filename is the only
        # record of it. An explicit "Channel:" row wins where one exists.
        channel = _channel_from_extension(filename)
        if channel is not None:
            parsed.setdefault("channel_number", channel)
        # The row dialect states its own start time; the flat one carries
        # "Date of Test", which _parse_start_time reads.
        row_started = parsed.pop("started", None)
        # One header read: reuse the text already loaded rather than going back
        # through read_start_time, which would re-read the file.
        started = self._parse_start_time(header_text, options)
        if started is None and row_started is not None:
            started = _parse_row_start_time(row_started, options)
        if started is not None:
            parsed["start_time"] = started
        nested = _nest_metadata(parsed)
        # Not from the header, but the same contract as every other reader --
        # see :meth:`BaseReader.read_metadata`. Overriding skips that default.
        nested.setdefault("test_setup", {})["cycler"] = self.name
        return nested


def maccor(
    filename: str | Path,
    extra_column_mappings: dict[str, str] | None = None,
    options: dict[str, str] | None = None,
) -> pl.DataFrame:
    return Maccor().run(
        filename, extra_column_mappings=extra_column_mappings, options=options
    )
