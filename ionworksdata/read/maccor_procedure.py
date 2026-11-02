"""Extract the procedure embedded in a Maccor raw data file.

The cycler writes each sub-procedure's XML into the file only when it first
runs, so a copy taken mid-test lacks the ones that have not started yet.
"""

from __future__ import annotations

from dataclasses import dataclass
import mmap
from pathlib import Path
import re
import warnings
import xml.etree.ElementTree as ET

__all__ = ["MaccorProcedure", "maccor_procedure"]

_XML_START = b"<?xml"
_XML_END = b"</MaccorTestProcedure>"
_PROCEDURE_ROOT = b"<MaccorTestProcedure>"

#: Ends the label written before a sub-procedure's XML; the name precedes it.
_SUB_LABEL_MARKER = b".000Sub routine running"

#: The longest name read back from a label. The run of printable bytes before the
#: marker can include bytes of the record before it, so it is bounded.
_MAX_NAME_BYTES = 255

#: The master procedure's filename in the file header, e.g. ``Formation.000``.
_HEADER_NAME_RE = re.compile(rb"(?<![\x20-\xff])([\x20-\xff]{1,255}?)\.000")

#: Maccor procedure files carry this extension; references may include it.
_PROCEDURE_EXTENSION = ".000"


@dataclass(frozen=True)
class MaccorProcedure:
    """A test procedure recovered from a Maccor raw data file.

    ``sub_procedures`` has one entry for every sub-procedure the procedure calls,
    keyed by the name exactly as the calling step writes it, which is the key
    ``ionworks_ucp.parse_protocol`` expects in ``additional_content``. The value
    is the sub-procedure's XML, or None when the file does not contain it yet.
    """

    #: The master procedure's filename without ``.000``, or None when the file
    #: header does not name it.
    name: str | None
    #: The master procedure's XML.
    master: str
    #: Every sub-procedure called, master's call order first, mapped to its XML
    #: or to None when it has not been written to the file yet.
    sub_procedures: dict[str, str | None]

    @property
    def missing(self) -> list[str]:
        """Names of the sub-procedures the file does not contain yet."""
        return [name for name, xml in self.sub_procedures.items() if xml is None]

    @property
    def is_complete(self) -> bool:
        """Whether the file contains every sub-procedure the procedure calls."""
        return not self.missing


def maccor_procedure(
    source: str | Path | bytes | bytearray | memoryview,
) -> MaccorProcedure:
    """Extract the procedure embedded in a Maccor raw data file.

    Parameters
    ----------
    source : str, Path, bytes, bytearray or memoryview
        Path to a Maccor raw (binary) data file, or its contents. Text exports do
        not carry the procedure.

    Returns
    -------
    MaccorProcedure
        The master procedure's name and XML, and each sub-procedure it calls with
        its XML, or None for those the file does not contain yet.

    Raises
    ------
    ValueError
        If the file does not contain an embedded Maccor procedure, or one it
        contains is not valid XML.

    Examples
    --------
    >>> procedure = iwdata.read.maccor_procedure("TEST.001")  # doctest: +SKIP
    >>> procedure.missing  # doctest: +SKIP
    ['Discharge Rate']
    """
    if isinstance(source, (bytes, bytearray, memoryview)):
        return _extract(bytes(source))
    with open(source, "rb") as f:
        if Path(source).stat().st_size == 0:
            raise ValueError("File is empty; it holds no Maccor procedure.")
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as data:
            return _extract(data)


def _extract(data: bytes | mmap.mmap) -> MaccorProcedure:
    blocks = _xml_blocks(data)
    if not blocks:
        raise ValueError(
            "No embedded Maccor procedure found. Only Maccor raw data files carry "
            "their procedure; text exports do not."
        )
    first_start, first_end, _ = blocks[0]
    master = _decode(data[first_start:first_end])
    name = _header_name(data[:first_start])

    labelled: list[tuple[str, str]] = []
    previous_end = first_end
    for start, end, floor in blocks[1:]:
        xml = _decode(data[start:end])
        label = _label_between(
            data, floor if floor is not None else previous_end, start
        )
        previous_end = end
        if label is None:
            # Unlabelled: the master, written again before an activation.
            if xml != master:
                warnings.warn(
                    "The file contains a second, different master procedure; "
                    "keeping the first.",
                    stacklevel=3,
                )
            continue
        labelled.append((label, xml))

    sub_procedures = _resolve_sub_procedures(master, labelled)
    return MaccorProcedure(name=name, master=master, sub_procedures=sub_procedures)


def _xml_blocks(data: bytes | mmap.mmap) -> list[tuple[int, int, int | None]]:
    """Return ``(start, end, label_floor)`` for each embedded procedure's XML.

    A write cut off mid-way and followed by another leaves two ``<?xml`` before
    one end tag. The block is the last of them; ``label_floor`` is where the
    cut-off copy starts, so its label is not taken for the block's.
    """
    blocks = []
    position = 0
    while (start := data.find(_XML_START, position)) != -1:
        end = data.find(_XML_END, start)
        if end == -1:
            # The copy was taken mid-write; the tail holds no whole procedure.
            break
        floor = None
        restart = data.rfind(_XML_START, start + 1, end)
        if restart != -1:
            floor = data.rfind(_XML_START, start, restart)
            start = restart
        end += len(_XML_END)
        if data.find(_PROCEDURE_ROOT, start, end) != -1:
            blocks.append((start, end, floor))
        position = end
    return blocks


def _label_between(data: bytes | mmap.mmap, low: int, high: int) -> str | None:
    """The name in the last sub-procedure label between two offsets, or None.

    Finds the fixed marker, then reads back over at most ``_MAX_NAME_BYTES``
    printable bytes, so a long printable run costs one pass, not one per byte.
    """
    marker = data.rfind(_SUB_LABEL_MARKER, low, high)
    if marker == -1:
        return None
    start = marker
    floor = max(low, marker - _MAX_NAME_BYTES)
    while start > floor and data[start - 1] >= 0x20:
        start -= 1
    return bytes(data[start:marker]).decode("latin-1")


def _resolve_sub_procedures(
    master: str, labelled: list[tuple[str, str]]
) -> dict[str, str | None]:
    """Map every sub-procedure the procedure calls to its XML, if found.

    Each label is resolved to the called name it ends with, the longest when
    several do (``#7PreFormation`` is ``PreFormation``, not ``Formation``), and
    only then are copies compared. Sub-procedures can call sub-procedures, so
    the calls of each one found are followed until no new name appears.
    """
    names = list(dict.fromkeys(_called_sub_procedures(master)))
    labelled = _readable(labelled)
    while True:
        # One file, many spellings (``Formation.000``, ``formation``; Windows).
        stems: dict[str, list[str]] = {}
        for name in names:
            stems.setdefault(_stem(name).casefold(), []).append(name)
        by_stem: dict[str, str] = {}
        conflicts: set[str] = set()
        for label, xml in labelled:
            stem = _owner(_stem(label).casefold(), stems)
            if stem is None:
                continue
            if stem in by_stem:
                if by_stem[stem] != xml:
                    conflicts.add(stems[stem][0])
                continue
            by_stem[stem] = xml
        by_name = {name: xml for stem, xml in by_stem.items() for name in stems[stem]}
        nested = [
            called
            for xml in by_name.values()
            for called in _called_sub_procedures(xml)
            if called not in names
        ]
        if not nested:
            break
        names.extend(dict.fromkeys(nested))
    for name in sorted(conflicts):
        warnings.warn(
            f"Sub-procedure '{name}' is written twice with different contents; "
            "keeping the first.",
            stacklevel=4,
        )
    return {name: by_name.get(name) for name in names}


def _readable(labelled: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Drop sub-procedure copies that are not valid XML, with a warning.

    One damaged copy must not cost the master and every other sub-procedure; its
    sub-procedure is reported missing instead.
    """
    kept = []
    for label, xml in labelled:
        try:
            ET.fromstring(xml)
        except ET.ParseError as e:
            warnings.warn(
                f"Sub-procedure '{_stem(label)}' is not valid XML ({e}); "
                "treating it as not recorded.",
                stacklevel=5,
            )
            continue
        kept.append((label, xml))
    return kept


def _called_sub_procedures(xml: str) -> list[str]:
    """Names of the sub-procedures a procedure's steps call, in step order."""
    try:
        root = ET.fromstring(xml)
    except ET.ParseError as e:
        raise ValueError(f"Embedded Maccor procedure is not valid XML: {e}") from e
    calls = (
        step.findtext("StepValue", "").strip()
        for step in root.iter("TestStep")
        if step.findtext("StepType", "").strip() == "SubRout"
    )
    # A call with no name names nothing that could ever be filled.
    return [name for name in calls if name]


def _owner(label: str, stems: dict[str, list[str]]) -> str | None:
    """The called stem a label stands for: exact, else the longest it ends with."""
    if label in stems:
        return label
    endings = [stem for stem in stems if label.endswith(stem)]
    return max(endings, key=len) if endings else None


def _stem(name: str) -> str:
    """A procedure reference without any directory or ``.000`` extension."""
    base = name.replace("\\", "/").rsplit("/", 1)[-1]
    if base.lower().endswith(_PROCEDURE_EXTENSION):
        base = base[: -len(_PROCEDURE_EXTENSION)]
    return base.strip()


def _header_name(header: bytes) -> str | None:
    match = _HEADER_NAME_RE.search(header)
    return _stem(match.group(1).decode("latin-1")) if match else None


def _decode(xml: bytes) -> str:
    try:
        return xml.decode("utf-8")
    except UnicodeDecodeError:
        return xml.decode("latin-1")
