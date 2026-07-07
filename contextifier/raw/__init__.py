# contextifier/raw
"""
Raw document access — the lossless twin of the extraction pipeline.

Contextifier has two ways to look at a document:

* ``DocumentProcessor.extract_text()`` / ``.process()`` — the existing
  pipeline that renders an **AI-friendly** view (clean text, normalized
  tables/charts) and throws the rest away.
* ``open_raw()`` (this package) — a **lossless, addressable, writable**
  view of the same file. Nothing is discarded: every OPC part stays
  available, XML is parsed lazily, and edits are *surgical* — when you
  save, untouched parts are written back **byte-identical** (the
  byte-preservation contract), so charts, pivot tables, sparklines,
  custom XML, styles and anything else the higher-level libraries can't
  model all survive.

Usage::

    from contextifier import open_raw

    raw = open_raw("report.xlsx")          # XlsxRawDocument
    raw.sheets["Sales"].set_cell("B3", 142)
    raw.charts[0].set_data(categories=["Q1", "Q2"], series=[("Sales", [1, 2])])
    raw.save("report-edited.xlsx")         # or raw.to_bytes()

    raw = open_raw("deck.pptx")            # PptxRawDocument
    raw = open_raw("paper.docx")           # DocxRawDocument

Every format model also exposes ``.package`` (the raw
:class:`~contextifier.raw.opc.OpcPackage`) for part-level work, so the
"easy" interface never locks you out of the full container.

Supported today: the OOXML trio (.xlsx / .docx / .pptx). Other handlers
raise :class:`RawUnsupportedError`.
"""

from __future__ import annotations

from contextifier.raw.opc import OpcPackage, OpcPart, RawUnsupportedError
from contextifier.raw.xmlpart import NS, XmlPart, qn

__all__ = [
    "OpcPackage",
    "OpcPart",
    "RawUnsupportedError",
    "XmlPart",
    "NS",
    "qn",
    "open_raw",
]


def open_raw(source, *, extension: str | None = None):
    """Open a document for lossless, writable access.

    Args:
        source: path (str/Path), bytes, or a binary file object.
        extension: override the format sniff (e.g. ``"xlsx"``); by default
            the file extension (for paths) or the package content is used.

    Returns:
        ``XlsxRawDocument`` / ``DocxRawDocument`` / ``PptxRawDocument``.

    Raises:
        RawUnsupportedError: format has no raw model yet.
    """
    from pathlib import Path

    ext = (extension or "").lower().lstrip(".")
    if not ext and isinstance(source, (str, Path)):
        ext = Path(source).suffix.lower().lstrip(".")

    package = OpcPackage.open(source)
    if not ext:
        ext = package.sniff_format() or ""

    if ext == "xlsx":
        from contextifier.raw.xlsx import XlsxRawDocument

        return XlsxRawDocument(package)
    if ext == "docx":
        from contextifier.raw.docx import DocxRawDocument

        return DocxRawDocument(package)
    if ext == "pptx":
        from contextifier.raw.pptx import PptxRawDocument

        return PptxRawDocument(package)
    raise RawUnsupportedError(
        f"No raw model for {ext or 'unknown format'!r} yet "
        "(supported: xlsx, docx, pptx)"
    )
