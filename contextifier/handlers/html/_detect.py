# contextifier/handlers/html/_detect.py
"""
Content sniffing for files that are HTML under another name.

Report portals routinely serve an HTML table with a ``.xls`` or ``.xlsx``
filename. Excel renders them, so people treat them as spreadsheets, but xlrd
and openpyxl reject them outright and the document fails to convert.

The check is a signature test, not a guess: a real XLS begins with the OLE
compound-file magic and a real XLSX with the ZIP local-file header, so neither
can begin with markup.
"""

from __future__ import annotations

_SNIFF_BYTES = 1024
_BOMS = (b"\xef\xbb\xbf", b"\xff\xfe", b"\xfe\xff")

# Openers that only an HTML document starts with.
_HTML_OPENERS = (b"<html", b"<!doctype html", b"<table", b"<body", b"<meta")


def looks_like_html(data: bytes) -> bool:
    """
    Report whether *data* is an HTML document.

    Args:
        data: The leading bytes of the file (the whole file is fine).

    Returns:
        True when the content opens as HTML.
    """
    if not data or len(data) < 16:
        return False

    head = data[:_SNIFF_BYTES]
    for bom in _BOMS:
        if head.startswith(bom):
            head = head[len(bom) :]
            break
    head = head.lstrip().lower()

    if head.startswith(_HTML_OPENERS):
        return True

    # An XML declaration is ambiguous: Excel's own SpreadsheetML format also
    # opens with `<?xml` but is a genuine workbook (it declares `<Workbook>`).
    # Only treat it as HTML when an <html> element actually follows.
    if head.startswith(b"<?xml"):
        return b"<html" in head

    return False


__all__ = ["looks_like_html"]
