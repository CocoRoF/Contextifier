# contextifier_new/handlers/pdf/_constants.py
"""
Shared constants and configuration for both PDF handler modes.

Both pdf_default and pdf_plus share:
- PDF magic bytes
- PDF date parsing
- Common config values
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Optional

# ── Magic bytes ──────────────────────────────────────────────────────────────

PDF_MAGIC = b"%PDF"

# ── PDF Date Parsing ─────────────────────────────────────────────────────────

_PDF_DATE_RE = re.compile(
    r"D:(\d{4})(\d{2})?(\d{2})?(\d{2})?(\d{2})?(\d{2})?"
)


def parse_pdf_date(date_str: Optional[str]) -> Optional[datetime]:
    """
    Parse a PDF date string  ``D:YYYYMMDDHHmmSS``  into a datetime.

    Returns None if *date_str* is None or unparseable.
    """
    if not date_str:
        return None
    m = _PDF_DATE_RE.match(date_str)
    if not m:
        return None
    try:
        parts = [int(g) if g else d for g, d in zip(
            m.groups(), (0, 1, 1, 0, 0, 0),
        )]
        return datetime(*parts)  # type: ignore[arg-type]
    except (ValueError, TypeError):
        return None


# ── PDF mode names ───────────────────────────────────────────────────────────

PDF_MODE_DEFAULT = "default"
PDF_MODE_PLUS = "plus"
PDF_FORMAT_OPTION_KEY = "pdf"
PDF_MODE_OPTION = "mode"


__all__ = [
    "PDF_MAGIC",
    "parse_pdf_date",
    "PDF_MODE_DEFAULT",
    "PDF_MODE_PLUS",
    "PDF_FORMAT_OPTION_KEY",
    "PDF_MODE_OPTION",
]
