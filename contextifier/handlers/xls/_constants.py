# contextifier/handlers/xls/_constants.py
"""Constants for XLS (BIFF) processing."""

from __future__ import annotations

# ── Magic bytes ──────────────────────────────────────────────────────────────
OLE2_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
ZIP_MAGIC = b"PK\x03\x04"

# ── Scan limits (same as XLSX) ───────────────────────────────────────────────
# Guard against a corrupt dimension record, not a working budget: xlrd's
# nrows/ncols are exact, so capping them at a small number silently truncated
# every sheet past the cap.
SCAN_ROW_LIMIT = 200_000
SCAN_COL_LIMIT = 16_384

# Smallest shape worth rendering as a table (see xlsx/_constants.py).
MIN_TABLE_ROWS = 2
MIN_TABLE_COLS = 2

__all__ = [
    "OLE2_MAGIC",
    "ZIP_MAGIC",
    "SCAN_ROW_LIMIT",
    "SCAN_COL_LIMIT",
    "MIN_TABLE_ROWS",
    "MIN_TABLE_COLS",
]
