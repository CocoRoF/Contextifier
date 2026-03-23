# contextifier/handlers/xls/_constants.py
"""Constants for XLS (BIFF) processing."""

from __future__ import annotations

# ── Magic bytes ──────────────────────────────────────────────────────────────
OLE2_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
ZIP_MAGIC = b"PK\x03\x04"

# ── Scan limits (same as XLSX) ───────────────────────────────────────────────
MAX_SCAN_ROWS = 1000
MAX_SCAN_COLS = 100

__all__ = [
    "OLE2_MAGIC",
    "ZIP_MAGIC",
    "MAX_SCAN_ROWS",
    "MAX_SCAN_COLS",
]
