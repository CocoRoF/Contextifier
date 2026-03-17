"""
PPT handler constants.

Contains:
- OLE2 magic signature for genuine PPT detection
- ZIP magic for misnamed PPTX detection
- PowerPoint-specific OLE stream names
"""

from __future__ import annotations


# ═══════════════════════════════════════════════════════════════════════════════
# Magic bytes
# ═══════════════════════════════════════════════════════════════════════════════

OLE2_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
ZIP_MAGIC = b"PK\x03\x04"

# ═══════════════════════════════════════════════════════════════════════════════
# OLE2 PowerPoint stream names
# ═══════════════════════════════════════════════════════════════════════════════

# Main PowerPoint data stream
PP_DOCUMENT_STREAM = "PowerPoint Document"
# Current user stream (entry point)
CURRENT_USER_STREAM = "Current User"
# Summary info (metadata)
SUMMARY_INFO_STREAM = "\x05SummaryInformation"
DOC_SUMMARY_STREAM = "\x05DocumentSummaryInformation"

# Known OLE streams in PPT files
PPT_KNOWN_STREAMS = frozenset({
    PP_DOCUMENT_STREAM,
    CURRENT_USER_STREAM,
    SUMMARY_INFO_STREAM,
    DOC_SUMMARY_STREAM,
})


__all__ = [
    "OLE2_MAGIC",
    "ZIP_MAGIC",
    "PP_DOCUMENT_STREAM",
    "CURRENT_USER_STREAM",
    "SUMMARY_INFO_STREAM",
    "DOC_SUMMARY_STREAM",
    "PPT_KNOWN_STREAMS",
]
