# contextifier/services/libreoffice.py
"""
LibreOffice headless conversion helper.

Converts legacy binary formats (.doc, .ppt, .xls, .hwp) to their
modern OOXML counterparts (.docx, .pptx, .xlsx) so that the standard
handlers can process them with full fidelity.

Usage::

    from contextifier.services.libreoffice import convert_with_libreoffice

    output_path = convert_with_libreoffice(
        "legacy.doc",
        output_format="docx",
    )

Requirements:
    LibreOffice must be installed and ``soffice`` available on ``PATH``
    (or provide the full path via the ``soffice_path`` parameter).

This module does **not** add a mandatory dependency on LibreOffice.
Handlers that wish to use it should catch ``LibreOfficeNotFoundError``
and fall back to their native binary parsers.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Union

from contextifier.errors import ContextifierError

logger = logging.getLogger("contextifier.services.libreoffice")


class LibreOfficeNotFoundError(ContextifierError):
    """Raised when the soffice binary cannot be located."""


class LibreOfficeConversionError(ContextifierError):
    """Raised when a LibreOffice conversion fails."""


# Mapping from legacy extension → OOXML target format string
_FORMAT_MAP: dict[str, str] = {
    "doc": "docx",
    "ppt": "pptx",
    "xls": "xlsx",
    "hwp": "docx",  # best-effort; depends on LO HWP filter
}


def find_soffice() -> Optional[str]:
    """Locate the ``soffice`` binary on the system ``PATH``."""
    return shutil.which("soffice") or shutil.which("libreoffice")


def convert_with_libreoffice(
    input_path: Union[str, Path],
    *,
    output_format: Optional[str] = None,
    output_dir: Optional[Union[str, Path]] = None,
    soffice_path: Optional[str] = None,
    timeout: float = 120,
) -> Path:
    """Convert a document via LibreOffice headless mode.

    Args:
        input_path: Path to the source file.
        output_format: Target format (e.g. ``"docx"``, ``"pptx"``).
            If ``None``, inferred from ``_FORMAT_MAP`` using the
            input file's extension.
        output_dir: Directory for the converted file.  Defaults to
            a temporary directory that the caller should clean up.
        soffice_path: Explicit path to ``soffice``.  Searched on
            ``PATH`` if not provided.
        timeout: Maximum seconds to wait for the conversion.

    Returns:
        :class:`Path` to the converted file.

    Raises:
        LibreOfficeNotFoundError: ``soffice`` not found.
        LibreOfficeConversionError: Conversion process failed.
    """
    input_path = Path(input_path).resolve()
    if not input_path.is_file():
        raise LibreOfficeConversionError(
            f"Input file not found: {input_path}",
            context={"input_path": str(input_path)},
        )

    soffice = soffice_path or find_soffice()
    if soffice is None:
        raise LibreOfficeNotFoundError(
            "LibreOffice (soffice) not found on PATH. "
            "Install LibreOffice or provide soffice_path.",
        )

    ext = input_path.suffix.lower().lstrip(".")
    fmt = output_format or _FORMAT_MAP.get(ext)
    if fmt is None:
        raise LibreOfficeConversionError(
            f"No default output format for '.{ext}'. Provide output_format explicitly.",
            context={"extension": ext},
        )

    out_dir = Path(output_dir) if output_dir else Path(tempfile.mkdtemp(prefix="ctxify_lo_"))
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        soffice,
        "--headless",
        "--norestore",
        "--convert-to", fmt,
        "--outdir", str(out_dir),
        str(input_path),
    ]

    logger.info("Running LibreOffice conversion: %s → %s", input_path.name, fmt)
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        raise LibreOfficeConversionError(
            f"LibreOffice conversion timed out after {timeout}s",
            context={"input": str(input_path), "format": fmt},
        )

    if proc.returncode != 0:
        raise LibreOfficeConversionError(
            f"LibreOffice exited with code {proc.returncode}: {proc.stderr.strip()[:500]}",
            context={"input": str(input_path), "format": fmt, "stderr": proc.stderr[:500]},
        )

    expected = out_dir / f"{input_path.stem}.{fmt}"
    if not expected.is_file():
        # LibreOffice sometimes uses a different extension
        candidates = list(out_dir.glob(f"{input_path.stem}.*"))
        if candidates:
            expected = candidates[0]
        else:
            raise LibreOfficeConversionError(
                f"Converted file not found at expected path: {expected}",
                context={"expected": str(expected), "outdir_contents": [p.name for p in out_dir.iterdir()]},
            )

    logger.info("Conversion complete: %s", expected)
    return expected


__all__ = [
    "convert_with_libreoffice",
    "find_soffice",
    "LibreOfficeNotFoundError",
    "LibreOfficeConversionError",
]
