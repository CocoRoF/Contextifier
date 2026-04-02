# tests/unit/handlers/test_zip_bomb_defense.py
"""P0-4: ZIP bomb defense — verify check_zip_bomb rejects oversized archives."""

from __future__ import annotations

import io
import zipfile

import pytest

from contextifier.pipeline.converter import check_zip_bomb, MAX_ZIP_DECOMPRESSED_BYTES
from contextifier.errors import ConversionError


def _make_zip(entries: dict[str, bytes]) -> bytes:
    """Create a minimal in-memory ZIP archive from name→data pairs."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return buf.getvalue()


class TestCheckZipBomb:
    def test_small_zip_passes(self) -> None:
        """A ZIP well within the size limit should not raise."""
        raw = _make_zip({"a.txt": b"hello", "b.txt": b"world"})
        with zipfile.ZipFile(io.BytesIO(raw), "r") as zf:
            check_zip_bomb(zf, handler="test")  # should not raise

    def test_oversized_zip_raises(self) -> None:
        """A ZIP exceeding the limit should raise ConversionError."""
        # Create a ZIP with a single entry that claims ~100 bytes,
        # then use a low threshold to trigger the check.
        raw = _make_zip({"big.bin": b"\x00" * 200})
        with zipfile.ZipFile(io.BytesIO(raw), "r") as zf:
            with pytest.raises(ConversionError, match="ZIP bomb detected"):
                check_zip_bomb(zf, max_bytes=100, handler="test")

    def test_multiple_entries_sum(self) -> None:
        """Total size is the sum of all entries, not just one."""
        raw = _make_zip({
            "a.bin": b"\x00" * 60,
            "b.bin": b"\x00" * 60,
        })
        # Each is 60; total 120 > 100
        with zipfile.ZipFile(io.BytesIO(raw), "r") as zf:
            with pytest.raises(ConversionError, match="ZIP bomb detected"):
                check_zip_bomb(zf, max_bytes=100, handler="test")

    def test_exact_limit_passes(self) -> None:
        """Total size exactly at the limit should not raise."""
        raw = _make_zip({"exact.bin": b"\x00" * 100})
        with zipfile.ZipFile(io.BytesIO(raw), "r") as zf:
            check_zip_bomb(zf, max_bytes=100, handler="test")  # should not raise


class TestConverterValidateZipBomb:
    """Verify that each OOXML converter's validate() rejects ZIP bombs."""

    def _make_ooxml_zip(self, entries: dict[str, bytes]) -> bytes:
        """Create a ZIP with [Content_Types].xml + extra entries."""
        all_entries = {"[Content_Types].xml": b"<Types></Types>"}
        all_entries.update(entries)
        return _make_zip(all_entries)

    def test_docx_validate_rejects_bomb(self) -> None:
        from contextifier.handlers.docx.converter import DocxConverter
        import contextifier.pipeline.converter as conv_mod

        original = conv_mod.MAX_ZIP_DECOMPRESSED_BYTES
        try:
            conv_mod.MAX_ZIP_DECOMPRESSED_BYTES = 50  # 50 bytes threshold
            raw = self._make_ooxml_zip({"word/document.xml": b"\x00" * 200})
            ctx = {"file_data": raw, "file_extension": "docx"}
            converter = DocxConverter()
            assert converter.validate(ctx) is False
        finally:
            conv_mod.MAX_ZIP_DECOMPRESSED_BYTES = original

    def test_pptx_validate_rejects_bomb(self) -> None:
        from contextifier.handlers.pptx.converter import PptxConverter
        import contextifier.pipeline.converter as conv_mod

        original = conv_mod.MAX_ZIP_DECOMPRESSED_BYTES
        try:
            conv_mod.MAX_ZIP_DECOMPRESSED_BYTES = 50
            raw = self._make_ooxml_zip({"ppt/presentation.xml": b"\x00" * 200})
            ctx = {"file_data": raw, "file_extension": "pptx"}
            converter = PptxConverter()
            assert converter.validate(ctx) is False
        finally:
            conv_mod.MAX_ZIP_DECOMPRESSED_BYTES = original

    def test_xlsx_validate_rejects_bomb(self) -> None:
        from contextifier.handlers.xlsx.converter import XlsxConverter
        import contextifier.pipeline.converter as conv_mod

        original = conv_mod.MAX_ZIP_DECOMPRESSED_BYTES
        try:
            conv_mod.MAX_ZIP_DECOMPRESSED_BYTES = 50
            raw = self._make_ooxml_zip({"xl/workbook.xml": b"\x00" * 200})
            ctx = {"file_data": raw, "file_extension": "xlsx"}
            converter = XlsxConverter()
            assert converter.validate(ctx) is False
        finally:
            conv_mod.MAX_ZIP_DECOMPRESSED_BYTES = original

    def test_hwpx_validate_rejects_bomb(self) -> None:
        from contextifier.handlers.hwpx.converter import HwpxConverter
        import contextifier.pipeline.converter as conv_mod

        original = conv_mod.MAX_ZIP_DECOMPRESSED_BYTES
        try:
            conv_mod.MAX_ZIP_DECOMPRESSED_BYTES = 50
            raw = _make_zip({"Contents/content.hpf": b"\x00" * 200})
            ctx = {"file_data": raw, "file_extension": "hwpx"}
            converter = HwpxConverter()
            assert converter.validate(ctx) is False
        finally:
            conv_mod.MAX_ZIP_DECOMPRESSED_BYTES = original
