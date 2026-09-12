# tests/unit/handlers/test_pdf_text_quality.py
"""
Text-layer quality assessment and the fallbacks it drives.

Three things are pinned here:

* garbled pages are recognised from what is *in* the text, not only from its
  absence — a page whose font mapping produced private-use or squared-unit
  characters still returns plenty of characters, and used to sail through;
* a page whose glyphs lost their line grouping is rebuilt from their
  positions rather than re-read;
* neither fallback fires on a page that reads correctly.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from contextifier.handlers.pdf_plus._text_quality_analyzer import (
    FragmentedTextReconstructor,
    TextQualityAnalyzer,
    is_fragmented_text,
)


class FakePage:
    """Stand-in exposing the two ``get_text`` shapes the analyser uses."""

    def __init__(self, text: str = "", rawdict: Dict[str, Any] | None = None) -> None:
        self._text = text
        self._rawdict = rawdict or {"blocks": []}

    def get_text(self, kind: str = "text", **_: Any):
        return self._rawdict if kind == "rawdict" else self._text


def _rawdict(chars: List[tuple]) -> Dict[str, Any]:
    """Build a rawdict from ``(x0, y0, x1, y1, character)`` tuples."""
    return {
        "blocks": [
            {
                "type": 0,
                "lines": [
                    {
                        "spans": [
                            {
                                "chars": [
                                    {"bbox": (x0, y0, x1, y1), "c": c}
                                    for x0, y0, x1, y1, c in chars
                                ]
                            }
                        ]
                    }
                ],
            }
        ]
    }


class TestFragmentDetection:
    def test_one_character_per_line_is_fragmented(self) -> None:
        assert is_fragmented_text("현\n재\n시\n장\n에\n대\n한") is True

    def test_a_bulleted_list_is_not_fragmented(self) -> None:
        """Short lines are normal; only near-empty ones are suspicious."""
        text = "\n".join(
            ["- 매출 증가", "- 비용 절감", "- 인력 확충", "- 신규 진출", "- 품질 개선"]
        )
        assert is_fragmented_text(text) is False

    def test_a_table_of_contents_is_not_fragmented(self) -> None:
        text = "\n".join(f"{i}. Chapter {i}" for i in range(1, 12))
        assert is_fragmented_text(text) is False

    def test_a_column_of_numbers_is_not_fragmented(self) -> None:
        text = "\n".join(str(1000 + i) for i in range(10))
        assert is_fragmented_text(text) is False

    def test_too_few_lines_to_judge(self) -> None:
        assert is_fragmented_text("a\nb\nc") is False


class TestQualitySignals:
    def test_clean_page_needs_no_fallback(self) -> None:
        page = FakePage("A perfectly ordinary paragraph of readable text.\n")
        result = TextQualityAnalyzer(page).analyze()

        assert result.needs_ocr is False
        assert result.cjk_compat_chars == 0
        assert result.is_fragmented is False

    def test_squared_unit_characters_standing_in_for_punctuation(self) -> None:
        """A broken font export turns brackets into CJK Compatibility symbols."""
        page = FakePage("보고서㏙2026㏚ 매출㏊3㏙4㏚ 분기㏙5㏚\n" * 3)
        result = TextQualityAnalyzer(page).analyze()

        assert result.cjk_compat_chars > 0
        assert result.needs_ocr is True

    def test_occasional_unit_symbol_is_not_garbled(self) -> None:
        """㎏ and ㎞ are ordinary in Korean technical writing."""
        page = FakePage(
            "본 장비의 중량은 45㎏ 이며 최대 주행 거리는 300㎞ 입니다. "
            "설치 면적은 12㎡ 로 산정되었고 소음은 55㏈ 수준입니다. "
            "추가 사양은 부록을 참조하십시오.\n"
        )
        result = TextQualityAnalyzer(page).analyze()

        assert result.cjk_compat_chars > 0
        assert result.needs_ocr is False, (
            "unit symbols at normal density must not trigger a re-read"
        )

    def test_private_use_characters_are_garbled(self) -> None:
        page = FakePage(" text \n" * 3)
        result = TextQualityAnalyzer(page).analyze()
        assert result.needs_ocr is True


class TestReconstruction:
    def test_glyphs_regroup_into_lines(self) -> None:
        chars = []
        for i, ch in enumerate("현재 시장"):
            chars.append((10 + i * 8, 100, 18 + i * 8, 112, ch))
        for i, ch in enumerate("분석 결과"):
            chars.append((10 + i * 8, 130, 18 + i * 8, 142, ch))

        page = FakePage(rawdict=_rawdict(chars))
        out = FragmentedTextReconstructor(page).reconstruct()

        assert out.split("\n") == ["현재 시장", "분석 결과"]

    def test_glyphs_are_ordered_by_position_not_by_stream_order(self) -> None:
        chars = [
            (30, 100, 38, 112, "C"),
            (10, 100, 18, 112, "A"),
            (20, 100, 28, 112, "B"),
        ]
        page = FakePage(rawdict=_rawdict(chars))
        assert FragmentedTextReconstructor(page).reconstruct() == "ABC"

    def test_sub_point_jitter_stays_on_one_line(self) -> None:
        """Producers place glyphs with small vertical wobble; tolerance absorbs it."""
        chars = [
            (10, 100.0, 18, 112.0, "l"),
            (20, 101.2, 28, 113.2, "i"),
            (30, 99.4, 38, 111.4, "t"),
        ]
        page = FakePage(rawdict=_rawdict(chars))
        assert FragmentedTextReconstructor(page).reconstruct() == "lit"

    def test_separate_lines_stay_separate(self) -> None:
        chars = [
            (10, 100, 18, 112, "a"),
            (10, 140, 18, 152, "b"),
        ]
        page = FakePage(rawdict=_rawdict(chars))
        assert FragmentedTextReconstructor(page).reconstruct() == "a\nb"

    def test_excluded_regions_are_skipped(self) -> None:
        chars = [
            (10, 100, 18, 112, "A"),
            (200, 300, 208, 312, "X"),  # inside the excluded box
        ]
        page = FakePage(rawdict=_rawdict(chars))
        out = FragmentedTextReconstructor(
            page, exclude_bboxes=[(190, 290, 260, 340)]
        ).reconstruct()

        assert out == "A"

    def test_empty_page_returns_empty(self) -> None:
        assert FragmentedTextReconstructor(FakePage()).reconstruct() == ""


class TestRealDocumentsAreUnaffected:
    @pytest.mark.parametrize(
        "path",
        [
            "_deep_test/test_files/multipage.pdf",
            "_deep_test/test_files/sample.pdf",
            "_deep_test/test_files/korean.pdf",
        ],
    )
    def test_no_fallback_on_a_healthy_pdf(self, path: str) -> None:
        """The new triggers must not fire on documents that read correctly."""
        pymupdf = pytest.importorskip("pymupdf")
        from pathlib import Path

        full = Path(__file__).resolve().parents[3] / path
        if not full.exists():
            pytest.skip(f"{path} not available")

        doc = pymupdf.open(full)
        try:
            for index in range(doc.page_count):
                result = TextQualityAnalyzer(doc[index], index).analyze()
                if result.total_chars == 0:
                    continue  # genuinely image-only page
                assert result.is_fragmented is False, f"{path} page {index + 1}"
                assert result.needs_ocr is False, (
                    f"{path} page {index + 1}: {result.details}"
                )
        finally:
            doc.close()
