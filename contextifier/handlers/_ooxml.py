# contextifier/handlers/_ooxml.py
"""
Shared readers for the OOXML chart part.

DOCX, XLSX and HWPX all embed the same ``c:chart`` XML, and each handler used
to read the title its own way — one took the first text run, one took every
run and joined them with spaces, one took the first again. This module is the
single answer.
"""

from __future__ import annotations

from typing import Any, List, Optional

# Namespaces used by the chart part.
NS_C = "http://schemas.openxmlformats.org/drawingml/2006/chart"
NS_A = "http://schemas.openxmlformats.org/drawingml/2006/main"


def _qn(ns: str, tag: str) -> str:
    return f"{{{ns}}}{tag}"


def _paragraph_text(paragraph: Any, ns_a: str) -> str:
    """Join the runs of one ``a:p``.

    Runs are joined with nothing between them: a run boundary is a formatting
    change, not a word boundary, so a title styled as a bold year followed by
    plain text is stored as two runs of one word — ``2026`` and ``년 실적`` —
    and separating them would read ``2026 년 실적``.
    """
    return "".join(t.text or "" for t in paragraph.iter(_qn(ns_a, "t")))


def extract_chart_title(
    chart_el: Any,
    *,
    ns_c: str = NS_C,
    ns_a: str = NS_A,
) -> Optional[str]:
    """
    Read the title of a chart element.

    Handles both forms the schema allows: rich text (``c:tx/c:rich``), where
    the title is authored in the chart, and a string reference
    (``c:tx/c:strRef``), where it points at a cell.

    Args:
        chart_el: An element at or above ``c:title`` in the chart part.
        ns_c: Chart namespace URI.
        ns_a: DrawingML namespace URI.

    Returns:
        The title, or None when the chart has none.
    """
    title_el = chart_el.find(f".//{_qn(ns_c, 'title')}")
    if title_el is None:
        return None

    rich = title_el.find(f".//{_qn(ns_c, 'rich')}")
    if rich is not None:
        paragraphs: List[str] = []
        for paragraph in rich.iter(_qn(ns_a, "p")):
            text = _paragraph_text(paragraph, ns_a)
            if text.strip():
                paragraphs.append(text.strip())
        if paragraphs:
            # A title that wraps is authored as several paragraphs; they read
            # as one line.
            return " ".join(paragraphs)
        # Some producers put runs directly under c:rich with no a:p wrapper.
        flat = _paragraph_text(rich, ns_a).strip()
        if flat:
            return flat

    cache = title_el.find(
        f".//{_qn(ns_c, 'strRef')}/{_qn(ns_c, 'strCache')}"
    )
    if cache is not None:
        value = cache.find(f".//{_qn(ns_c, 'v')}")
        if value is not None and value.text and value.text.strip():
            return value.text.strip()

    return None


__all__ = ["extract_chart_title", "NS_C", "NS_A"]
