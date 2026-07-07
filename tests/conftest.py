# tests/conftest.py
"""
Shared fixtures for the Contextifier test suite.

Provides:
- Mock services (ImageService, TagService, ChartService, TableService)
- Default and custom ProcessingConfig instances
- Temporary directory helpers
- Sample FileContext factories
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import MagicMock

import pytest

from contextifier.config import ProcessingConfig
from contextifier.types import FileContext, get_category


# ── Config fixtures ───────────────────────────────────────────────────────

@pytest.fixture()
def default_config() -> ProcessingConfig:
    """Default ProcessingConfig with all defaults."""
    return ProcessingConfig()


@pytest.fixture()
def tmp_output_dir(tmp_path: Path) -> Path:
    """Temporary directory for image/chart output."""
    out = tmp_path / "output"
    out.mkdir()
    return out


# ── Mock service fixtures ─────────────────────────────────────────────────

@pytest.fixture()
def mock_tag_service() -> MagicMock:
    ts = MagicMock()
    ts.make_page_tag.side_effect = lambda n: f"[Page {n}]"
    ts.make_slide_tag.side_effect = lambda n: f"[Slide {n}]"
    ts.make_sheet_tag.side_effect = lambda name: f"[Sheet: {name}]"
    ts.create_image_tag.side_effect = lambda p: f"[Image:{p}]"
    ts.create_chart_open_tag.return_value = "[Chart Start]"
    ts.create_chart_close_tag.return_value = "[Chart End]"
    return ts


@pytest.fixture()
def mock_image_service() -> MagicMock:
    ims = MagicMock()
    ims.save.return_value = "/saved/image.png"
    ims.save_and_tag.return_value = "[Image:/saved/image.png]"
    ims.extract_and_deduplicate.return_value = "[Image:/saved/image.png]"
    ims.get_processed_count.return_value = 0
    ims.get_processed_paths.return_value = []
    return ims


@pytest.fixture()
def mock_chart_service() -> MagicMock:
    cs = MagicMock()
    cs.format_chart.return_value = "[Chart Start]\nChart Type: Bar\n[Chart End]"
    return cs


@pytest.fixture()
def mock_table_service() -> MagicMock:
    tbs = MagicMock()
    tbs.format_table.return_value = "<table><tr><td>cell</td></tr></table>"
    return tbs


@pytest.fixture()
def mock_metadata_service() -> MagicMock:
    ms = MagicMock()
    ms.format_metadata.return_value = "---\ntitle: Test\n---"
    return ms


@pytest.fixture()
def all_mock_services(
    mock_image_service: MagicMock,
    mock_tag_service: MagicMock,
    mock_chart_service: MagicMock,
    mock_table_service: MagicMock,
    mock_metadata_service: MagicMock,
) -> Dict[str, Any]:
    return {
        "image_service": mock_image_service,
        "tag_service": mock_tag_service,
        "chart_service": mock_chart_service,
        "table_service": mock_table_service,
        "metadata_service": mock_metadata_service,
    }


# ── FileContext factory ───────────────────────────────────────────────────

def make_file_context(
    content: bytes,
    extension: str = "txt",
    file_name: str = "test_file",
    file_path: Optional[str] = None,
) -> FileContext:
    """Create a FileContext for testing without touching the filesystem."""
    path = file_path or f"/tmp/test/{file_name}.{extension}"
    return FileContext(
        file_path=path,
        file_name=f"{file_name}.{extension}",
        file_extension=extension,
        file_category=get_category(extension).value,
        file_data=content,
        file_stream=None,
        file_size=len(content),
    )


@pytest.fixture()
def sample_text_context() -> FileContext:
    """Simple text file context for testing."""
    return make_file_context(b"Hello World\nLine two\n", extension="txt")
