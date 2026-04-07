"""
Deep Integration Test Runner for Contextify v0.2.6
===================================================
Tests every handler, chunking strategy, service, config option, and edge case.
Results are collected into a structured report.
"""
import os
import sys
import json
import time
import traceback
import asyncio
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from contextifier import (
    DocumentProcessor, AsyncDocumentProcessor, CachedDocumentProcessor,
    ProcessingConfig, ChunkingConfig, TextChunker,
    ExtractionResult, ChunkResult, Chunk, ChunkMetadata,
    ContextifierError, UnsupportedFormatError,
)
from contextifier.config import (
    TagConfig, ImageConfig, ChartConfig, MetadataConfig,
    TableConfig, OCRConfig, EncodingConfig,
)
from contextifier.types import (
    OutputFormat, NamingStrategy, FileContext,
)

BASE = Path(__file__).parent / "test_files"
OUTPUT = Path(__file__).parent / "output"
RESULTS = Path(__file__).parent / "results"
OUTPUT.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)


@dataclass
class TestResult:
    name: str
    category: str
    status: str  # PASS, FAIL, ERROR, WARN, SKIP
    duration_ms: float = 0.0
    details: str = ""
    error: str = ""
    data: dict = field(default_factory=dict)


class DeepTestRunner:
    def __init__(self):
        self.results: list[TestResult] = []
        self.processor = DocumentProcessor()

    def run_test(self, name: str, category: str, func, *args, **kwargs) -> TestResult:
        start = time.perf_counter()
        try:
            result_data = func(*args, **kwargs)
            elapsed = (time.perf_counter() - start) * 1000
            result = TestResult(
                name=name, category=category, status="PASS",
                duration_ms=elapsed, data=result_data or {}
            )
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            result = TestResult(
                name=name, category=category, status="ERROR",
                duration_ms=elapsed, error=f"{type(e).__name__}: {e}",
                details=traceback.format_exc()
            )
        self.results.append(result)
        status_icon = {"PASS": "OK", "FAIL": "FAIL", "ERROR": "ERR", "WARN": "WARN", "SKIP": "SKIP"}
        print(f"  [{status_icon.get(result.status, '?'):>4s}] {name} ({elapsed:.1f}ms)")
        return result

    def add_result(self, result: TestResult):
        self.results.append(result)

    # ============================================================
    # 1. HANDLER TESTS - extract_text for every format
    # ============================================================
    def test_extract_text_all_formats(self):
        print("\n" + "="*70)
        print("SECTION 1: extract_text() for all formats")
        print("="*70)

        test_files = {
            # Text formats
            "TXT": "sample.txt",
            "Markdown": "sample.md",
            "Python": "sample.py",
            "JSON": "sample.json",
            "YAML": "sample.yaml",
            "XML": "sample.xml",
            "INI": "sample.ini",
            "LOG": "sample.log",
            # Document formats
            "CSV": "sample.csv",
            "TSV": "sample.tsv",
            "HTML": "sample.html",
            "RTF": "sample.rtf",
            "RTF (table)": "table.rtf",
            "DOCX": "sample.docx",
            "PPTX": "sample.pptx",
            "XLSX": "sample.xlsx",
            "XLS": "sample.xls",
            "PDF": "sample.pdf",
        }

        for label, filename in test_files.items():
            filepath = BASE / filename
            if not filepath.exists():
                self.results.append(TestResult(
                    name=f"extract_text: {label}", category="extract_text",
                    status="SKIP", details=f"File not found: {filename}"
                ))
                continue

            def _test(fp=filepath, lbl=label):
                text = self.processor.extract_text(str(fp))
                data = {
                    "file": fp.name,
                    "text_length": len(text),
                    "line_count": text.count("\n") + 1,
                    "preview": text[:500].replace("\n", "\\n"),
                    "empty": len(text.strip()) == 0,
                }
                if data["empty"]:
                    raise ValueError(f"extract_text returned empty result for {lbl}")
                return data

            self.run_test(f"extract_text: {label}", "extract_text", _test)

    # ============================================================
    # 2. FULL PROCESS TESTS - process() for each format
    # ============================================================
    def test_process_all_formats(self):
        print("\n" + "="*70)
        print("SECTION 2: process() full extraction for key formats")
        print("="*70)

        test_files = {
            "DOCX": "sample.docx",
            "PPTX": "sample.pptx",
            "XLSX": "sample.xlsx",
            "XLS": "sample.xls",
            "PDF": "sample.pdf",
            "HTML": "sample.html",
            "CSV": "sample.csv",
            "RTF (table)": "table.rtf",
        }

        for label, filename in test_files.items():
            filepath = BASE / filename
            if not filepath.exists():
                continue

            def _test(fp=filepath, lbl=label):
                result = self.processor.process(str(fp))
                meta = result.metadata
                has_meta = meta is not None and not meta.is_empty() if meta else False
                meta_dict = meta.to_dict() if meta else {}
                data = {
                    "file": fp.name,
                    "text_length": len(result.text),
                    "has_metadata": has_meta,
                    "metadata_keys": list(meta_dict.keys()) if meta_dict else [],
                    "table_count": len(result.tables) if result.tables else 0,
                    "image_count": len(result.images) if result.images else 0,
                    "chart_count": len(result.charts) if result.charts else 0,
                    "text_preview": result.text[:300].replace("\n", "\\n"),
                }
                if meta_dict:
                    data["metadata_sample"] = {k: str(v)[:100] for k, v in list(meta_dict.items())[:5]}
                if result.tables:
                    data["first_table_preview"] = str(result.tables[0])[:200]
                return data

            self.run_test(f"process: {label}", "process", _test)

    # ============================================================
    # 3. EXTRACT_CHUNKS TESTS
    # ============================================================
    def test_extract_chunks(self):
        print("\n" + "="*70)
        print("SECTION 3: extract_chunks() for each format")
        print("="*70)

        test_files = {
            "DOCX": "sample.docx",
            "DOCX (long)": "long.docx",
            "PPTX": "sample.pptx",
            "XLSX": "sample.xlsx",
            "PDF": "sample.pdf",
            "PDF (multipage)": "multipage.pdf",
            "HTML": "sample.html",
            "CSV": "sample.csv",
            "CSV (large)": "large.csv",
            "TXT": "sample.txt",
            "Markdown": "sample.md",
        }

        for label, filename in test_files.items():
            filepath = BASE / filename
            if not filepath.exists():
                continue

            def _test(fp=filepath, lbl=label):
                chunk_result = self.processor.extract_chunks(str(fp), chunk_size=500)
                chunks = chunk_result.chunks
                data = {
                    "file": fp.name,
                    "chunk_count": len(chunks),
                    "strategy": chunk_result.strategy if hasattr(chunk_result, 'strategy') else "unknown",
                    "chunk_sizes": [len(c) if isinstance(c, str) else len(c.content) for c in chunks[:10]],
                    "total_chars": sum(len(c) if isinstance(c, str) else len(c.content) for c in chunks),
                }
                if chunks:
                    first = chunks[0]
                    if isinstance(first, str):
                        data["first_chunk_preview"] = first[:200].replace("\n", "\\n")
                    else:
                        data["first_chunk_preview"] = first.content[:200].replace("\n", "\\n")
                if len(chunks) == 0:
                    raise ValueError(f"No chunks produced for {lbl}")
                return data

            self.run_test(f"extract_chunks: {label}", "extract_chunks", _test)

    # ============================================================
    # 4. CHUNKING STRATEGY TESTS
    # ============================================================
    def test_chunking_strategies(self):
        print("\n" + "="*70)
        print("SECTION 4: Chunking Strategy Tests")
        print("="*70)

        config = ProcessingConfig()
        chunker = TextChunker(config)

        # 4a. Plain text chunking
        def test_plain_chunking():
            text = "Hello world. " * 200  # ~2600 chars
            chunks = chunker.chunk(text, chunk_size=500, chunk_overlap=100)
            data = {
                "input_len": len(text),
                "chunk_count": len(chunks),
                "chunk_sizes": [len(c) if isinstance(c, str) else len(c.content) for c in chunks],
            }
            for c in chunks:
                size = len(c) if isinstance(c, str) else len(c.content)
                if size > 600:  # allow some tolerance
                    raise ValueError(f"Chunk exceeds limit: {size} > 600")
            return data
        self.run_test("Chunking: Plain text splitting", "chunking", test_plain_chunking)

        # 4b. Page-based chunking
        def test_page_chunking():
            pages = []
            for i in range(5):
                pages.append(f"[Page Number: {i+1}]\nContent of page {i+1}. " + "Lorem ipsum. " * 50)
            text = "\n\n".join(pages)
            chunks = chunker.chunk(text, chunk_size=500, chunk_overlap=100)
            data = {
                "input_len": len(text),
                "chunk_count": len(chunks),
                "has_page_markers": any("[Page Number:" in (c if isinstance(c, str) else c.content) for c in chunks),
                "chunk_sizes": [len(c) if isinstance(c, str) else len(c.content) for c in chunks],
            }
            return data
        self.run_test("Chunking: Page-based splitting", "chunking", test_page_chunking)

        # 4c. Table-preserving chunking (spreadsheet format)
        def test_table_chunking():
            text = ""
            for i in range(3):
                text += f"[Sheet: Sheet{i+1}]\n"
                text += "<table>\n<tr><th>Col1</th><th>Col2</th></tr>\n"
                for j in range(10):
                    text += f"<tr><td>Data{j}</td><td>Value{j}</td></tr>\n"
                text += "</table>\n\n"

            config = ProcessingConfig(
                chunking=ChunkingConfig(chunk_size=500, chunk_overlap=50)
            )
            proc = DocumentProcessor(config=config)
            # Use chunker directly with file context
            chunks = chunker.chunk(
                text, chunk_size=500, chunk_overlap=50,
                file_extension=".xlsx"
            )
            data = {
                "input_len": len(text),
                "chunk_count": len(chunks),
                "chunk_sizes": [len(c) if isinstance(c, str) else len(c.content) for c in chunks],
            }
            return data
        self.run_test("Chunking: Table/Sheet-based splitting", "chunking", test_table_chunking)

        # 4d. Protected region chunking (HTML tables)
        def test_protected_chunking():
            text = "Some intro text.\n\n"
            text += "<table>\n<tr><th>A</th><th>B</th></tr>\n"
            for i in range(20):
                text += f"<tr><td>Cell{i}A</td><td>Cell{i}B</td></tr>\n"
            text += "</table>\n\n"
            text += "Some trailing text after the table."

            chunks = chunker.chunk(text, chunk_size=300, chunk_overlap=50)
            data = {
                "input_len": len(text),
                "chunk_count": len(chunks),
                "table_preserved": any("<table>" in (c if isinstance(c, str) else c.content) for c in chunks),
                "chunk_sizes": [len(c) if isinstance(c, str) else len(c.content) for c in chunks],
            }
            return data
        self.run_test("Chunking: Protected region preservation", "chunking", test_protected_chunking)

        # 4e. Chunk overlap verification
        def test_chunk_overlap():
            text = " ".join([f"word{i}" for i in range(500)])
            chunks = chunker.chunk(text, chunk_size=200, chunk_overlap=50)
            data = {
                "chunk_count": len(chunks),
                "chunk_sizes": [len(c) if isinstance(c, str) else len(c.content) for c in chunks],
            }
            # Check if overlapping content exists between consecutive chunks
            if len(chunks) >= 2:
                c1 = chunks[0] if isinstance(chunks[0], str) else chunks[0].content
                c2 = chunks[1] if isinstance(chunks[1], str) else chunks[1].content
                overlap_found = False
                # Check last 50 chars of c1 appear in beginning of c2
                c1_tail = c1[-50:]
                if any(word in c2[:100] for word in c1_tail.split()):
                    overlap_found = True
                data["overlap_detected"] = overlap_found
            return data
        self.run_test("Chunking: Overlap verification", "chunking", test_chunk_overlap)

        # 4f. Empty text chunking
        def test_empty_chunking():
            chunks = chunker.chunk("", chunk_size=500, chunk_overlap=50)
            return {"chunk_count": len(chunks), "result": "empty list" if len(chunks) == 0 else "non-empty"}
        self.run_test("Chunking: Empty text", "chunking", test_empty_chunking)

        # 4g. Very small chunk size
        def test_small_chunk():
            text = "Hello world, this is a test."
            chunks = chunker.chunk(text, chunk_size=10, chunk_overlap=2)
            data = {
                "chunk_count": len(chunks),
                "chunk_sizes": [len(c) if isinstance(c, str) else len(c.content) for c in chunks],
            }
            return data
        self.run_test("Chunking: Very small chunk_size=10", "chunking", test_small_chunk)

        # 4h. Position metadata
        def test_position_metadata():
            proc = DocumentProcessor()
            text = "First paragraph. " * 30 + "\n\nSecond paragraph. " * 30
            chunks = proc.chunk_text(text, chunk_size=200, chunk_overlap=50, include_position_metadata=True)
            data = {
                "chunk_count": len(chunks),
                "has_metadata": False,
            }
            if chunks:
                first = chunks[0]
                if isinstance(first, Chunk):
                    data["has_metadata"] = first.metadata is not None
                    if first.metadata:
                        data["metadata_fields"] = {
                            "chunk_index": first.metadata.chunk_index,
                            "page_number": first.metadata.page_number,
                            "line_start": first.metadata.line_start,
                            "line_end": first.metadata.line_end,
                        }
                else:
                    data["type"] = type(first).__name__
            return data
        self.run_test("Chunking: Position metadata", "chunking", test_position_metadata)

    # ============================================================
    # 5. TABLE SERVICE & TABLE FORMAT TESTS
    # ============================================================
    def test_table_handling(self):
        print("\n" + "="*70)
        print("SECTION 5: Table Handling Tests")
        print("="*70)

        # 5a. DOCX tables
        def test_docx_tables():
            result = self.processor.process(str(BASE / "sample.docx"))
            data = {
                "table_count": len(result.tables) if result.tables else 0,
                "table_previews": [str(t)[:200] for t in (result.tables or [])]
            }
            if not result.tables:
                raise ValueError("No tables extracted from DOCX with tables")
            return data
        self.run_test("Tables: DOCX extraction", "tables", test_docx_tables)

        # 5b. DOCX merged cells
        def test_docx_merged():
            result = self.processor.process(str(BASE / "merged_cells.docx"))
            return {
                "table_count": len(result.tables) if result.tables else 0,
                "text_has_merged": "Merged" in result.text,
                "text_preview": result.text[:500].replace("\n", "\\n"),
            }
        self.run_test("Tables: DOCX merged cells", "tables", test_docx_merged)

        # 5c. HTML tables
        def test_html_tables():
            result = self.processor.process(str(BASE / "sample.html"))
            return {
                "table_count": len(result.tables) if result.tables else 0,
                "text_preview": result.text[:500].replace("\n", "\\n"),
            }
        self.run_test("Tables: HTML extraction", "tables", test_html_tables)

        # 5d. HTML complex table (colspan/rowspan)
        def test_html_complex_table():
            result = self.processor.process(str(BASE / "complex_table.html"))
            return {
                "table_count": len(result.tables) if result.tables else 0,
                "text_contains_quarterly": "Quarterly" in result.text,
                "text_preview": result.text[:500].replace("\n", "\\n"),
            }
        self.run_test("Tables: HTML colspan/rowspan", "tables", test_html_complex_table)

        # 5e. XLSX multi-sheet tables
        def test_xlsx_tables():
            result = self.processor.process(str(BASE / "sample.xlsx"))
            text = result.text
            data = {
                "table_count": len(result.tables) if result.tables else 0,
                "text_length": len(text),
                "has_sales_data": "Laptop" in text,
                "has_regions": "Asia" in text or "Region" in text,
                "text_preview": text[:800].replace("\n", "\\n"),
            }
            return data
        self.run_test("Tables: XLSX multi-sheet", "tables", test_xlsx_tables)

        # 5f. CSV table extraction
        def test_csv_tables():
            result = self.processor.process(str(BASE / "sample.csv"))
            return {
                "table_count": len(result.tables) if result.tables else 0,
                "text_has_data": "Alice" in result.text,
                "text_preview": result.text[:500].replace("\n", "\\n"),
            }
        self.run_test("Tables: CSV extraction", "tables", test_csv_tables)

        # 5g. Table format HTML
        def test_table_format_html():
            config = ProcessingConfig(
                tables=TableConfig(output_format=OutputFormat.HTML)
            )
            proc = DocumentProcessor(config=config)
            result = proc.process(str(BASE / "sample.xlsx"))
            has_html_tags = "<table" in result.text or "<tr" in result.text or "<td" in result.text
            return {"format": "HTML", "has_html_tags": has_html_tags, "text_len": len(result.text)}
        self.run_test("Tables: HTML format output", "tables", test_table_format_html)

        # 5h. Table format Markdown
        def test_table_format_md():
            config = ProcessingConfig(
                tables=TableConfig(output_format=OutputFormat.MARKDOWN)
            )
            proc = DocumentProcessor(config=config)
            result = proc.process(str(BASE / "sample.xlsx"))
            has_md_table = "|" in result.text and "---" in result.text
            return {"format": "MARKDOWN", "has_markdown_pipes": has_md_table, "text_len": len(result.text)}
        self.run_test("Tables: Markdown format output", "tables", test_table_format_md)

        # 5i. Table format Text
        def test_table_format_text():
            config = ProcessingConfig(
                tables=TableConfig(output_format=OutputFormat.TEXT)
            )
            proc = DocumentProcessor(config=config)
            result = proc.process(str(BASE / "sample.xlsx"))
            return {"format": "TEXT", "text_len": len(result.text), "preview": result.text[:500].replace("\n", "\\n")}
        self.run_test("Tables: Text format output", "tables", test_table_format_text)

    # ============================================================
    # 6. METADATA EXTRACTION TESTS
    # ============================================================
    def test_metadata_extraction(self):
        print("\n" + "="*70)
        print("SECTION 6: Metadata Extraction Tests")
        print("="*70)

        # 6a. DOCX metadata
        def test_docx_meta():
            result = self.processor.process(str(BASE / "sample.docx"))
            meta = result.metadata
            meta_dict = meta.to_dict() if meta else {}
            return {
                "metadata_keys": list(meta_dict.keys()),
                "has_title": bool(meta and meta.title),
                "has_author": bool(meta and meta.author),
                "title": meta.title if meta else None,
                "author": meta.author if meta else None,
                "metadata_values": {k: str(v)[:100] for k, v in meta_dict.items()},
            }
        self.run_test("Metadata: DOCX", "metadata", test_docx_meta)

        # 6b. PDF metadata
        def test_pdf_meta():
            result = self.processor.process(str(BASE / "sample.pdf"))
            meta = result.metadata
            meta_dict = meta.to_dict() if meta else {}
            return {
                "metadata_keys": list(meta_dict.keys()),
                "metadata_values": {k: str(v)[:100] for k, v in meta_dict.items()},
            }
        self.run_test("Metadata: PDF", "metadata", test_pdf_meta)

        # 6c. PPTX metadata
        def test_pptx_meta():
            result = self.processor.process(str(BASE / "sample.pptx"))
            meta = result.metadata
            meta_dict = meta.to_dict() if meta else {}
            return {
                "metadata_keys": list(meta_dict.keys()),
                "metadata_values": {k: str(v)[:100] for k, v in meta_dict.items()},
            }
        self.run_test("Metadata: PPTX", "metadata", test_pptx_meta)

        # 6d. Metadata language (Korean)
        def test_meta_korean():
            config = ProcessingConfig(
                metadata=MetadataConfig(language="ko")
            )
            proc = DocumentProcessor(config=config)
            result = proc.process(str(BASE / "sample.docx"))
            text = result.text
            # Korean metadata labels should be present
            return {
                "text_preview": text[:500].replace("\n", "\\n"),
                "has_korean_labels": any(ord(c) > 0xAC00 for c in text[:500]),
            }
        self.run_test("Metadata: Korean language", "metadata", test_meta_korean)

        # 6e. Metadata language (English)
        def test_meta_english():
            config = ProcessingConfig(
                metadata=MetadataConfig(language="en")
            )
            proc = DocumentProcessor(config=config)
            result = proc.process(str(BASE / "sample.docx"))
            return {"text_preview": result.text[:500].replace("\n", "\\n")}
        self.run_test("Metadata: English language", "metadata", test_meta_english)

    # ============================================================
    # 7. TAG SYSTEM TESTS
    # ============================================================
    def test_tag_system(self):
        print("\n" + "="*70)
        print("SECTION 7: Tag System Tests")
        print("="*70)

        # 7a. Custom page tags
        def test_custom_page_tags():
            config = ProcessingConfig(
                tags=TagConfig(page_prefix="<page>", page_suffix="</page>")
            )
            proc = DocumentProcessor(config=config)
            result = proc.process(str(BASE / "multipage.pdf"))
            text = result.text
            return {
                "has_custom_page_tags": "<page>" in text,
                "page_tag_count": text.count("<page>"),
                "text_preview": text[:500].replace("\n", "\\n"),
            }
        self.run_test("Tags: Custom page tags", "tags", test_custom_page_tags)

        # 7b. Default tags (PDF)
        def test_default_tags_pdf():
            text = self.processor.extract_text(str(BASE / "multipage.pdf"))
            return {
                "has_page_markers": "[Page" in text or "Page Number" in text,
                "text_preview": text[:500].replace("\n", "\\n"),
            }
        self.run_test("Tags: Default PDF page tags", "tags", test_default_tags_pdf)

        # 7c. PPTX slide tags
        def test_slide_tags():
            text = self.processor.extract_text(str(BASE / "sample.pptx"))
            return {
                "has_slide_markers": "Slide" in text or "slide" in text,
                "text_preview": text[:500].replace("\n", "\\n"),
            }
        self.run_test("Tags: PPTX slide tags", "tags", test_slide_tags)

        # 7d. XLSX sheet tags
        def test_sheet_tags():
            text = self.processor.extract_text(str(BASE / "sample.xlsx"))
            return {
                "has_sheet_markers": "Sheet" in text or "sheet" in text or "Sales" in text,
                "text_preview": text[:800].replace("\n", "\\n"),
            }
        self.run_test("Tags: XLSX sheet tags", "tags", test_sheet_tags)

    # ============================================================
    # 8. EDGE CASE TESTS
    # ============================================================
    def test_edge_cases(self):
        print("\n" + "="*70)
        print("SECTION 8: Edge Case Tests")
        print("="*70)

        # 8a. Empty file
        def test_empty_txt():
            text = self.processor.extract_text(str(BASE / "empty.txt"))
            return {"text_length": len(text), "is_empty": len(text.strip()) == 0}
        self.run_test("Edge: Empty TXT file", "edge_cases", test_empty_txt)

        # 8b. Empty CSV
        def test_empty_csv():
            text = self.processor.extract_text(str(BASE / "empty.csv"))
            return {"text_length": len(text), "is_empty": len(text.strip()) == 0}
        self.run_test("Edge: Empty CSV file", "edge_cases", test_empty_csv)

        # 8c. Empty XLSX
        def test_empty_xlsx():
            text = self.processor.extract_text(str(BASE / "empty.xlsx"))
            return {"text_length": len(text), "preview": text[:200].replace("\n", "\\n")}
        self.run_test("Edge: Empty XLSX file", "edge_cases", test_empty_xlsx)

        # 8d. Header-only CSV
        def test_header_only_csv():
            text = self.processor.extract_text(str(BASE / "header_only.csv"))
            return {"text_length": len(text), "has_headers": "Name" in text, "preview": text[:200]}
        self.run_test("Edge: Header-only CSV", "edge_cases", test_header_only_csv)

        # 8e. Very long single line
        def test_long_line():
            text = self.processor.extract_text(str(BASE / "long_line.txt"))
            return {"text_length": len(text), "original_length": 100002}
        self.run_test("Edge: 100K char single line", "edge_cases", test_long_line)

        # 8f. Whitespace-only file
        def test_whitespace():
            text = self.processor.extract_text(str(BASE / "whitespace.txt"))
            return {"text_length": len(text), "stripped_length": len(text.strip())}
        self.run_test("Edge: Whitespace-only file", "edge_cases", test_whitespace)

        # 8g. UTF-8 BOM file
        def test_bom():
            text = self.processor.extract_text(str(BASE / "sample_bom.txt"))
            has_bom = text.startswith("\ufeff")
            return {
                "text_length": len(text),
                "has_bom_in_output": has_bom,
                "preview": repr(text[:50]),
            }
        self.run_test("Edge: UTF-8 BOM handling", "edge_cases", test_bom)

        # 8h. EUC-KR encoding
        def test_euckr():
            text = self.processor.extract_text(str(BASE / "sample_euckr.txt"))
            has_korean = any(ord(c) > 0xAC00 for c in text)
            return {
                "text_length": len(text),
                "has_korean": has_korean,
                "preview": text[:200],
            }
        self.run_test("Edge: EUC-KR encoding detection", "edge_cases", test_euckr)

        # 8i. Deeply nested JSON
        def test_nested_json():
            text = self.processor.extract_text(str(BASE / "nested.json"))
            return {"text_length": len(text), "preview": text[:300].replace("\n", "\\n")}
        self.run_test("Edge: Deeply nested JSON", "edge_cases", test_nested_json)

        # 8j. Wide CSV (100 columns)
        def test_wide_csv():
            text = self.processor.extract_text(str(BASE / "wide.csv"))
            return {
                "text_length": len(text),
                "has_col_99": "Col_99" in text,
                "preview": text[:300].replace("\n", "\\n"),
            }
        self.run_test("Edge: Wide CSV (100 columns)", "edge_cases", test_wide_csv)

        # 8k. Large CSV (1000 rows)
        def test_large_csv():
            text = self.processor.extract_text(str(BASE / "large.csv"))
            return {
                "text_length": len(text),
                "has_first_row": "Item_0" in text,
                "has_last_row": "Item_999" in text,
            }
        self.run_test("Edge: Large CSV (1000 rows)", "edge_cases", test_large_csv)

        # 8l. Large XLSX
        def test_large_xlsx():
            text = self.processor.extract_text(str(BASE / "large.xlsx"))
            return {
                "text_length": len(text),
                "has_data": "Item_0" in text,
            }
        self.run_test("Edge: Large XLSX (500 rows)", "edge_cases", test_large_xlsx)

        # 8m. CSV with special characters
        def test_complex_csv():
            text = self.processor.extract_text(str(BASE / "complex.csv"))
            return {
                "text_length": len(text),
                "has_quotes": "quotes" in text,
                "has_korean": "한글" in text,
                "preview": text[:500].replace("\n", "\\n"),
            }
        self.run_test("Edge: CSV with special chars", "edge_cases", test_complex_csv)

        # 8n. Semicolon-delimited CSV
        def test_semicolon_csv():
            text = self.processor.extract_text(str(BASE / "semicolon.csv"))
            return {
                "text_length": len(text),
                "has_data": "Alice" in text,
                "preview": text[:300].replace("\n", "\\n"),
            }
        self.run_test("Edge: Semicolon-delimited CSV", "edge_cases", test_semicolon_csv)

        # 8o. Deeply nested HTML
        def test_nested_html():
            text = self.processor.extract_text(str(BASE / "nested.html"))
            return {
                "text_length": len(text),
                "has_content": "nested content" in text.lower(),
                "preview": text[:300].replace("\n", "\\n"),
            }
        self.run_test("Edge: Deeply nested HTML", "edge_cases", test_nested_html)

        # 8p. HTML with XSS content
        def test_xss_html():
            result = self.processor.process(str(BASE / "xss_test.html"))
            text = result.text
            has_script_tag = "<script>" in text
            return {
                "text_length": len(text),
                "has_raw_script_tag": has_script_tag,
                "security_ok": not has_script_tag,
                "preview": text[:300].replace("\n", "\\n"),
            }
        self.run_test("Edge: HTML XSS safety", "edge_cases", test_xss_html)

        # 8q. Unsupported format
        def test_unsupported():
            try:
                self.processor.extract_text("nonexistent.zzz")
                return {"raised": False, "error": "No error raised!"}
            except (UnsupportedFormatError, ContextifierError, FileNotFoundError) as e:
                return {"raised": True, "error_type": type(e).__name__, "message": str(e)[:200]}
        self.run_test("Edge: Unsupported format", "edge_cases", test_unsupported)

        # 8r. Non-existent file
        def test_nonexistent():
            try:
                self.processor.extract_text("/tmp/does_not_exist_12345.pdf")
                return {"raised": False}
            except (FileNotFoundError, ContextifierError) as e:
                return {"raised": True, "error_type": type(e).__name__}
        self.run_test("Edge: Non-existent file", "edge_cases", test_nonexistent)

        # 8s. Korean HTML
        def test_korean_html():
            text = self.processor.extract_text(str(BASE / "korean.html"))
            return {
                "text_length": len(text),
                "has_korean": "홍길동" in text or "한글" in text,
                "preview": text[:300].replace("\n", "\\n"),
            }
        self.run_test("Edge: Korean HTML", "edge_cases", test_korean_html)

        # 8t. Long DOCX (many chapters)
        def test_long_docx():
            text = self.processor.extract_text(str(BASE / "long.docx"))
            return {
                "text_length": len(text),
                "has_chapter_1": "Chapter 1" in text,
                "has_chapter_50": "Chapter 50" in text,
            }
        self.run_test("Edge: Long DOCX (50 chapters)", "edge_cases", test_long_docx)

    # ============================================================
    # 9. CONFIGURATION TESTS
    # ============================================================
    def test_configuration(self):
        print("\n" + "="*70)
        print("SECTION 9: Configuration Tests")
        print("="*70)

        # 9a. Default config
        def test_default_config():
            config = ProcessingConfig()
            return {
                "chunk_size": config.chunking.chunk_size,
                "chunk_overlap": config.chunking.chunk_overlap,
                "table_format": str(config.tables.output_format),
                "metadata_language": str(config.metadata.language),
                "ocr_enabled": config.ocr.enabled,
            }
        self.run_test("Config: Default values", "config", test_default_config)

        # 9b. Fluent API
        def test_fluent_api():
            config = ProcessingConfig()
            config2 = config.with_chunking(chunk_size=2000, chunk_overlap=300)
            return {
                "original_chunk_size": config.chunking.chunk_size,
                "new_chunk_size": config2.chunking.chunk_size,
                "immutable": config.chunking.chunk_size != config2.chunking.chunk_size,
            }
        self.run_test("Config: Fluent API immutability", "config", test_fluent_api)

        # 9c. Serialization round-trip
        def test_serialization():
            config = ProcessingConfig(
                chunking=ChunkingConfig(chunk_size=2000),
                tables=TableConfig(output_format=OutputFormat.MARKDOWN),
            )
            d = config.to_dict()
            config2 = ProcessingConfig.from_dict(d)
            return {
                "chunk_size_match": config.chunking.chunk_size == config2.chunking.chunk_size,
                "table_format_match": config.tables.output_format == config2.tables.output_format,
                "dict_keys": list(d.keys()),
            }
        self.run_test("Config: Serialization round-trip", "config", test_serialization)

        # 9d. Format options
        def test_format_options():
            config = ProcessingConfig(
                format_options={"csv": {"delimiter": ";"}}
            )
            proc = DocumentProcessor(config=config)
            text = proc.extract_text(str(BASE / "semicolon.csv"))
            return {
                "text_length": len(text),
                "has_alice": "Alice" in text,
                "preview": text[:300].replace("\n", "\\n"),
            }
        self.run_test("Config: format_options (CSV delimiter)", "config", test_format_options)

        # 9e. EncodingConfig
        def test_encoding_config():
            config = ProcessingConfig(
                encoding=EncodingConfig(fallback_encodings=["euc-kr", "cp949", "utf-8"])
            )
            proc = DocumentProcessor(config=config)
            text = proc.extract_text(str(BASE / "sample_euckr.txt"))
            return {
                "text_length": len(text),
                "has_korean": any(ord(c) > 0xAC00 for c in text),
                "preview": text[:200],
            }
        self.run_test("Config: EncodingConfig", "config", test_encoding_config)

        # 9f. Image config
        def test_image_config():
            img_dir = OUTPUT / "images"
            img_dir.mkdir(exist_ok=True)
            config = ProcessingConfig(
                images=ImageConfig(
                    directory_path=str(img_dir),
                    naming_strategy=NamingStrategy.HASH,
                )
            )
            proc = DocumentProcessor(config=config)
            # Just verify it doesn't error
            result = proc.process(str(BASE / "sample.docx"))
            return {"images_extracted": len(result.images) if result.images else 0}
        self.run_test("Config: ImageConfig", "config", test_image_config)

    # ============================================================
    # 10. ASYNC PROCESSOR TESTS
    # ============================================================
    def test_async_processor(self):
        print("\n" + "="*70)
        print("SECTION 10: Async Processor Tests")
        print("="*70)

        # 10a. Basic async extract_text
        def test_async_extract():
            async def _run():
                proc = AsyncDocumentProcessor()
                text = await proc.extract_text(str(BASE / "sample.docx"))
                return {"text_length": len(text), "has_content": len(text.strip()) > 0}
            return asyncio.run(_run())
        self.run_test("Async: extract_text", "async", test_async_extract)

        # 10b. Async process
        def test_async_process():
            async def _run():
                proc = AsyncDocumentProcessor()
                result = await proc.process(str(BASE / "sample.docx"))
                return {
                    "text_length": len(result.text),
                    "has_tables": bool(result.tables),
                    "has_metadata": result.metadata is not None and not result.metadata.is_empty(),
                }
            return asyncio.run(_run())
        self.run_test("Async: process", "async", test_async_process)

        # 10c. Async extract_chunks
        def test_async_chunks():
            async def _run():
                proc = AsyncDocumentProcessor()
                chunk_result = await proc.extract_chunks(str(BASE / "sample.docx"), chunk_size=500)
                return {"chunk_count": len(chunk_result.chunks)}
            return asyncio.run(_run())
        self.run_test("Async: extract_chunks", "async", test_async_chunks)

        # 10d. Async batch processing
        def test_async_batch():
            async def _run():
                proc = AsyncDocumentProcessor()
                files = [
                    str(BASE / "sample.txt"),
                    str(BASE / "sample.csv"),
                    str(BASE / "sample.html"),
                ]
                results = await proc.extract_batch(files, max_concurrent=3)
                return {
                    "result_count": len(results),
                    "result_keys": list(results.keys()) if isinstance(results, dict) else "not_dict",
                    "all_success": all(isinstance(v, str) for v in results.values()) if isinstance(results, dict) else False,
                }
            return asyncio.run(_run())
        self.run_test("Async: batch processing", "async", test_async_batch)

    # ============================================================
    # 11. CACHED PROCESSOR TESTS
    # ============================================================
    def test_cached_processor(self):
        print("\n" + "="*70)
        print("SECTION 11: Cached Processor Tests")
        print("="*70)

        # 11a. Cache hit
        def test_cache_hit():
            proc = CachedDocumentProcessor()
            filepath = str(BASE / "sample.txt")

            start1 = time.perf_counter()
            text1 = proc.extract_text(filepath)
            time1 = time.perf_counter() - start1

            start2 = time.perf_counter()
            text2 = proc.extract_text(filepath)
            time2 = time.perf_counter() - start2

            return {
                "texts_match": text1 == text2,
                "first_call_ms": round(time1 * 1000, 2),
                "cached_call_ms": round(time2 * 1000, 2),
                "cache_faster": time2 < time1,
            }
        self.run_test("Cache: Hit verification", "cache", test_cache_hit)

        # 11b. Cache with different configs
        def test_cache_config_isolation():
            proc = CachedDocumentProcessor()
            filepath = str(BASE / "sample.txt")

            text1 = proc.extract_text(filepath)

            proc2 = CachedDocumentProcessor(
                config=ProcessingConfig(chunking=ChunkingConfig(chunk_size=100))
            )
            text2 = proc2.extract_text(filepath)

            return {
                "text1_len": len(text1),
                "text2_len": len(text2),
            }
        self.run_test("Cache: Config isolation", "cache", test_cache_config_isolation)

    # ============================================================
    # 12. CONTENT INTEGRITY TESTS
    # ============================================================
    def test_content_integrity(self):
        print("\n" + "="*70)
        print("SECTION 12: Content Integrity Verification")
        print("="*70)

        # 12a. CSV data completeness
        def test_csv_integrity():
            text = self.processor.extract_text(str(BASE / "sample.csv"))
            expected = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Seoul", "Tokyo", "New York", "London", "Paris"]
            found = {w: w in text for w in expected}
            missing = [w for w, f in found.items() if not f]
            return {
                "all_found": len(missing) == 0,
                "missing": missing,
                "text_preview": text[:500].replace("\n", "\\n"),
            }
        self.run_test("Integrity: CSV data completeness", "integrity", test_csv_integrity)

        # 12b. DOCX content preservation
        def test_docx_integrity():
            text = self.processor.extract_text(str(BASE / "sample.docx"))
            expected = ["Test Document", "Section 1", "Section 2", "Section 3", "Table",
                        "Alice", "Bob", "Charlie", "Seoul", "Tokyo", "Conclusion"]
            found = {w: w in text for w in expected}
            missing = [w for w, f in found.items() if not f]
            return {
                "all_found": len(missing) == 0,
                "missing": missing,
                "text_length": len(text),
            }
        self.run_test("Integrity: DOCX content preservation", "integrity", test_docx_integrity)

        # 12c. PPTX content (all slides + notes)
        def test_pptx_integrity():
            text = self.processor.extract_text(str(BASE / "sample.pptx"))
            expected = ["Test Presentation", "Content Slide", "Table Slide",
                        "bullet point", "Laptop", "Mouse", "speaker notes"]
            found = {w: w.lower() in text.lower() for w in expected}
            missing = [w for w, f in found.items() if not f]
            return {
                "all_found": len(missing) == 0,
                "missing": missing,
                "text_preview": text[:800].replace("\n", "\\n"),
            }
        self.run_test("Integrity: PPTX all slides + notes", "integrity", test_pptx_integrity)

        # 12d. HTML structure preservation
        def test_html_integrity():
            text = self.processor.extract_text(str(BASE / "sample.html"))
            expected = ["Main Title", "Section 1", "Section 2", "Item A", "Alice", "Bob", "hello"]
            found = {w: w.lower() in text.lower() for w in expected}
            missing = [w for w, f in found.items() if not f]
            return {
                "all_found": len(missing) == 0,
                "missing": missing,
                "text_preview": text[:500].replace("\n", "\\n"),
            }
        self.run_test("Integrity: HTML content preservation", "integrity", test_html_integrity)

        # 12e. Multi-page PDF page markers
        def test_pdf_pages():
            text = self.processor.extract_text(str(BASE / "multipage.pdf"))
            return {
                "text_length": len(text),
                "has_multiple_pages": len(text) > 1000,
                "page_count_estimate": text.lower().count("page"),
                "text_preview": text[:500].replace("\n", "\\n"),
            }
        self.run_test("Integrity: Multi-page PDF", "integrity", test_pdf_pages)

        # 12f. RTF table content
        def test_rtf_integrity():
            text = self.processor.extract_text(str(BASE / "table.rtf"))
            return {
                "text_length": len(text),
                "has_alice": "Alice" in text,
                "has_bob": "Bob" in text,
                "has_table_header": "Name" in text,
                "preview": text[:500].replace("\n", "\\n"),
            }
        self.run_test("Integrity: RTF table content", "integrity", test_rtf_integrity)

        # 12g. XLS content
        def test_xls_integrity():
            if not (BASE / "sample.xls").exists():
                return {"skipped": True}
            text = self.processor.extract_text(str(BASE / "sample.xls"))
            return {
                "text_length": len(text),
                "has_alice": "Alice" in text,
                "has_bob": "Bob" in text,
                "preview": text[:500].replace("\n", "\\n"),
            }
        self.run_test("Integrity: XLS content", "integrity", test_xls_integrity)

        # 12h. Chunking preserves all content
        def test_chunk_completeness():
            text = self.processor.extract_text(str(BASE / "sample.docx"))
            chunk_result = self.processor.extract_chunks(str(BASE / "sample.docx"), chunk_size=500)
            chunks = chunk_result.chunks
            reassembled = " ".join(
                c if isinstance(c, str) else c.content for c in chunks
            )
            # Check key words are preserved
            keywords = ["Alice", "Bob", "Charlie", "Section", "Conclusion"]
            found_in_text = {w: w in text for w in keywords}
            found_in_chunks = {w: w in reassembled for w in keywords}
            missing = [w for w in keywords if found_in_text[w] and not found_in_chunks[w]]
            return {
                "original_len": len(text),
                "chunks_total_len": len(reassembled),
                "missing_after_chunking": missing,
                "integrity_ok": len(missing) == 0,
            }
        self.run_test("Integrity: Chunking content preservation", "integrity", test_chunk_completeness)

    # ============================================================
    # 13. SUPPORTED EXTENSIONS / REGISTRY
    # ============================================================
    def test_registry(self):
        print("\n" + "="*70)
        print("SECTION 13: Handler Registry & Extension Support")
        print("="*70)

        def test_extensions():
            exts = self.processor.supported_extensions
            expected_exts = [
                ".pdf", ".docx", ".doc", ".pptx", ".ppt",
                ".xlsx", ".xls", ".csv", ".tsv",
                ".html", ".htm", ".rtf",
                ".txt", ".md", ".py", ".json", ".yaml", ".yml",
                ".xml", ".ini", ".log",
                ".hwp", ".hwpx",
                ".jpg", ".jpeg", ".png", ".gif", ".bmp",
            ]
            found = {ext: ext in exts for ext in expected_exts}
            missing = [ext for ext, f in found.items() if not f]
            return {
                "total_supported": len(exts),
                "expected_found": len(expected_exts) - len(missing),
                "missing": missing,
                "all_extensions": sorted(exts),
            }
        self.run_test("Registry: Supported extensions", "registry", test_extensions)

        def test_is_supported():
            results = {}
            for ext in [".pdf", ".docx", ".csv", ".html", ".xyz", ".zzz"]:
                results[ext] = self.processor.is_supported(ext)
            return results
        self.run_test("Registry: is_supported()", "registry", test_is_supported)

    # ============================================================
    # 14. PERFORMANCE TESTS
    # ============================================================
    def test_performance(self):
        print("\n" + "="*70)
        print("SECTION 14: Performance Tests")
        print("="*70)

        # 14a. Large CSV processing time
        def test_large_csv_perf():
            start = time.perf_counter()
            text = self.processor.extract_text(str(BASE / "large.csv"))
            elapsed = time.perf_counter() - start
            return {
                "elapsed_ms": round(elapsed * 1000, 2),
                "text_length": len(text),
                "chars_per_ms": round(len(text) / (elapsed * 1000), 2) if elapsed > 0 else 0,
            }
        self.run_test("Performance: Large CSV (1000 rows)", "performance", test_large_csv_perf)

        # 14b. Large XLSX processing time
        def test_large_xlsx_perf():
            start = time.perf_counter()
            text = self.processor.extract_text(str(BASE / "large.xlsx"))
            elapsed = time.perf_counter() - start
            return {
                "elapsed_ms": round(elapsed * 1000, 2),
                "text_length": len(text),
            }
        self.run_test("Performance: Large XLSX (500 rows)", "performance", test_large_xlsx_perf)

        # 14c. Long DOCX processing
        def test_long_docx_perf():
            start = time.perf_counter()
            text = self.processor.extract_text(str(BASE / "long.docx"))
            elapsed = time.perf_counter() - start
            return {
                "elapsed_ms": round(elapsed * 1000, 2),
                "text_length": len(text),
            }
        self.run_test("Performance: Long DOCX (50 chapters)", "performance", test_long_docx_perf)

        # 14d. Multi-page PDF
        def test_multipage_pdf_perf():
            start = time.perf_counter()
            text = self.processor.extract_text(str(BASE / "multipage.pdf"))
            elapsed = time.perf_counter() - start
            return {
                "elapsed_ms": round(elapsed * 1000, 2),
                "text_length": len(text),
            }
        self.run_test("Performance: Multi-page PDF (10 pages)", "performance", test_multipage_pdf_perf)

        # 14e. Chunking performance
        def test_chunking_perf():
            text = "Lorem ipsum dolor sit amet. " * 5000  # ~140K chars
            chunker = TextChunker(ProcessingConfig())
            start = time.perf_counter()
            chunks = chunker.chunk(text, chunk_size=1000, chunk_overlap=200)
            elapsed = time.perf_counter() - start
            return {
                "elapsed_ms": round(elapsed * 1000, 2),
                "input_chars": len(text),
                "chunk_count": len(chunks),
            }
        self.run_test("Performance: Chunking 140K chars", "performance", test_chunking_perf)

    # ============================================================
    # MAIN EXECUTION
    # ============================================================
    def run_all(self):
        print("="*70)
        print("CONTEXTIFY v0.2.6 DEEP INTEGRATION TEST")
        print(f"Test files directory: {BASE}")
        print("="*70)

        self.test_extract_text_all_formats()
        self.test_process_all_formats()
        self.test_extract_chunks()
        self.test_chunking_strategies()
        self.test_table_handling()
        self.test_metadata_extraction()
        self.test_tag_system()
        self.test_edge_cases()
        self.test_configuration()
        self.test_async_processor()
        self.test_cached_processor()
        self.test_content_integrity()
        self.test_registry()
        self.test_performance()

        # Summary
        print("\n" + "="*70)
        print("TEST SUMMARY")
        print("="*70)

        categories = {}
        for r in self.results:
            if r.category not in categories:
                categories[r.category] = {"PASS": 0, "FAIL": 0, "ERROR": 0, "WARN": 0, "SKIP": 0}
            categories[r.category][r.status] += 1

        total_pass = sum(1 for r in self.results if r.status == "PASS")
        total_fail = sum(1 for r in self.results if r.status == "FAIL")
        total_error = sum(1 for r in self.results if r.status == "ERROR")
        total_warn = sum(1 for r in self.results if r.status == "WARN")
        total_skip = sum(1 for r in self.results if r.status == "SKIP")
        total = len(self.results)

        for cat, counts in sorted(categories.items()):
            line = f"  {cat:25s} "
            line += f"PASS:{counts['PASS']:3d}  "
            if counts['FAIL']: line += f"FAIL:{counts['FAIL']:3d}  "
            if counts['ERROR']: line += f"ERROR:{counts['ERROR']:3d}  "
            if counts['WARN']: line += f"WARN:{counts['WARN']:3d}  "
            if counts['SKIP']: line += f"SKIP:{counts['SKIP']:3d}  "
            print(line)

        print(f"\n  TOTAL: {total} tests | PASS: {total_pass} | FAIL: {total_fail} | ERROR: {total_error} | WARN: {total_warn} | SKIP: {total_skip}")
        print(f"  Pass Rate: {total_pass/total*100:.1f}%")

        # Save detailed results
        results_json = []
        for r in self.results:
            results_json.append({
                "name": r.name,
                "category": r.category,
                "status": r.status,
                "duration_ms": r.duration_ms,
                "details": r.details[:500] if r.details else "",
                "error": r.error[:500] if r.error else "",
                "data": r.data,
            })

        with open(RESULTS / "deep_test_results.json", "w", encoding="utf-8") as f:
            json.dump(results_json, f, indent=2, ensure_ascii=False, default=str)

        print(f"\n  Detailed results saved to: {RESULTS / 'deep_test_results.json'}")

        # Save errors list
        errors = [r for r in self.results if r.status in ("ERROR", "FAIL")]
        if errors:
            print(f"\n  ERRORS/FAILURES ({len(errors)}):")
            for e in errors:
                print(f"    - [{e.status}] {e.name}: {e.error[:150]}")

        return self.results


if __name__ == "__main__":
    runner = DeepTestRunner()
    runner.run_all()
