# Contextify v0.2.6 Bug Fix Report

**Date:** 2026-04-07  
**Based on:** [DEEP_TEST_REPORT_v0.2.6.md](DEEP_TEST_REPORT_v0.2.6.md)

---

## Summary

| Metric | Before | After |
|--------|--------|-------|
| **Custom Integration Tests** | 106/110 (96.4%) | **110/110 (100%)** |
| **Existing Unit Tests** | 476/476 (100%) | **476/476 (100%)** |
| **SVG Registration Warning** | Every instantiation | **Eliminated** |
| **Bugs Fixed** | 0 | **6** |

---

## Fixes Applied

### FIX #1: Empty File Handling (BUG #1 - Critical)

**Problem:** 0-byte TXT/CSV files caused `ConversionError` crash at validation stage.

**Root Cause:** `BaseConverter.validate()`, `TextConverter.validate()`, and `CsvConverter.validate()` all rejected empty `file_data` (length == 0).

**Changes:**

| File | Change |
|------|--------|
| `contextifier/pipeline/converter.py` | `BaseConverter.validate()` now returns `True` always (empty file handling deferred to `convert()`) |
| `contextifier/handlers/text/converter.py` | `validate()` returns `True`; `convert()` returns empty `TextConvertedData` for empty files instead of raising |
| `contextifier/handlers/csv/converter.py` | `validate()` returns `True`; `convert()` returns empty `CsvConvertedData` for empty files instead of raising |
| `tests/unit/test_security.py` | `test_empty_file_rejected` -> `test_empty_file_returns_empty_text` (expects empty string return) |

**Verification:**
```python
proc.extract_text("empty.txt")  # Returns "" (was: ConversionError)
proc.extract_text("empty.csv")  # Returns metadata-only text (was: ConversionError)
```

---

### FIX #2: SVG Extension Double-Registration (BUG #3 & #6 - Medium)

**Problem:** SVG was registered in both `TextHandler` (as XML text) and `ImageFileHandler` (as image), causing a warning on every instantiation. Additionally, `.svg` was missing from `supported_extensions`.

**Root Cause:** `_TEXT_EXTENSIONS` in `text/handler.py` and `IMAGE_EXTENSIONS` in `image/_constants.py` both included `"svg"`.

**Change:**

| File | Change |
|------|--------|
| `contextifier/handlers/image/_constants.py` | Removed `"svg"` from `IMAGE_EXTENSIONS` (SVG stays in TextHandler as it's XML-based) |

**Verification:**
```python
# No more warning on instantiation
proc = DocumentProcessor()  # Clean, no SVG warning
proc.is_supported(".svg")   # Returns True (was: False)
```

---

### FIX #3: Empty Text Chunking (BUG #4 - Medium)

**Problem:** `TextChunker.chunk("")` returned `[""]` (list with one empty string) instead of `[]` (empty list).

**Root Cause:** Early return in `chunker.py` line 111-112 returned `[""]`.

**Changes:**

| File | Change |
|------|--------|
| `contextifier/chunking/chunker.py` | `return [""]` -> `return []` for empty/whitespace-only text |
| `tests/unit/chunking/test_chunker.py` | Updated 2 tests to expect `[]` instead of `[""]` |

**Verification:**
```python
chunker.chunk("")        # Returns [] (was: [""])
chunker.chunk("  \n  ")  # Returns [] (was: [""])
```

---

### FIX #4: Error Priority Order (ISSUE #4)

**Problem:** Processing `nonexistent.zzz` raised `FileNotFoundError` instead of `UnsupportedFormatError`. Format validation should happen before file existence check.

**Changes:**

| File | Change |
|------|--------|
| `contextifier/document_processor.py` | In `extract_text()` and `process()`: moved extension resolution and format support check **before** file existence check |

**Verification:**
```python
proc.extract_text("nonexistent.zzz")  # UnsupportedFormatError (was: FileNotFoundError)
proc.extract_text("nonexistent.pdf")  # FileNotFoundError (unchanged, .pdf is supported)
```

---

### FIX #5: ChunkResult Strategy Field (ISSUE #5)

**Problem:** `ChunkResult` did not expose which chunking strategy was selected, making debugging difficult.

**Changes:**

| File | Change |
|------|--------|
| `contextifier/document_processor.py` | Added `strategy: Optional[str] = None` field to `ChunkResult` dataclass |
| `contextifier/chunking/chunker.py` | Added `last_strategy_name` property to `TextChunker`; tracks strategy used in each `chunk()` call |
| `contextifier/document_processor.py` | `extract_chunks()` now passes `self._chunker.last_strategy_name` to `ChunkResult.strategy` |

**Verification:**
```python
result = proc.extract_chunks("file.txt", chunk_size=500)
result.strategy  # "plain"

result = proc.extract_chunks("file.csv", chunk_size=500)
result.strategy  # "table"

result = proc.extract_chunks("file.pdf", chunk_size=500)
result.strategy  # "protected" or "page"
```

---

## Files Modified

| File | Lines Changed | Type |
|------|--------------|------|
| `contextifier/pipeline/converter.py` | ~5 | Bug fix |
| `contextifier/handlers/text/converter.py` | ~10 | Bug fix |
| `contextifier/handlers/csv/converter.py` | ~10 | Bug fix |
| `contextifier/handlers/image/_constants.py` | ~3 | Bug fix |
| `contextifier/chunking/chunker.py` | ~10 | Bug fix + Feature |
| `contextifier/document_processor.py` | ~25 | Bug fix + Feature |
| `tests/unit/test_security.py` | ~5 | Test update |
| `tests/unit/chunking/test_chunker.py` | ~4 | Test update |

**Total: 8 files, ~72 lines changed**

---

## Test Results After Fixes

### Existing Unit Tests
```
476 passed, 6 warnings in 3.15s
```

### Custom Deep Integration Tests
```
110 tests | PASS: 110 | FAIL: 0 | ERROR: 0 | WARN: 0 | SKIP: 0
Pass Rate: 100.0%
```

### Specific Improvements

| Test | Before | After |
|------|--------|-------|
| Edge: Empty TXT file | ERROR | PASS |
| Edge: Empty CSV file | ERROR | PASS |
| Chunking: Empty text | PASS (wrong result) | PASS (correct result) |
| Edge: Unsupported format | PASS (wrong error type) | PASS (correct error type) |
| SVG registration warning | Printed every time | Eliminated |

---

## Remaining Known Issues (Not Fixed)

These are minor and not blocking:

1. **XLSX processing speed** (~380-420ms): Inherent openpyxl initialization cost. Consider `read_only` mode for large files.
2. **PDF processing speed** (~77ms/page): pdfplumber overhead. Consider PyMuPDF for faster processing.
3. **Windows console encoding**: Korean metadata labels may cause `UnicodeEncodeError` on cp949 consoles. This is a Windows limitation.
4. **BUG #2 (position metadata)**: Re-assessed as **by-design**. `ChunkResult.chunks` always returns `List[str]` for backwards compatibility. `ChunkResult.chunks_with_metadata` provides `List[Chunk]` with position metadata when `include_position_metadata=True`. The `has_metadata` property indicates availability.
