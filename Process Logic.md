# Contextify Processing Flow

---

## Main Flow

```
User calls: processor.extract_chunks(file_path)
                    │
                    ▼
         DocumentProcessor.extract_chunks()
                    │
                    ├─► extract_text()
                    │       │
                    │       ├─► _create_current_file(file_path)
                    │       ├─► _get_handler(extension)
                    │       ├─► handler.extract_text(current_file)
                    │       └─► OCR processing (optional)
                    │
                    └─► chunk_text()
                            │
                            └─► create_chunks()
```

---

## PDF Handler Flow

```
PDFHandler.extract_text(current_file)
    │
    ├─► file_converter.convert()                        [INTERFACE: PDFFileConverter]
    │       └─► Binary → fitz.Document
    │
    ├─► metadata_extractor.extract()                    [INTERFACE: PDFMetadataExtractor]
    ├─► metadata_extractor.format()                     [INTERFACE: PDFMetadataExtractor]
    │
    └─► For each page:
            │
            ├─► page_tag_processor.create_page_tag()    [INTERFACE: PageTagProcessor]
            │
            ├─► format_image_processor.extract_images_from_page()   [INTERFACE: PDFImageProcessor]
            │
            ├─► extract_text_blocks()                   [FUNCTION]
            │
            ├─► extract_all_tables()                    [FUNCTION]
            │
            └─► merge_page_elements()                   [FUNCTION]
```

---

## DOCX Handler Flow

```
DOCXHandler.extract_text(current_file)
    │
    ├─► file_converter.convert()                        [INTERFACE: DOCXFileConverter]
    │       └─► Binary → docx.Document
    │
    ├─► metadata_extractor.extract()                    [INTERFACE: DOCXMetadataExtractor]
    ├─► metadata_extractor.format()                     [INTERFACE: DOCXMetadataExtractor]
    │
    ├─► chart_extractor.extract_all_from_file()         [INTERFACE: DOCXChartExtractor]
    │
    └─► For each element in doc.body:
            │
            ├─► process_paragraph_element()             [FUNCTION]
            │       │
            │       ├─► format_image_processor.process_drawing_element()  [INTERFACE: DOCXImageProcessor]
            │       │
            │       └─► format_image_processor.extract_from_pict()        [INTERFACE: DOCXImageProcessor]
            │
            └─► process_table_element()                 [FUNCTION]
```

---

## DOC Handler Flow

```
DOCHandler.extract_text(current_file)
    │
    ├─► file_converter.convert()                        [INTERFACE: DOCFileConverter]
    │       │
    │       ├─► _detect_format() → DocFormat (RTF/OLE/HTML/DOCX)
    │       │
    │       ├─► RTF: _convert_rtf() → RTFDocument       [see RTF Handler Flow below]
    │       ├─► OLE: _convert_ole() → olefile.OleFileIO
    │       ├─► HTML: _convert_html() → BeautifulSoup
    │       └─► DOCX: _convert_docx() → docx.Document
    │
    ├─► RTF format detected:
    │       └─► _delegate_to_rtf_handler()              [DELEGATION]
    │               └─► RTFHandler.extract_from_rtf_document()
    │
    ├─► OLE format detected:
    │       ├─► _extract_ole_metadata()                 [INTERNAL]
    │       ├─► _extract_ole_text()                     [INTERNAL]
    │       └─► _extract_ole_images()                   [INTERNAL]
    │
    ├─► HTML format detected:
    │       ├─► _extract_html_metadata()                [INTERNAL]
    │       └─► BeautifulSoup parsing                   [EXTERNAL LIBRARY]
    │
    └─► DOCX format detected:
            └─► DOCXHandler delegation                  [DELEGATION]
```

---

## RTF Handler Flow

**구조**: Converter는 pass-through, Preprocessor에서 binary 처리, Handler에서 순차적 처리.

```
RTFHandler.extract_text(current_file)
    │
    ├─► file_converter.convert()                        [INTERFACE: RTFFileConverter]
    │       └─► Pass-through (returns raw bytes)
    │
    ├─► preprocessor.preprocess()                       [INTERFACE: RTFPreprocessor]
    │       │
    │       ├─► \binN tag processing (skip binary data)
    │       ├─► \pict group image extraction
    │       └─► Returns PreprocessedData (clean_content, image_tags, encoding)
    │
    ├─► decode_content()                                [FUNCTION: rtf_decoder]
    │       └─► bytes → string with detected encoding
    │
    ├─► Build RTFConvertedData                          [DATACLASS]
    │
    └─► _extract_from_converted()                       [INTERNAL]
            │
            ├─► metadata_extractor.extract()            [INTERFACE: RTFMetadataExtractor]
            ├─► metadata_extractor.format()
            │
            ├─► extract_tables_with_positions()         [FUNCTION: rtf_table_extractor]
            │
            ├─► extract_inline_content()                [FUNCTION: rtf_content_extractor]
            │
            └─► Build result string
```

---

## Excel Handler Flow (XLSX)

```
ExcelHandler.extract_text(current_file) [XLSX]
    │
    ├─► file_converter.convert()                        [INTERFACE: ExcelFileConverter]
    │       └─► Binary → openpyxl.Workbook
    │
    ├─► metadata_extractor.extract()                    [INTERFACE: XLSXMetadataExtractor]
    ├─► metadata_extractor.format()                     [INTERFACE: XLSXMetadataExtractor]
    │
    ├─► chart_extractor.extract_all_from_file()         [INTERFACE: ExcelChartExtractor]
    │
    ├─► format_image_processor.extract_images_from_xlsx()  [INTERFACE: ExcelImageProcessor]
    │
    └─► For each sheet:
            │
            ├─► page_tag_processor.create_sheet_tag()   [INTERFACE: PageTagProcessor]
            │
            ├─► format_image_processor.get_sheet_images()  [INTERFACE: ExcelImageProcessor]
            │
            ├─► _process_xlsx_sheet()                   [INTERNAL]
            │
            └─► format_image_processor.process_sheet_images()  [INTERFACE: ExcelImageProcessor]
```

---

## Excel Handler Flow (XLS)

```
ExcelHandler.extract_text(current_file) [XLS]
    │
    ├─► file_converter.convert()                        [INTERFACE: XLSFileConverter]
    │       └─► Binary → xlrd.Book
    │
    ├─► metadata_extractor.extract()                    [INTERFACE: XLSMetadataExtractor]
    ├─► metadata_extractor.format()                     [INTERFACE: XLSMetadataExtractor]
    │
    └─► For each sheet:
            │
            ├─► page_tag_processor.create_sheet_tag()   [INTERFACE: PageTagProcessor]
            │
            └─► _process_xls_sheet()                    [INTERNAL]
```

---

## PPT Handler Flow

```
PPTHandler.extract_text(current_file)
    │
    ├─► file_converter.convert()                        [INTERFACE: PPTFileConverter]
    │       └─► Binary → pptx.Presentation
    │
    ├─► metadata_extractor.extract()                    [INTERFACE: PPTMetadataExtractor]
    ├─► metadata_extractor.format()                     [INTERFACE: PPTMetadataExtractor]
    │
    ├─► chart_extractor.extract_all_from_file()         [INTERFACE: PPTChartExtractor]
    │
    └─► For each slide:
            │
            ├─► page_tag_processor.create_slide_tag()   [INTERFACE: PageTagProcessor]
            │
            ├─► format_image_processor.extract_from_slide()  [INTERFACE: PPTImageProcessor]
            │
            └─► _process_shapes()                       [INTERNAL]
```

---

## HWP Handler Flow

```
HWPHandler.extract_text(current_file)
    │
    ├─► file_converter.convert()                        [INTERFACE: HWPFileConverter]
    │       └─► Binary → olefile.OleFileIO
    │
    ├─► metadata_extractor.extract()                    [INTERFACE: HWPMetadataExtractor]
    ├─► metadata_extractor.format()                     [INTERFACE: HWPMetadataExtractor]
    │
    ├─► chart_extractor.extract_all_from_file()         [INTERFACE: HWPChartExtractor]
    │
    ├─► _parse_docinfo(ole)                             [INTERNAL]
    │       └─► parse_doc_info()                        [FUNCTION]
    │
    ├─► _extract_body_text(ole)                         [INTERNAL]
    │       │
    │       └─► _process_picture(record)                [INTERNAL]
    │               │
    │               ├─► format_image_processor.extract_bindata_index()     [INTERFACE: HWPImageProcessor]
    │               ├─► format_image_processor.find_bindata_stream()       [INTERFACE: HWPImageProcessor]
    │               └─► format_image_processor.extract_and_save_image()    [INTERFACE: HWPImageProcessor]
    │
    └─► format_image_processor.process_images_from_bindata()  [INTERFACE: HWPImageProcessor]
```

---

## HWPX Handler Flow

```
HWPXHandler.extract_text(current_file)
    │
    ├─► file_converter.convert()                        [INTERFACE: HWPXFileConverter]
    │       └─► Binary → zipfile.ZipFile
    │
    ├─► metadata_extractor.extract()                    [INTERFACE: HWPXMetadataExtractor]
    ├─► metadata_extractor.format()                     [INTERFACE: HWPXMetadataExtractor]
    │
    ├─► chart_extractor.extract_all_from_file()         [INTERFACE: HWPXChartExtractor]
    │
    ├─► format_image_processor.process_from_zip()       [INTERFACE: HWPXImageProcessor]
    │
    ├─► For each section:
    │       │
    │       └─► parse_hwpx_section()                    [FUNCTION]
    │               │
    │               ├─► format_image_processor.process_images()  [INTERFACE: HWPXImageProcessor]
    │               │
    │               └─► parse_hwpx_table()              [FUNCTION]
    │
    └─► format_image_processor.get_remaining_images()   [INTERFACE: HWPXImageProcessor]
        format_image_processor.process_remaining_images()  [INTERFACE: HWPXImageProcessor]
```

---

## CSV Handler Flow

```
CSVHandler.extract_text(current_file)
    │
    ├─► file_converter.convert()                        [INTERFACE: CSVFileConverter]
    │       └─► Binary → Text (with encoding detection)
    │
    ├─► metadata_extractor.extract()                    [INTERFACE: CSVMetadataExtractor]
    ├─► metadata_extractor.format()                     [INTERFACE: CSVMetadataExtractor]
    │
    ├─► CSVParser.parse()                               [FUNCTION]
    │
    └─► CSVTable.to_html()                              [FUNCTION]
```

---

## Text Handler Flow

```
TextHandler.extract_text(current_file)
    │
    ├─► file_converter.convert()                        [INTERFACE: TextFileConverter]
    │       └─► Binary → Text (with encoding detection)
    │
    ├─► metadata_extractor.extract()                    [INTERFACE: TextMetadataExtractor]
    ├─► metadata_extractor.format()                     [INTERFACE: TextMetadataExtractor]
    │
    └─► decode_text()                                   [FUNCTION]
```

---

## HTML Handler Flow

```
HTMLReprocessor.extract_text(current_file)
    │
    ├─► file_converter.convert()                        [INTERFACE: HTMLFileConverter]
    │       └─► Binary → BeautifulSoup
    │
    └─► BeautifulSoup parsing                           [EXTERNAL LIBRARY]
```

---

## Image File Handler Flow

```
ImageFileHandler.extract_text(current_file)
    │
    ├─► file_converter.convert()                        [INTERFACE: ImageFileConverter]
    │       └─► Binary → Binary (pass-through)
    │
    ├─► metadata_extractor.extract()                    [INTERFACE: ImageFileMetadataExtractor]
    ├─► metadata_extractor.format()                     [INTERFACE: ImageFileMetadataExtractor]
    │
    └─► format_image_processor.save_image()             [INTERFACE: ImageFileImageProcessor]
```

---

## Chunking Flow

```
chunk_text(text, chunk_size, chunk_overlap)
    │
    └─► create_chunks()                                 [FUNCTION]
            │
            ├─► _extract_document_metadata()            [FUNCTION]
            │
            ├─► Detect file type:
            │       │
            │       ├─► Table-based (xlsx, xls, csv):
            │       │       └─► chunk_multi_sheet_content()  [FUNCTION]
            │       │
            │       ├─► Text with page markers:
            │       │       └─► chunk_by_pages()        [FUNCTION]
            │       │
            │       └─► Plain text:
            │               └─► chunk_plain_text()      [FUNCTION]
            │
            └─► _prepend_metadata_to_chunks()           [FUNCTION]
```

---

## Interface Integration Summary

```
┌─────────────┬─────────────────────┬─────────────────────┬─────────────────────┬─────────────────────┐
│ Handler     │ FileConverter       │ MetadataExtractor   │ ChartExtractor      │ FormatImageProcessor│
├─────────────┼─────────────────────┼─────────────────────┼─────────────────────┼─────────────────────┤
│ PDF         │ ✅ PDFFile           │ ✅ PDFMetadata       │ ✅ PDFChart          │ ✅ PDFImage          │
│ DOCX        │ ✅ DOCXFile          │ ✅ DOCXMetadata      │ ✅ DOCXChart         │ ✅ DOCXImage         │
│ DOC         │ ✅ DOCFile           │ ❌ NullMetadata      │ ❌ NullChart         │ ✅ DOCImage          │
│ RTF         │ ✅ RTFFile (pass)    │ ✅ RTFMetadata       │ ❌ NullChart         │ ✅ RTFPreprocessor   │
│ XLSX        │ ✅ XLSXFile          │ ✅ XLSXMetadata      │ ✅ ExcelChart        │ ✅ ExcelImage        │
│ XLS         │ ✅ XLSFile           │ ✅ XLSMetadata       │ ❌ NullChart         │ ✅ ExcelImage        │
│ PPT/PPTX    │ ✅ PPTFile           │ ✅ PPTMetadata       │ ✅ PPTChart          │ ✅ PPTImage          │
│ HWP         │ ✅ HWPFile           │ ✅ HWPMetadata       │ ✅ HWPChart          │ ✅ HWPImage          │
│ HWPX        │ ✅ HWPXFile          │ ✅ HWPXMetadata      │ ✅ HWPXChart         │ ✅ HWPXImage         │
│ CSV         │ ✅ CSVFile           │ ✅ CSVMetadata       │ ❌ NullChart         │ ❌ None              │
│ TXT/MD/JSON │ ✅ TextFile          │ ✅ TextMetadata      │ ❌ NullChart         │ ❌ None              │
│ HTML        │ ✅ HTMLFile          │ ❌ None              │ ❌ NullChart         │ ❌ None              │
│ Image Files │ ✅ ImageFile (pass)  │ ✅ ImageFileMeta     │ ❌ NullChart         │ ✅ ImageFileImage    │
└─────────────┴─────────────────────┴─────────────────────┴─────────────────────┴─────────────────────┘

✅ = Interface implemented correctly
❌ = Not applicable / NullExtractor
```

---

## Remaining Function-Based Components

```
┌─────────────┬────────────────────────────────────────────────────────────┐
│ Handler     │ Function-Based Components                                  │
├─────────────┼────────────────────────────────────────────────────────────┤
│ PDF         │ extract_text_blocks(), extract_all_tables(),              │
│             │ merge_page_elements()                                      │
├─────────────┼────────────────────────────────────────────────────────────┤
│ DOCX        │ process_paragraph_element(), process_table_element()      │
├─────────────┼────────────────────────────────────────────────────────────┤
│ DOC         │ Format detection, OLE/HTML/DOCX delegation                │
├─────────────┼────────────────────────────────────────────────────────────┤
│ RTF         │ decode_content() (rtf_decoder.py)                         │
│             │ extract_tables_with_positions() (rtf_table_extractor.py)  │
│             │ extract_inline_content() (rtf_content_extractor.py)       │
│             │ clean_rtf_text() (rtf_text_cleaner.py)                    │
├─────────────┼────────────────────────────────────────────────────────────┤
│ HWP         │ parse_doc_info(), parse_table(), decompress_section()     │
├─────────────┼────────────────────────────────────────────────────────────┤
│ HWPX        │ parse_hwpx_section(), parse_hwpx_table()                  │
├─────────────┼────────────────────────────────────────────────────────────┤
│ CSV         │ CSVParser, CSVTable                                       │
├─────────────┼────────────────────────────────────────────────────────────┤
│ Chunking    │ create_chunks(), chunk_by_pages(), chunk_plain_text(),    │
│             │ chunk_multi_sheet_content(), chunk_large_table()          │
└─────────────┴────────────────────────────────────────────────────────────┘
```
