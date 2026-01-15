"""
HWPX (ZIP/XML 기반) 문서 처리 모듈

리팩터링된 구조:
- 기능별 로직은 hwpx_helper/ 모듈로 분리
- HwpxProcessor는 조합 및 조율 역할
"""
import zipfile
import logging
from typing import Dict, Any

from .hwp_helper import MetadataHelper
from .hwpx_helper import (
    # Metadata
    extract_hwpx_metadata,
    parse_bin_item_map,
    # Section
    parse_hwpx_section,
    # Image
    process_hwpx_images,
    get_remaining_images,
    # Chart
    extract_charts_from_hwpx,
)

# Try to import OCR handler if available in the project
try:
    from libs.core.ocr_legacy import convert_image_to_text
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

logger = logging.getLogger("document-processor")


class HwpxProcessor:
    """HWPX (ZIP/XML 기반 한글 문서) 처리 클래스"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}

    async def process_hwpx(self, file_path: str, app_db=None, extract_default_metadata: bool = True) -> str:
        """
        Process HWPX (Zip/XML) file.
        Extracts text, tables (as HTML/Markdown), and images inline.

        Args:
            file_path: HWPX 파일 경로
            app_db: 데이터베이스 연결
            extract_default_metadata: 기본 메타데이터 추출 여부 (기본값: True)
        """
        text_content = []

        try:
            if not zipfile.is_zipfile(file_path):
                logger.error(f"Not a valid Zip file: {file_path}")
                return ""

            with zipfile.ZipFile(file_path, 'r') as zf:
                # Extract metadata first (using helper)
                if extract_default_metadata:
                    metadata = extract_hwpx_metadata(zf)
                    metadata_text = MetadataHelper.format_metadata(metadata)
                    if metadata_text:
                        text_content.append(metadata_text)
                        text_content.append("")  # Empty line after metadata

                # 0. Parse Manifest (content.hpf) to map BinItem ID to file path (using helper)
                bin_item_map = parse_bin_item_map(zf)

                # 1. Parse Content Sections (using helper)
                section_files = [f for f in zf.namelist() if f.startswith("Contents/section") and f.endswith(".xml")]
                section_files.sort(key=lambda x: int(x.replace("Contents/section", "").replace(".xml", "")))

                processed_images = set()

                for sec_file in section_files:
                    with zf.open(sec_file) as f:
                        xml_content = f.read()
                        # Pass zf and map to parse section for inline images
                        section_text = parse_hwpx_section(xml_content, zf, bin_item_map, app_db, processed_images)
                        text_content.append(section_text)

                # 2. Process Remaining Images (BinData) that were not inline (using helper)
                if OCR_AVAILABLE:
                    remaining_images = get_remaining_images(zf, processed_images)

                    if remaining_images:
                        image_text = await process_hwpx_images(zf, remaining_images, app_db=app_db)
                        if image_text:
                            text_content.append("\n\n=== Extracted Images (Not Inline) ===\n")
                            text_content.append(image_text)

                # 3. Extract Charts (using helper)
                chart_texts = extract_charts_from_hwpx(zf, app_db=app_db)
                if chart_texts:
                    text_content.extend(chart_texts)

        except Exception as e:
            logger.error(f"Error processing HWPX file: {e}")
            return f"Error processing HWPX file: {str(e)}"

        final_result = "\n".join(text_content)
        logger.info(f"--- Final Extracted Text ({file_path}) ---\n{final_result}\n-----------------------------------")
        return final_result


async def extract_text_from_hwpx(file_path: str, config: Dict[str, Any], app_db=None, extract_default_metadata: bool = True) -> str:
    """HWPX 파일에서 텍스트 추출 (외부 호출용 함수)"""
    processor = HwpxProcessor(config)
    return await processor.process_hwpx(file_path, app_db=app_db, extract_default_metadata=extract_default_metadata)
