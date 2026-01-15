# your_package/document_processor/document_processor.py
import logging, re, bisect
from pathlib import Path
from typing import Any, Dict, List, Optional

from libs.core.functions.utils import (clean_text, clean_code_text, sanitize_text_for_json)
from libs.chunking.chunking import (split_text_preserving_html_blocks, reconstruct_text_from_chunks,
                       find_overlap_length, chunk_code_text, estimate_chunks_count)
from libs.core.processor.pdf_handler import extract_text_from_pdf
from libs.core.processor.pdf_legacy.pdf_handler_ocr import extract_text_from_pdf_ocr
from libs.core.processor.pdf_handler_v4 import extract_text_from_pdf_v4
from libs.core.processor.docx_handler import extract_text_from_docx
from libs.core.processor.doc_handler import extract_text_from_doc
from libs.core.processor.ppt_handler import extract_text_from_ppt
from libs.core.processor.excel_handler import extract_text_from_excel
from libs.core.processor.csv_handler import extract_text_from_csv
from libs.core.processor.text_handler import extract_text_from_text_file
from libs.core.processor.hwp_processor import extract_text_from_hwp
from libs.core.processor.hwpx_processor import extract_text_from_hwpx

logger = logging.getLogger("document-processor")

class DocumentProcessor:
    """
    원본 클래스의 공개 메서드/시그니처/동작을 그대로 유지.
    내부 구현은 모듈로 분리(위임).
    """
    def __init__(self, config_composer = None):
        self.config_composer = config_composer

        # 타입 세트(원본과 동일)
        self.document_types = ['pdf', 'docx', 'doc', 'pptx', 'ppt', 'hwp', 'hwpx']
        self.text_types = ['txt','md','markdown','rtf']
        self.code_types = ['py','js','ts','java','cpp','c','h','cs','go','rs',
                           'php','rb','swift','kt','scala','dart','r','sql',
                           'html','css','jsx','tsx','vue','svelte']
        self.config_types = ['json','yaml','yml','xml','toml','ini','cfg','conf','properties','env']
        self.data_types   = ['csv','tsv','xlsx','xls']
        self.script_types = ['sh','bat','ps1','zsh','fish']
        self.log_types    = ['log']
        self.web_types    = ['htm','xhtml']
        self.image_types  = ['jpg','jpeg','png','gif','bmp','webp']

        self.supported_types = ( self.document_types + self.text_types + self.code_types +
                                 self.config_types + self.data_types + self.script_types +
                                 self.log_types + self.web_types + self.image_types )

        # 가용 라이브러리 체크(원본과 동일한 경고/동작)
        try:
            from openpyxl import load_workbook  # noqa
            OPENPYXL_AVAILABLE = True
        except Exception:
            OPENPYXL_AVAILABLE = False
        try:
            import xlrd  # noqa
            XLRD_AVAILABLE = True
        except Exception:
            XLRD_AVAILABLE = False
        try:
            from langchain_openai import ChatOpenAI  # noqa
            LANGCHAIN_OPENAI_AVAILABLE = True
        except Exception:
            LANGCHAIN_OPENAI_AVAILABLE = False
        try:
            from pdfminer.high_level import extract_text  # noqa
            PDFMINER_AVAILABLE = True
        except Exception:
            PDFMINER_AVAILABLE = False
        try:
            from pdf2image import convert_from_path  # noqa
            PDF2IMAGE_AVAILABLE = True
        except Exception:
            PDF2IMAGE_AVAILABLE = False
        try:
            from docx2pdf import convert as docx_to_pdf_convert  # noqa
            DOCX2PDF_AVAILABLE = True
        except Exception:
            DOCX2PDF_AVAILABLE = False
        try:
            from pptx import Presentation  # noqa
            PYTHON_PPTX_AVAILABLE = True
        except Exception:
            PYTHON_PPTX_AVAILABLE = False
        try:
            from PIL import Image  # noqa
            PIL_AVAILABLE = True
        except Exception:
            PIL_AVAILABLE = False

        if not OPENPYXL_AVAILABLE and not XLRD_AVAILABLE:
            self.supported_types = [t for t in self.supported_types if t not in ['xlsx','xls']]
            logger.warning("openpyxl and xlrd not available. Excel processing disabled.")
        if not LANGCHAIN_OPENAI_AVAILABLE:
            self.supported_types = [t for t in self.supported_types if t not in self.image_types]
            logger.warning("langchain_openai not available. Image processing disabled.")
        if not PDFMINER_AVAILABLE:
            logger.warning("pdfminer not available. Using PyPDF2 fallback.")
        if not PDF2IMAGE_AVAILABLE:
            logger.warning("pdf2image not available. OCR disabled.")
        if not DOCX2PDF_AVAILABLE and not PIL_AVAILABLE:
            logger.warning("docx2pdf and PIL not available. DOCX/PPT OCR disabled.")

        self.encodings = ['utf-8','utf-8-sig','cp949','euc-kr','latin-1','ascii']

    # === 기존 private 메서드 대체 ===
    def _get_current_image_text_config(self) -> Dict[str, Any]:
        if self.config_composer:
            provider = str(self.config_composer.get_config_by_name("VISION_LANGUAGE_MODEL_PROVIDER").value).lower()

            if provider == "openai":
                config = {
                    'provider': str(self.config_composer.get_config_by_name("VISION_LANGUAGE_MODEL_PROVIDER").value).lower(),
                    'base_url': str(self.config_composer.get_config_by_name("VISION_LANGUAGE_OPENAI_BASE_URL").value),
                    'api_key': str(self.config_composer.get_config_by_name("VISION_LANGUAGE_OPENAI_API_KEY").value),
                    'model': str(self.config_composer.get_config_by_name("VISION_LANGUAGE_OPENAI_MODEL_NAME").value),
                    'temperature': float(self.config_composer.get_config_by_name("VISION_LANGUAGE_OPENAI_TEMPERATURE").value),
                    'image_quality': str(self.config_composer.get_config_by_name("VISION_LANGUAGE_OPENAI_IMAGE_QUALITY").value).lower(),
                    'batch_size': int(self.config_composer.get_config_by_name("VISION_LANGUAGE_OPENAI_BATCH_SIZE").value),
                }

            elif provider == "anthropic":
                config = {
                    'provider': str(self.config_composer.get_config_by_name("VISION_LANGUAGE_MODEL_PROVIDER").value).lower(),
                    'base_url': str(self.config_composer.get_config_by_name("VISION_LANGUAGE_ANTHROPIC_BASE_URL").value),
                    'api_key': str(self.config_composer.get_config_by_name("VISION_LANGUAGE_ANTHROPIC_API_KEY").value),
                    'model': str(self.config_composer.get_config_by_name("VISION_LANGUAGE_ANTHROPIC_MODEL_NAME").value),
                    'temperature': float(self.config_composer.get_config_by_name("VISION_LANGUAGE_ANTHROPIC_TEMPERATURE").value),
                    'image_quality': str(self.config_composer.get_config_by_name("VISION_LANGUAGE_ANTHROPIC_IMAGE_QUALITY").value).lower(),
                    'batch_size': int(self.config_composer.get_config_by_name("VISION_LANGUAGE_ANTHROPIC_BATCH_SIZE").value),
                }

            elif provider == "gemini":
                config = {
                    'provider': str(self.config_composer.get_config_by_name("VISION_LANGUAGE_MODEL_PROVIDER").value).lower(),
                    'base_url': str(self.config_composer.get_config_by_name("VISION_LANGUAGE_GEMINI_BASE_URL").value),
                    'api_key': str(self.config_composer.get_config_by_name("VISION_LANGUAGE_GEMINI_API_KEY").value),
                    'model': str(self.config_composer.get_config_by_name("VISION_LANGUAGE_GEMINI_MODEL_NAME").value),
                    'temperature': float(self.config_composer.get_config_by_name("VISION_LANGUAGE_GEMINI_TEMPERATURE").value),
                    'image_quality': str(self.config_composer.get_config_by_name("VISION_LANGUAGE_GEMINI_IMAGE_QUALITY").value).lower(),
                    'batch_size': int(self.config_composer.get_config_by_name("VISION_LANGUAGE_GEMINI_BATCH_SIZE").value),
                }

            elif provider == "vllm":
                config = {
                    'provider': str(self.config_composer.get_config_by_name("VISION_LANGUAGE_MODEL_PROVIDER").value).lower(),
                    'base_url': str(self.config_composer.get_config_by_name("VISION_LANGUAGE_VLLM_BASE_URL").value),
                    'api_key': str(self.config_composer.get_config_by_name("VISION_LANGUAGE_VLLM_API_KEY").value),
                    'model': str(self.config_composer.get_config_by_name("VISION_LANGUAGE_VLLM_MODEL_NAME").value),
                    'temperature': float(self.config_composer.get_config_by_name("VISION_LANGUAGE_OPENAI_TEMPERATURE").value),
                    'image_quality': 'auto',
                    'batch_size': 1,
                }

            elif provider == "aws_bedrock":
                # AWS Bedrock endpoint_url 처리
                endpoint_url_raw = str(self.config_composer.get_config_by_name("VISION_LANGUAGE_AWS_BEDROCK_ENDPOINT_URL").value)
                endpoint_url = None if endpoint_url_raw.lower() in ("", "auto", "none", "unset") else endpoint_url_raw

                config = {
                    'provider': str(self.config_composer.get_config_by_name("VISION_LANGUAGE_MODEL_PROVIDER").value).lower(),
                    'aws_access_key_id': str(self.config_composer.get_config_by_name("VISION_LANGUAGE_AWS_ACCESS_KEY_ID").value),
                    'aws_secret_access_key': str(self.config_composer.get_config_by_name("VISION_LANGUAGE_AWS_SECRET_ACCESS_KEY").value),
                    'aws_session_token': str(self.config_composer.get_config_by_name("VISION_LANGUAGE_AWS_SESSION_TOKEN").value),
                    'aws_region': str(self.config_composer.get_config_by_name("VISION_LANGUAGE_AWS_REGION").value),
                    'endpoint_url': endpoint_url,
                    'model': str(self.config_composer.get_config_by_name("VISION_LANGUAGE_AWS_MODEL_NAME").value),
                    'temperature': 0.0,
                    'image_quality': 'auto',
                    'batch_size': 1,
                }

            elif provider == "no_model":
                config = {
                    'provider': str(self.config_composer.get_config_by_name("VISION_LANGUAGE_MODEL_PROVIDER").value).lower(),
                    'base_url': "",
                    'api_key': "",
                    'model': "",
                    'temperature': 0,
                    'image_quality': 'auto',
                    'batch_size': 1,
                }

            else:
                raise ValueError(f"Unsupported VISION_LANGUAGE_MODEL_PROVIDER: {provider}")
        else:
            raise ValueError("Config composer not provided")

        return config

    # def _is_image_text_enabled(self, config: Dict[str, Any]) -> bool:
    #     try:
    #         from langchain_openai import ChatOpenAI  # noqa
    #         langchain_ok = True
    #     except Exception:
    #         langchain_ok = False
    #     return is_image_text_enabled(config, langchain_ok)

    # === 공개 API들: 원본과 동일 시그니처/동작 ===

    def get_supported_types(self) -> List[str]:
        return self.supported_types.copy()

    def get_file_category(self, file_type: str) -> str:
        ft = file_type.lower()
        if ft in self.document_types: return 'document'
        if ft in self.text_types:     return 'text'
        if ft in self.code_types:     return 'code'
        if ft in self.config_types:   return 'config'
        if ft in self.data_types:     return 'data'
        if ft in self.script_types:   return 'script'
        if ft in self.log_types:      return 'log'
        if ft in self.web_types:      return 'web'
        if ft in self.image_types:    return 'image'
        return 'unknown'

    def clean_text(self, text: str) -> str:
        return clean_text(text)

    def _is_text_quality_sufficient(self, text: Optional[str], min_chars: int = 500, min_word_ratio: float = 0.6) -> bool:
        from libs.core.functions.utils import is_text_quality_sufficient
        return is_text_quality_sufficient(text, min_chars, min_word_ratio)

    def clean_code_text(self, text: str, file_type: str) -> str:
        return clean_code_text(text)

    async def extract_text_from_file(
        self,
        file_path: str,
        file_extension: str,
        process_type: str,
        app_db=None,
        extract_default_metadata: bool = True,
        minio_bucket: Optional[str] = None,
        minio_object_name: Optional[str] = None,
    ) -> str:
        """
        파일에서 텍스트를 추출합니다.

        Args:
            file_path: 파일 경로
            file_extension: 파일 확장자
            process_type: 처리 유형
            app_db: 데이터베이스 연결
            extract_default_metadata: 메타데이터 추출 여부 (기본값: True)
                   - True: 문서의 메타데이터(제목, 작성자, 생성일 등)를 추출하여 결과에 포함
                   - False: 메타데이터 추출 생략
            minio_bucket: MinIO 버킷 이름 (선택사항, 로깅/추적용)
            minio_object_name: MinIO 객체 이름 (선택사항, 로깅/추적용)

        Returns:
            추출된 텍스트
        """
        category = self.get_file_category(file_extension)
        minio_info = f" (MinIO: {minio_bucket}/{minio_object_name})" if minio_bucket and minio_object_name else ""
        logger.info(f"Extracting text from {file_extension} file ({category} category): {file_path}{minio_info}")

        # config가 필요한 파일 타입에만 config를 가져옴
        cfg = None
        try:
            cfg = self._get_current_image_text_config()
        except ValueError:
            # Excel/CSV/DOC 등 config 없이도 동작하는 핸들러
            if file_extension not in ['xlsx', 'xls', 'csv', 'tsv', 'doc', 'docx', 'pptx', 'ppt', 'hwp', 'hwpx']:
                raise

        # 추출된 결과를 저장하는 변수
        result: str = ""

        if file_extension == 'pdf':
            # process_type에 따른 PDF 처리 방식 선택
            if process_type == "default":
                # PyMuPDF 전용 고속 처리 (pdfplumber 미사용, 강건한 테이블 추출)
                # V4와 동일 처리로 통합(추후 제거)
                result = await extract_text_from_pdf_v4(file_path, cfg, app_db, extract_default_metadata)
            elif process_type == "enhanced":
                # 이미지/테이블 추출 포함 (pdfplumber 우선)
                result = await extract_text_from_pdf(file_path, cfg, app_db, extract_default_metadata=extract_default_metadata)
            elif process_type == "enhanced_v2":
                # 고도화된 테이블 추출 V2 (PyMuPDF find_tables 기반, 복잡한 병합 셀 처리)
                # V4와 동일 처리로 통합(추후 제거)
                result = await extract_text_from_pdf_v4(file_path, cfg, app_db, extract_default_metadata=extract_default_metadata)
            elif process_type == "enhanced_v3":
                # 고도화된 테이블 추출 V3 (다중 전략 테이블 감지, 벡터 텍스트 OCR, 복잡한 병합 셀)
                # V4와 동일 처리로 통합(추후 제거)
                result = await extract_text_from_pdf_v4(file_path, cfg, app_db, extract_default_metadata=extract_default_metadata)
            elif process_type == "enhanced_v4":
                # 적응형 복잡도 기반 처리 V4 (복잡한 영역 이미지화 + MinIO 업로드)
                result = await extract_text_from_pdf_v4(file_path, cfg, app_db, extract_default_metadata=extract_default_metadata)
            elif process_type == "enhanced_unstructured":
                # pdfminer.six + unstructured cleaning 기반 처리
                # V4와 동일 처리로 통합 (추후 제거)
                result = await extract_text_from_pdf_v4(file_path, cfg, app_db, extract_default_metadata=extract_default_metadata)
            elif process_type == "enhanced_ocr":
                # 고도화된 OCR 기반 처리: 텍스트는 텍스트로, 비정형은 이미지로
                result = await extract_text_from_pdf_ocr(file_path, cfg, app_db, extract_default_metadata=extract_default_metadata)
            else:
                # 기본값
                result = await extract_text_from_pdf(file_path, cfg, app_db, extract_default_metadata=extract_default_metadata)
        elif file_extension == 'docx':
            # DOCX enhanced 모드 지원 (기본값)
            result = await extract_text_from_docx(file_path, cfg, app_db, extract_default_metadata=extract_default_metadata)
        elif file_extension == 'doc':
            # 구형 DOC 파일 처리 (RTF, OLE, HTML 자동 감지)
            result = await extract_text_from_doc(file_path, cfg, app_db, extract_default_metadata=extract_default_metadata)
        elif file_extension in ['pptx', 'ppt']:
            # PPT도 enhanced 모드 지원 (기본값)
            result = await extract_text_from_ppt(file_path, cfg, app_db, extract_default_metadata=extract_default_metadata)
        elif file_extension in ['xlsx', 'xls']:
            # Excel 파일 enhanced 모드 지원 (차트, 이미지, 병합 셀)
            result = await extract_text_from_excel(file_path, cfg, app_db, extract_default_metadata=extract_default_metadata)
        elif file_extension in ['csv','tsv']:
            # CSV/TSV enhanced 모드 지원 (HTML 테이블, 자동 인코딩/구분자 감지)
            result = await extract_text_from_csv(file_path, cfg, extract_default_metadata=extract_default_metadata)
        elif file_extension == 'hwp':
            result = await extract_text_from_hwp(file_path, cfg, app_db, extract_default_metadata=extract_default_metadata)
        elif file_extension == 'hwpx':
            result = await extract_text_from_hwpx(file_path, cfg, app_db, extract_default_metadata=extract_default_metadata)
        elif file_extension in self.image_types:
            # 이미지 파일은 단건 OCR(원본과 동일 동작: 내부에서 config 검사)
            from libs.core.ocr_legacy import convert_image_to_text
            result = await convert_image_to_text(file_path, cfg)
        elif file_extension in (self.text_types + self.code_types + self.config_types +
                                self.script_types + self.log_types + self.web_types):
            is_code = file_extension in self.code_types
            result = await extract_text_from_text_file(file_path, file_extension, self.encodings, is_code=is_code)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")

        # 모든 결과에 대해 UTF-8 JSON 인코딩 안전성 보장
        # 서로게이트 문자, PUA 문자, 비문자 등 문제를 일으킬 수 있는 문자 제거
        return sanitize_text_for_json(result)

    # === 청킹 관련 ===
    def chunk_text(
        self,
        text: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        file_extension: Optional[str] = None,
        force_chunking: Optional[bool] = False,
        # Advanced Settings (향후 구현 예정, 현재는 호환성을 위해 인자만 수용)
        chunking_strategy: str = "recursive",
        stride: Optional[int] = None,
        parent_chunk_size: Optional[int] = None,
        child_chunk_size: Optional[int] = None,
        **kwargs  # 추가 인자 무시
    ) -> List[str]:
        """
        텍스트를 청크로 분할합니다.

        Args:
            text: 원본 텍스트
            chunk_size: 청크 최대 크기
            chunk_overlap: 청크 간 겹침 크기
            file_extension: 파일 확장자 (csv, xlsx 등) - 테이블 기반 처리 결정에 사용
            force_chunking: 강제 청킹 여부 (테이블 기반 파일 제외)
            chunking_strategy: 청킹 전략 (recursive, sliding, hierarchical) - 향후 구현 예정
            stride: Sliding Window 전략에서의 스트라이드 - 향후 구현 예정
            parent_chunk_size: Hierarchical 전략에서의 부모 청크 크기 - 향후 구현 예정
            child_chunk_size: Hierarchical 전략에서의 자식 청크 크기 - 향후 구현 예정

        Returns:
            청크 리스트
        """
        # TODO: chunking_strategy에 따른 다양한 청킹 전략 구현
        # 현재는 기본 recursive 전략만 사용
        if chunking_strategy != "recursive":
            logger.warning(f"Chunking strategy '{chunking_strategy}' is not yet implemented, falling back to 'recursive'")

        return split_text_preserving_html_blocks(text, chunk_size, chunk_overlap, file_extension, force_chunking)

    def _reconstruct_text_from_chunks(self, chunks: List[str], chunk_overlap: int) -> str:
        return reconstruct_text_from_chunks(chunks, chunk_overlap)

    def _find_overlap_length(self, chunk1: str, chunk2: str, max_overlap: int) -> int:
        return find_overlap_length(chunk1, chunk2, max_overlap)

    def chunk_code_text(self, text: str, file_type: str, chunk_size: int = 1500, chunk_overlap: int = 300) -> List[str]:
        return chunk_code_text(text, file_type, chunk_size, chunk_overlap)

    def estimate_chunks_count(self, text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> int:
        return estimate_chunks_count(text, chunk_size, chunk_overlap)

    # === 라인/페이지 매핑(원본 로직과 동일 효과) ===
    def _extract_page_mapping(self, text: str, file_extension: str) -> List[Dict[str, Any]]:
        try:
            page_mapping: List[Dict[str, Any]] = []
            if file_extension in ['pdf','ppt','pptx','doc','docx']:
                patterns = [
                    r'=== 페이지 (\d+) ===',
                    r'=== 페이지 (\d+) \(OCR\) ===',
                    r'=== 페이지 (\d+) \(OCR\+참고\) ===',
                    r'=== 슬라이드 (\d+) ===',
                    r'=== 슬라이드 (\d+) \(OCR\) ===',
                    r'<페이지\s*번호>\s*(\d+)\s*</페이지\s*번호>',
                    r'<페이지\s*번호>\s*(\d+)\s*\(OCR\)\s*</페이지\s*번호>',
                    r'<페이지\s*번호>\s*(\d+)\s*\(OCR\+참고\)\s*</페이지\s*번호>',
                    r'<슬라이드\s*번호>\s*(\d+)\s*</슬라이드\s*번호>',
                    r'<슬라이드\s*번호>\s*(\d+)\s*\(OCR\)\s*</슬라이드\s*번호>',
                ]
                for pat in patterns:
                    matches = list(re.finditer(pat, text))
                    if matches:
                        for i, m in enumerate(matches):
                            page_num = int(m.group(1))
                            start = m.end()
                            end = matches[i+1].start() if i+1 < len(matches) else len(text)
                            page_mapping.append({"page_num": page_num, "start_pos": start, "end_pos": end})
                        page_mapping.sort(key=lambda x: x["page_num"])
                        break
                if not page_mapping and file_extension in ['doc','docx']:
                    chars_per_page = 1500
                    L = len(text)
                    if L > chars_per_page:
                        est = (L + chars_per_page - 1)//chars_per_page
                        for pn in range(1, est+1):
                            s = (pn-1)*chars_per_page
                            e = min(pn*chars_per_page, L)
                            page_mapping.append({"page_num": pn, "start_pos": s, "end_pos": e})
                if not page_mapping:
                    page_mapping = [{"page_num":1,"start_pos":0,"end_pos":len(text)}]
            elif file_extension in ['xlsx','xls']:
                matches = list(re.finditer(r'=== 시트: ([^=]+) ===', text))
                if matches:
                    for i, m in enumerate(matches):
                        s = m.end()
                        e = matches[i+1].start() if i+1 < len(matches) else len(text)
                        page_mapping.append({"page_num": i+1, "start_pos": s, "end_pos": e,
                                             "sheet_name": m.group(1).strip()})
                else:
                    page_mapping = [{"page_num":1,"start_pos":0,"end_pos":len(text)}]
            else:
                lines = text.split('\n')
                lpp = 1000
                if len(lines) > lpp:
                    pc = (len(lines) + lpp - 1)//lpp
                    cur = 0
                    for pn in range(1, pc+1):
                        sline = (pn-1)*lpp
                        eline = min(pn*lpp, len(lines))
                        page_text = '\n'.join(lines[sline:eline])
                        s = cur; e = cur + len(page_text)
                        page_mapping.append({"page_num": pn, "start_pos": s, "end_pos": e})
                        cur = e + 1
                else:
                    page_mapping = [{"page_num":1,"start_pos":0,"end_pos":len(text)}]
            return page_mapping
        except Exception:
            return [{"page_num":1,"start_pos":0,"end_pos":len(text)}]

    def _find_line_index_by_pos(self, pos: int, line_table: List[Dict[str, int]]) -> int:
        try:
            if not line_table:
                return 0
            starts = [l["start"] for l in line_table]
            idx = bisect.bisect_right(starts, pos) - 1
            return 0 if idx < 0 else min(idx, len(line_table)-1)
        except Exception:
            return 0

    def _build_line_offset_table(self, text: str, file_extension: str) -> List[Dict[str, int]]:
        try:
            lines = text.split('\n')
            table: List[Dict[str, int]] = []
            pos = 0
            page_mapping = self._extract_page_mapping(text, file_extension)
            def _page_for_pos(p: int) -> int:
                for info in page_mapping:
                    if info["start_pos"] <= p < info["end_pos"]:
                        return info["page_num"]
                return 1
            for i, line in enumerate(lines):
                start = pos
                end = pos + len(line)
                mid = start + max(0, (end-start)//2)
                page = _page_for_pos(mid)
                table.append({"line_num": i+1, "start": start, "end": end, "page": page})
                pos = end + 1
            return table
        except Exception:
            return [{"line_num":1,"start":0,"end":len(text),"page":1}]

    def chunk_text_with_metadata(
        self,
        text: str,
        file_extension: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        force_chunking: Optional[bool] = False,
        # Advanced Settings (향후 구현 예정, 현재는 호환성을 위해 인자만 수용)
        chunking_strategy: str = "recursive",
        stride: Optional[int] = None,
        parent_chunk_size: Optional[int] = None,
        child_chunk_size: Optional[int] = None,
        **kwargs  # 추가 인자 무시
    ) -> List[Dict[str, Any]]:
        """
        메타데이터와 함께 텍스트를 청크로 분할합니다.

        Args:
            text: 원본 텍스트
            file_extension: 파일 확장자
            chunk_size: 청크 최대 크기
            chunk_overlap: 청크 간 겹침 크기
            force_chunking: 강제 청킹 여부 (테이블 기반 파일 제외)
            chunking_strategy: 청킹 전략 (recursive, sliding, hierarchical) - 향후 구현 예정
            stride: Sliding Window 전략에서의 스트라이드 - 향후 구현 예정
            parent_chunk_size: Hierarchical 전략에서의 부모 청크 크기 - 향후 구현 예정
            child_chunk_size: Hierarchical 전략에서의 자식 청크 크기 - 향후 구현 예정

        Returns:
            메타데이터가 포함된 청크 리스트
        """
        # file_extension을 chunk_text에 전달하여 CSV/Excel 등 테이블 기반 파일 처리
        chunks = self.chunk_text(
            text, chunk_size, chunk_overlap,
            file_extension=file_extension,
            force_chunking=force_chunking,
            chunking_strategy=chunking_strategy,
            stride=stride,
            parent_chunk_size=parent_chunk_size,
            child_chunk_size=child_chunk_size
        )
        reconstructed = self._reconstruct_text_from_chunks(chunks, chunk_overlap)
        line_table = self._build_line_offset_table(reconstructed, file_extension)

        out: List[Dict[str, Any]] = []
        cur = 0
        for idx, ch in enumerate(chunks):
            start = cur
            end = cur + len(ch) - 1
            sidx = self._find_line_index_by_pos(start, line_table)
            eidx = self._find_line_index_by_pos(end, line_table)
            line_start = line_table[sidx]["line_num"]
            line_end = line_table[eidx]["line_num"]
            page_number = line_table[sidx].get("page", 1)
            out.append({
                "text": ch,
                "page_number": page_number,
                "line_start": line_start,
                "line_end": line_end,
                "global_start": start,
                "global_end": end,
                "chunk_index": idx
            })
            cur += len(ch)
            if idx < len(chunks) - 1:
                ov = find_overlap_length(ch, chunks[idx+1], chunk_overlap)
                cur -= ov
        logger.info(f"Created {len(out)} chunks with metadata using reconstructed text")
        return out

    def validate_file_format(self, file_path: str) -> tuple[bool, str]:
        try:
            ext = Path(file_path).suffix[1:].lower()
            return (ext in self.supported_types, ext)
        except Exception:
            return (False, "")

    def get_file_info(self, file_path: str) -> Dict[str, str]:
        try:
            ext = Path(file_path).suffix[1:].lower()
            cat = self.get_file_category(ext)
            ok = ext in self.supported_types
            return {'extension': ext, 'category': cat, 'supported': str(ok)}
        except Exception:
            return {'extension':'unknown','category':'unknown','supported':'false'}

    async def extract_text_from_repository(
        self,
        gitlab_url: str,
        gitlab_token: str,
        repository_path: str,
        branch: str = "main",
        enable_annotation: bool = False,
        enable_api_extraction: bool = False,
        progress_callback = None
    ) -> Dict[str, Any]:
        """
        GitLab 레포지토리에서 코드를 추출하여 반환

        Args:
            gitlab_url: GitLab 인스턴스 URL
            gitlab_token: Personal Access Token
            repository_path: 레포지토리 경로 (예: group/project)
            branch: 브랜치 이름
            enable_annotation: LLM 기반 코드 주석 생성 여부
            enable_api_extraction: API 엔드포인트 추출 여부
            progress_callback: 진행 상황 콜백 함수 (Optional[Callable])

        Returns:
            {
                'files': [{
                    'path': str,
                    'content': str,
                    'annotated_content': str (optional),
                    'language': str,
                    'size': int
                }],
                'api_info': [...] (optional),
                'metadata': {...}
            }
        """
        cfg = self._get_current_image_text_config()

        return await extract_text_from_code_repository(
            gitlab_url=gitlab_url,
            gitlab_token=gitlab_token,
            repository_path=repository_path,
            branch=branch,
            config=cfg,
            enable_annotation=enable_annotation,
            enable_api_extraction=enable_api_extraction,
            progress_callback=progress_callback
        )

    def test(self):
        try:
            cfg = self._get_current_image_text_config()
            logger.info(f"🔍 Test - Current provider: {cfg.get('provider','no_model')}")
            logger.info(f"🔍 Test - Current config: {cfg}")
            try:
                from langchain_openai import ChatOpenAI  # noqa
                langchain_ok = True
            except Exception:
                langchain_ok = False
        except Exception as e:
            logger.error(f"Error in test method: {e}")
            raise
