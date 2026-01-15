# MINIO에서 파일을 다운로드 한 뒤 그것을 OCR 처리하는 모듈.
import re
import base64
import logging
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

from service.storage.minio_client import (
    get_minio_client,
    download_file,
    parse_minio_path,
    CACHE_DIR,
)

logger = logging.getLogger("ocr-processor")

# 이미지 태그 패턴: [Image:{minio_path}] 또는 [image:{minio_path}] (대소문자 무관)
IMAGE_TAG_PATTERN = re.compile(r'\[[Ii]mage:([^\]]+)\]')


def _b64_from_file(path: str) -> str:
    """파일을 Base64로 인코딩"""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _get_mime_type(file_path: str) -> str:
    """파일 확장자에 따른 MIME 타입 반환"""
    ext = os.path.splitext(file_path)[1].lower()
    mime_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".bmp": "image/bmp",
        ".webp": "image/webp",
        ".tiff": "image/tiff",
        ".svg": "image/svg+xml",
    }
    return mime_map.get(ext, "image/jpeg")


def extract_image_tags(text: str) -> List[str]:
    """
    텍스트에서 [Image:{minio_path}] 태그를 추출
    
    Args:
        text: 이미지 태그가 포함된 텍스트
        
    Returns:
        추출된 minio_path 목록
    """
    matches = IMAGE_TAG_PATTERN.findall(text)
    return matches


def download_image_from_minio(minio_path: str) -> Optional[str]:
    """
    MinIO에서 이미지 파일을 다운로드
    
    Args:
        minio_path: 'bucket_name/object_name' 형식의 경로
        
    Returns:
        다운로드된 파일의 로컬 경로 또는 None
    """
    try:
        bucket_name, object_name = parse_minio_path(minio_path)
        filename = os.path.basename(object_name)
        local_path = os.path.join(CACHE_DIR, filename)
        
        # 캐시 확인
        if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
            logger.info(f"[OCR] 캐시 히트: {local_path}")
            return local_path
        
        # MinIO에서 다운로드
        client = get_minio_client()
        logger.info(f"[OCR] MinIO 다운로드: bucket={bucket_name}, object={object_name}")
        download_file(client, object_name, local_path, bucket_name=bucket_name)
        logger.info(f"[OCR] 다운로드 완료: {local_path}")
        
        return local_path
        
    except Exception as e:
        logger.error(f"[OCR] 이미지 다운로드 실패: {minio_path}, error: {e}")
        return None


async def convert_image_to_text_with_llm(
    image_path: str,
    llm_client: Any,
    provider: str
) -> str:
    """
    VL 모델을 사용하여 이미지를 텍스트로 변환
    
    Args:
        image_path: 로컬 이미지 파일 경로
        llm_client: LangChain LLM 클라이언트
        provider: LLM 프로바이더 (openai, anthropic, gemini, vllm)
        
    Returns:
        이미지에서 추출된 텍스트
    """
    try:
        from langchain_core.messages import HumanMessage
        
        b64_image = _b64_from_file(image_path)
        mime_type = _get_mime_type(image_path)
        
        # vllm은 간단한 프롬프트 사용
        if provider == "vllm":
            prompt = "Describe the contents of this image."
        else:
            prompt = (
                "Extract meaningful information from this image.\n\n"
                "**If the image contains a TABLE:**\n"
                "- Convert to HTML table format (<table>, <tr>, <td>, <th>)\n"
                "- Use 'rowspan' and 'colspan' attributes for merged cells\n"
                "- Preserve all cell content exactly as shown\n"
                "- Example:\n"
                "  <table>\n"
                "    <tr><th colspan=\"2\">Header</th></tr>\n"
                "    <tr><td rowspan=\"2\">Merged</td><td>A</td></tr>\n"
                "    <tr><td>B</td></tr>\n"
                "  </table>\n\n"
                "**If the image contains TEXT (non-table):**\n"
                "- Extract all text exactly as shown\n"
                "- Keep layout, hierarchy, and structure\n\n"
                "**If the image contains DATA (charts, graphs, diagrams):**\n"
                "- Extract the data and its meaning\n"
                "- Describe trends, relationships, or key insights\n\n"
                "**If the image is decorative or has no semantic meaning:**\n"
                "- Simply state what it is in one short sentence\n"
                "- Example: 'A decorative geometric shape' or 'Company logo'\n"
                "- Do NOT over-analyze decorative elements\n\n"
                "**Rules:**\n"
                "- Output in Korean (except HTML tags)\n"
                "- Tables MUST use HTML format with proper rowspan/colspan\n"
                "- Be concise - only include what is semantically meaningful\n"
                "- No filler words or unnecessary descriptions"
            )
        
        # Provider별 메시지 구성
        if provider in ("openai", "vllm"):
            content = [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{b64_image}"}
                }
            ]
            message = HumanMessage(content=content)
            
        elif provider == "anthropic":
            content = [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": mime_type,
                        "data": b64_image
                    }
                },
                {"type": "text", "text": prompt}
            ]
            message = HumanMessage(content=content)
            
        elif provider == "gemini":
            content = [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{b64_image}"}
                }
            ]
            message = HumanMessage(content=content)
            
        elif provider == "aws_bedrock":
            # AWS Bedrock (Claude via Bedrock) - Anthropic과 동일한 형식 사용
            content = [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": mime_type,
                        "data": b64_image
                    }
                },
                {"type": "text", "text": prompt}
            ]
            message = HumanMessage(content=content)
            
        else:
            return None  # 지원하지 않는 프로바이더
        
        response = await llm_client.ainvoke([message])
        result = response.content.strip()
        
        # 결과를 [그림:...] 형식으로 감싸기
        result = f"[그림:{result}]"
            
        logger.info(f"[OCR] 이미지 텍스트 변환 완료: {os.path.basename(image_path)}")
        return result
        
    except Exception as e:
        logger.error(f"[OCR] 이미지 텍스트 변환 실패: {image_path}, error: {e}")
        return f"[이미지 변환 오류: {str(e)}]"


async def process_text_with_ocr(
    text: str,
    llm_client: Any,
    provider: str
) -> str:
    """
    텍스트 내 이미지 태그를 감지하고 OCR 처리하여 텍스트로 대체
    
    Args:
        text: [Image:{minio_path}] 태그가 포함된 텍스트
        llm_client: LangChain LLM 클라이언트
        provider: LLM 프로바이더
        
    Returns:
        이미지 태그가 OCR 결과로 대체된 텍스트
    """
    if not llm_client:
        logger.warning("[OCR] LLM 클라이언트가 없어 OCR 처리를 건너뜁니다")
        return text
    
    # 이미지 태그 추출
    image_paths = extract_image_tags(text)
    
    if not image_paths:
        logger.debug("[OCR] 텍스트에 이미지 태그가 없습니다")
        return text
    
    logger.info(f"[OCR] {len(image_paths)}개의 이미지 태그 감지됨")
    
    result_text = text
    
    for minio_path in image_paths:
        # 대소문자 무관하게 태그 매칭
        tag_pattern = re.compile(r'\[[Ii]mage:' + re.escape(minio_path) + r'\]')
        
        # MinIO에서 이미지 다운로드
        local_path = download_image_from_minio(minio_path)
        
        if local_path is None:
            # 다운로드 실패 시 원본 태그 유지
            logger.warning(f"[OCR] 이미지 다운로드 실패, 원본 태그 유지: {minio_path}")
            continue
        
        # VL 모델로 이미지를 텍스트로 변환
        ocr_result = await convert_image_to_text_with_llm(
            image_path=local_path,
            llm_client=llm_client,
            provider=provider
        )
        
        # OCR 실패 시 원본 태그 유지 (None 또는 에러 메시지인 경우)
        if ocr_result is None or ocr_result.startswith("[이미지 변환 오류:"):
            logger.warning(f"[OCR] 이미지 변환 실패, 원본 태그 유지: {minio_path}")
            continue
        
        # 태그를 OCR 결과로 대체
        result_text = tag_pattern.sub(ocr_result, result_text)
        logger.info(f"[OCR] 태그 대체 완료: {minio_path[:50]}...")
    
    return result_text


async def process_text_with_ocr_progress(
    text: str,
    llm_client: Any,
    provider: str,
    progress_callback: Optional[Callable[[Dict[str, Any]], Any]] = None
) -> str:
    """
    텍스트 내 이미지 태그를 감지하고 OCR 처리하여 텍스트로 대체 (진행 상황 콜백 지원)
    
    Args:
        text: [Image:{minio_path}] 태그가 포함된 텍스트
        llm_client: LangChain LLM 클라이언트
        provider: LLM 프로바이더
        progress_callback: 진행 상황 콜백 함수
        
    Returns:
        이미지 태그가 OCR 결과로 대체된 텍스트
    """
    if not llm_client:
        logger.warning("[OCR] LLM 클라이언트가 없어 OCR 처리를 건너뜁니다")
        return text
    
    # 이미지 태그 추출
    image_paths = extract_image_tags(text)
    
    if not image_paths:
        logger.debug("[OCR] 텍스트에 이미지 태그가 없습니다")
        return text
    
    total_chunks = len(image_paths)
    logger.info(f"[OCR] {total_chunks}개의 이미지 태그 감지됨")
    
    result_text = text
    success_count = 0
    failed_count = 0
    
    for idx, minio_path in enumerate(image_paths):
        # 진행 상황 콜백 - 처리 시작
        if progress_callback:
            await progress_callback({
                'event': 'ocr_tag_processing',
                'chunk_index': idx,
                'total_chunks': total_chunks,
                'minio_path': minio_path
            })
        
        # 대소문자 무관하게 태그 매칭
        tag_pattern = re.compile(r'\[[Ii]mage:' + re.escape(minio_path) + r'\]')
        
        # MinIO에서 이미지 다운로드
        local_path = download_image_from_minio(minio_path)
        
        if local_path is None:
            # 다운로드 실패 시 원본 태그 유지
            logger.warning(f"[OCR] 이미지 다운로드 실패, 원본 태그 유지: {minio_path}")
            failed_count += 1
            if progress_callback:
                await progress_callback({
                    'event': 'ocr_chunk_processed',
                    'chunk_index': idx,
                    'total_chunks': total_chunks,
                    'status': 'failed',
                    'error': f'Download failed: {minio_path}'
                })
            continue
        
        try:
            # VL 모델로 이미지를 텍스트로 변환
            ocr_result = await convert_image_to_text_with_llm(
                image_path=local_path,
                llm_client=llm_client,
                provider=provider
            )
            
            # OCR 실패 시 원본 태그 유지 (None 또는 에러 메시지인 경우)
            if ocr_result is None or ocr_result.startswith("[이미지 변환 오류:"):
                logger.warning(f"[OCR] 이미지 변환 실패, 원본 태그 유지: {minio_path}")
                failed_count += 1
                if progress_callback:
                    await progress_callback({
                        'event': 'ocr_chunk_processed',
                        'chunk_index': idx,
                        'total_chunks': total_chunks,
                        'status': 'failed',
                        'error': ocr_result or 'OCR returned None'
                    })
                continue
            
            # 태그를 OCR 결과로 대체
            result_text = tag_pattern.sub(ocr_result, result_text)
            success_count += 1
            logger.info(f"[OCR] 태그 대체 완료: {minio_path[:50]}...")
            
            if progress_callback:
                await progress_callback({
                    'event': 'ocr_chunk_processed',
                    'chunk_index': idx,
                    'total_chunks': total_chunks,
                    'status': 'success'
                })
                
        except Exception as e:
            logger.error(f"[OCR] 이미지 처리 오류: {minio_path}, error: {e}")
            failed_count += 1
            if progress_callback:
                await progress_callback({
                    'event': 'ocr_chunk_processed',
                    'chunk_index': idx,
                    'total_chunks': total_chunks,
                    'status': 'failed',
                    'error': str(e)
                })
    
    return result_text


async def process_batch_texts_with_ocr(
    texts: List[str],
    llm_client: Any,
    provider: str
) -> List[str]:
    """
    여러 텍스트에 대해 OCR 처리를 수행
    
    Args:
        texts: 텍스트 목록
        llm_client: LangChain LLM 클라이언트
        provider: LLM 프로바이더
        
    Returns:
        OCR 처리된 텍스트 목록
    """
    results = []
    for text in texts:
        processed = await process_text_with_ocr(text, llm_client, provider)
        results.append(processed)
    return results