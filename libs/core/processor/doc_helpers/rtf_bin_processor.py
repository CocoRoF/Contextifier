# service/document_processor/processor/doc_helpers/rtf_bin_processor.py
"""
RTF Binary Data Processor - RTF 파일의 바이너리 데이터 처리기

RTF 파일 내의 바이너리 이미지 데이터를 처리합니다:
- bin 태그: 직접 바이너리 데이터 (JPEG, PNG, WMF 등)
- pict 그룹: 16진수 인코딩 또는 바이너리 이미지

주요 기능:
1. \binN 태그 스킵 (N 바이트의 바이너리 데이터를 건너뜀)
2. \pict 그룹에서 이미지 추출
3. 이미지를 MinIO에 업로드하고 [image:path] 태그로 변환

RTF 스펙:
- \binN: N 바이트의 raw 바이너리 데이터가 뒤따름
- \pict: 이미지 그룹 시작
- \jpegblip: JPEG 형식
- \pngblip: PNG 형식
- \wmetafile: Windows Metafile
- \emfblip: Enhanced Metafile
"""
import logging
import re
import struct
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from libs.core.functions.utils import upload_image_to_minio

logger = logging.getLogger("document-processor")


# === 이미지 형식 상수 ===

# 매직 넘버로 이미지 형식 판별
IMAGE_SIGNATURES = {
    b'\xff\xd8\xff': 'jpeg',           # JPEG
    b'\x89PNG\r\n\x1a\n': 'png',       # PNG
    b'GIF87a': 'gif',                  # GIF87
    b'GIF89a': 'gif',                  # GIF89
    b'BM': 'bmp',                      # BMP
    b'\xd7\xcd\xc6\x9a': 'wmf',        # WMF (placeable)
    b'\x01\x00\x09\x00': 'wmf',        # WMF (standard)
    b'\x01\x00\x00\x00': 'emf',        # EMF
}

# RTF 이미지 타입 매핑
RTF_IMAGE_TYPES = {
    'jpegblip': 'jpeg',
    'pngblip': 'png',
    'wmetafile': 'wmf',
    'emfblip': 'emf',
    'dibitmap': 'bmp',
    'wbitmap': 'bmp',
}


@dataclass
class RTFBinaryRegion:
    """RTF 바이너리 데이터 영역 정보"""
    start_pos: int          # 원본에서의 시작 위치 (바이트)
    end_pos: int            # 원본에서의 끝 위치 (바이트)
    bin_type: str           # "bin" 또는 "pict"
    data_size: int          # 바이너리 데이터 크기
    image_format: str = ""  # 이미지 형식 (jpeg, png, wmf 등)
    image_data: bytes = b"" # 추출된 이미지 데이터


@dataclass
class RTFBinaryProcessResult:
    """RTF 바이너리 처리 결과"""
    clean_content: bytes                    # 바이너리가 제거/치환된 콘텐츠
    binary_regions: List[RTFBinaryRegion] = field(default_factory=list)
    image_tags: Dict[int, str] = field(default_factory=dict)  # 위치 -> 이미지 태그


class RTFBinaryProcessor:
    """
    RTF 바이너리 데이터 처리기

    RTF 파일에서 바이너리 이미지 데이터를 추출하고,
    MinIO에 업로드하여 이미지 태그로 변환합니다.
    """

    def __init__(self, app_db=None, processed_images: Optional[Set[str]] = None):
        """
        Args:
            app_db: 데이터베이스 연결 (이미지 메타데이터 저장용)
            processed_images: 이미 처리된 이미지 해시 집합 (중복 방지)
        """
        self.app_db = app_db
        self.processed_images = processed_images if processed_images is not None else set()
        self.binary_regions: List[RTFBinaryRegion] = []
        self.image_tags: Dict[int, str] = {}

    def process(self, content: bytes) -> RTFBinaryProcessResult:
        """
        RTF 바이너리 콘텐츠를 처리합니다.

        bin 태그의 바이너리 데이터를 스킵하고,
        pict 그룹의 이미지를 추출하여 MinIO에 업로드합니다.

        Args:
            content: RTF 파일 바이너리 콘텐츠

        Returns:
            처리 결과 (정제된 콘텐츠, 바이너리 영역 정보, 이미지 태그)
        """
        self.binary_regions = []
        self.image_tags = {}

        # 1단계: \bin 태그 위치 및 크기 파악
        bin_regions = self._find_bin_regions(content)

        # 2단계: \pict 그룹에서 이미지 추출 (bin 영역 외부)
        pict_regions = self._find_pict_regions(content, bin_regions)

        # 3단계: 바이너리 영역 통합 및 정렬
        all_regions = bin_regions + pict_regions
        all_regions.sort(key=lambda r: r.start_pos)
        self.binary_regions = all_regions

        # 4단계: 이미지 추출 및 MinIO 업로드
        self._process_images()

        # 5단계: 바이너리 데이터를 제거한 콘텐츠 생성
        clean_content = self._remove_binary_data(content)

        return RTFBinaryProcessResult(
            clean_content=clean_content,
            binary_regions=self.binary_regions,
            image_tags=self.image_tags
        )

    def _find_bin_regions(self, content: bytes) -> List[RTFBinaryRegion]:
        """
        \binN 태그를 찾아 바이너리 영역을 식별합니다.

        RTF 스펙에서 binN은 N 바이트의 raw 바이너리 데이터가 뒤따름을 의미합니다.
        이 데이터는 문자열 디코딩 시 깨지므로 건너뛰어야 합니다.

        중요: bin을 포함하는 상위 shppict 그룹 전체를 제거 영역으로 설정합니다.

        Args:
            content: RTF 바이너리 콘텐츠

        Returns:
            바이너리 영역 리스트
        """
        regions = []

        # \bin 패턴 찾기: \binN (N은 바이트 수)
        # RTF에서 \bin 다음의 숫자가 바이트 수를 나타냄
        pattern = rb'\\bin(\d+)'

        for match in re.finditer(pattern, content):
            try:
                bin_size = int(match.group(1))
                bin_tag_start = match.start()
                bin_tag_end = match.end()

                # \bin 태그 다음에 공백이 있을 수 있음
                data_start = bin_tag_end
                if data_start < len(content) and content[data_start:data_start+1] == b' ':
                    data_start += 1

                data_end = data_start + bin_size

                if data_end <= len(content):
                    # 바이너리 데이터 추출
                    binary_data = content[data_start:data_end]

                    # 이미지 형식 감지
                    image_format = self._detect_image_format(binary_data)

                    # 상위 \shppict 그룹 찾기
                    # \bin 위치에서 역방향으로 {\*\shppict 또는 {\\shppict 찾기
                    group_start = bin_tag_start
                    group_end = data_end

                    # 역방향으로 \shppict 검색 (최대 500바이트 뒤로)
                    search_start = max(0, bin_tag_start - 500)
                    search_area = content[search_start:bin_tag_start]

                    # \shppict 찾기
                    shppict_pos = search_area.rfind(b'\\shppict')
                    if shppict_pos != -1:
                        # 그룹 시작 { 찾기
                        abs_pos = search_start + shppict_pos
                        brace_pos = abs_pos
                        while brace_pos > 0 and content[brace_pos:brace_pos+1] != b'{':
                            brace_pos -= 1
                        group_start = brace_pos

                        # 그룹 끝 } 찾기 (바이너리 데이터 이후)
                        depth = 1
                        j = data_end
                        while j < len(content) and depth > 0:
                            if content[j:j+1] == b'{':
                                depth += 1
                            elif content[j:j+1] == b'}':
                                depth -= 1
                            j += 1
                        group_end = j

                    region = RTFBinaryRegion(
                        start_pos=group_start,
                        end_pos=group_end,
                        bin_type="bin",
                        data_size=bin_size,
                        image_format=image_format,
                        image_data=binary_data
                    )
                    regions.append(region)

                    logger.debug(
                        f"Found \\bin region: group_pos={group_start}-{group_end}, "
                        f"bin_pos={bin_tag_start}, size={bin_size}, "
                        f"format={image_format or 'unknown'}"
                    )

            except (ValueError, IndexError) as e:
                logger.debug(f"Error parsing \\bin tag: {e}")
                continue

        logger.info(f"Found {len(regions)} \\bin regions in RTF")
        return regions

    def _find_pict_regions(
        self,
        content: bytes,
        exclude_regions: List[RTFBinaryRegion]
    ) -> List[RTFBinaryRegion]:
        """
        pict 그룹에서 16진수 인코딩된 이미지를 찾습니다.

        주의: pict 그룹이 bin 태그를 포함하는 경우는 이미 _find_bin_regions에서
        처리되었으므로 여기서는 스킵합니다.

        RTF 이미지 인코딩 방식:
        1. \bin 태그: 직접 바이너리 데이터 (이미 처리됨)
        2. 16진수: \pict ... [hex data] } 형태

        Args:
            content: RTF 바이너리 콘텐츠
            exclude_regions: 제외할 영역 (이미 처리된 \bin 영역)

        Returns:
            pict 이미지 영역 리스트 (16진수 인코딩된 것만)
        """
        regions = []

        # \bin 태그 위치 집합 생성 (근처에 \bin이 있는 \pict는 스킵)
        bin_tag_positions = set()
        for region in exclude_regions:
            if region.bin_type == "bin":
                bin_tag_positions.add(region.start_pos)

        # 제외 영역을 빠르게 체크하기 위한 집합 생성
        excluded_ranges = [(r.start_pos, r.end_pos) for r in exclude_regions]

        def is_excluded(pos: int) -> bool:
            """주어진 위치가 제외 영역에 포함되는지 확인"""
            for start, end in excluded_ranges:
                if start <= pos < end:
                    return True
            return False

        def has_bin_nearby(pict_pos: int, search_range: int = 200) -> bool:
            """
            pict 근처에 bin 태그가 있는지 확인.
            pict 그룹이 bin 태그를 포함하면 True 반환.
            """
            # \pict 위치부터 search_range 내에 \bin 태그가 있는지 확인
            for bin_pos in bin_tag_positions:
                if pict_pos < bin_pos < pict_pos + search_range:
                    return True
            return False

        try:
            text_content = content.decode('cp1252', errors='replace')

            # \pict 그룹 찾기
            # 패턴: \pict\jpegblip... [hex data]}
            pict_start_pattern = r'\\pict\s*((?:\\[a-zA-Z]+\d*\s*)*)'

            for match in re.finditer(pict_start_pattern, text_content):
                start_pos = match.start()

                # 제외 영역인지 확인
                if is_excluded(start_pos):
                    continue

                # 근처에 \bin 태그가 있으면 스킵 (이미 처리됨)
                if has_bin_nearby(start_pos):
                    logger.debug(f"Skipping \\pict at {start_pos} - has \\bin tag nearby")
                    continue

                attrs = match.group(1)

                # 이미지 타입 확인
                image_format = ""
                for rtf_type, fmt in RTF_IMAGE_TYPES.items():
                    if rtf_type in attrs:
                        image_format = fmt
                        break

                # 16진수 데이터 추출
                # \pict 속성들 다음에 16진수 데이터가 옴
                hex_start = match.end()
                hex_data = []
                i = hex_start

                while i < len(text_content):
                    ch = text_content[i]
                    if ch in '0123456789abcdefABCDEF':
                        hex_data.append(ch)
                    elif ch in ' \t\r\n':
                        pass  # 공백 무시
                    elif ch == '}':
                        break  # 그룹 끝
                    elif ch == '\\':
                        # \bin 태그 확인
                        if text_content[i:i+4] == '\\bin':
                            # \bin 태그가 있으면 이 \pict는 스킵
                            logger.debug(f"Skipping \\pict at {start_pos} - contains \\bin tag")
                            hex_data = []  # 데이터 버리기
                            break
                        # 다른 제어 워드까지 스킵
                        while i < len(text_content) and text_content[i] not in ' \t\r\n}':
                            i += 1
                        continue
                    else:
                        break
                    i += 1

                hex_str = ''.join(hex_data)

                # 충분한 16진수 데이터가 있는 경우만 처리
                if len(hex_str) >= 32:  # 최소 16바이트 이상
                    try:
                        image_data = bytes.fromhex(hex_str)

                        # 이미지 형식이 없으면 데이터에서 감지
                        if not image_format:
                            image_format = self._detect_image_format(image_data)

                        # 유효한 이미지인지 확인
                        if image_format:
                            region = RTFBinaryRegion(
                                start_pos=start_pos,
                                end_pos=i,
                                bin_type="pict",
                                data_size=len(image_data),
                                image_format=image_format,
                                image_data=image_data
                            )
                            regions.append(region)

                            logger.debug(
                                f"Found \\pict region (hex): pos={start_pos}, "
                                f"hex_len={len(hex_str)}, format={image_format}"
                            )
                    except ValueError as e:
                        logger.debug(f"Failed to decode hex data at {start_pos}: {e}")

        except Exception as e:
            logger.warning(f"Error finding \\pict regions: {e}")

        logger.info(f"Found {len(regions)} \\pict regions (hex-encoded) in RTF")
        return regions

    def _detect_image_format(self, data: bytes) -> str:
        """
        바이너리 데이터의 이미지 형식을 감지합니다.

        Args:
            data: 이미지 바이너리 데이터

        Returns:
            이미지 형식 문자열 (jpeg, png, wmf 등) 또는 빈 문자열
        """
        if not data or len(data) < 4:
            return ""

        for signature, format_name in IMAGE_SIGNATURES.items():
            if data.startswith(signature):
                return format_name

        # JPEG 확장 체크 (EXIF 헤더 등)
        if len(data) >= 3:
            if data[0:2] == b'\xff\xd8':
                return 'jpeg'

        return ""

    def _process_images(self) -> None:
        """
        추출된 이미지를 MinIO에 업로드하고 태그를 생성합니다.
        """
        for region in self.binary_regions:
            if not region.image_data:
                continue

            # 지원 가능한 이미지 형식인지 확인
            # WMF, EMF는 PIL에서 지원하지 않을 수 있음
            supported_formats = {'jpeg', 'png', 'gif', 'bmp'}

            if region.image_format in supported_formats:
                minio_path = upload_image_to_minio(
                    region.image_data,
                    app_db=self.app_db,
                    processed_images=self.processed_images
                )

                if minio_path:
                    self.image_tags[region.start_pos] = f"\n[image:{minio_path}]\n"
                    logger.info(
                        f"Uploaded RTF image to MinIO: {minio_path} "
                        f"(format={region.image_format}, size={region.data_size})"
                    )
                else:
                    # 업로드 실패 시 빈 태그 (무시됨)
                    self.image_tags[region.start_pos] = ""
                    logger.warning(f"Image upload failed, removing (pos={region.start_pos})")
            else:
                # WMF, EMF 등 미지원 형식은 플레이스홀더
                if region.image_format:
                    logger.debug(
                        f"Skipping unsupported image format: {region.image_format}"
                    )
                self.image_tags[region.start_pos] = ""  # 빈 태그 (무시)

    def _remove_binary_data(self, content: bytes) -> bytes:
        """
        바이너리 데이터 영역을 제거한 콘텐츠를 생성합니다.

        \bin 태그와 바이너리 데이터를 이미지 태그로 치환하거나 제거합니다.

        Args:
            content: 원본 RTF 바이너리 콘텐츠

        Returns:
            정제된 콘텐츠
        """
        if not self.binary_regions:
            return content

        # 영역을 역순으로 정렬하여 뒤에서부터 치환 (위치 변경 방지)
        sorted_regions = sorted(self.binary_regions, key=lambda r: r.start_pos, reverse=True)

        result = bytearray(content)

        for region in sorted_regions:
            # 해당 영역을 빈 바이트로 치환 (완전히 제거)
            # 이미지 태그는 나중에 텍스트 레벨에서 삽입
            replacement = b''

            # 이미지 태그가 있으면 마커 삽입 (나중에 텍스트 처리 시 사용)
            if region.start_pos in self.image_tags:
                tag = self.image_tags[region.start_pos]
                if tag:
                    # 이미지 태그를 마커로 삽입 (ASCII 안전)
                    replacement = tag.encode('ascii', errors='replace')

            result[region.start_pos:region.end_pos] = replacement

        return bytes(result)

    def get_image_tag(self, position: int) -> str:
        """
        특정 위치의 이미지 태그를 반환합니다.

        Args:
            position: RTF 내 위치

        Returns:
            이미지 태그 문자열 또는 빈 문자열
        """
        return self.image_tags.get(position, "")


def preprocess_rtf_binary(
    content: bytes,
    app_db=None,
    processed_images: Optional[Set[str]] = None
) -> Tuple[bytes, Dict[int, str]]:
    """
    RTF 콘텐츠에서 바이너리 데이터를 전처리합니다.

    \bin 태그의 바이너리 데이터를 제거하고,
    이미지는 MinIO에 업로드하여 태그로 변환합니다.

    이 함수는 RTF 파서 전에 호출하여 바이너리 데이터로 인한
    텍스트 깨짐을 방지합니다.

    Args:
        content: RTF 파일 바이너리 콘텐츠
        app_db: 데이터베이스 연결 (optional)
        processed_images: 처리된 이미지 해시 집합 (optional)

    Returns:
        (정제된 콘텐츠, 위치->이미지태그 딕셔너리) 튜플

    Example:
        >>> with open('file.rtf', 'rb') as f:
        ...     raw_content = f.read()
        >>> clean_content, image_tags = preprocess_rtf_binary(raw_content, app_db)
        >>> # 이후 RTF 파서에 clean_content 전달
    """
    processor = RTFBinaryProcessor(app_db, processed_images)
    result = processor.process(content)
    return result.clean_content, result.image_tags


def extract_rtf_images(
    content: bytes,
    app_db=None,
    processed_images: Optional[Set[str]] = None
) -> List[str]:
    """
    RTF 콘텐츠에서 모든 이미지를 추출하여 MinIO에 업로드합니다.

    Args:
        content: RTF 파일 바이너리 콘텐츠
        app_db: 데이터베이스 연결 (optional)
        processed_images: 처리된 이미지 해시 집합 (optional)

    Returns:
        이미지 태그 리스트 (예: ["[image:bucket/uploads/hash.png]", ...])
    """
    processor = RTFBinaryProcessor(app_db, processed_images)
    result = processor.process(content)

    # 위치순으로 정렬된 이미지 태그 반환
    sorted_tags = sorted(result.image_tags.items(), key=lambda x: x[0])
    return [tag for pos, tag in sorted_tags if tag]
