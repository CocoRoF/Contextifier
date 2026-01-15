# service/document_processor/processor/doc_helpers/rtf_text_cleaner.py
"""
RTF 텍스트 정리 유틸리티

RTF 제어 코드 제거 및 텍스트 정리 관련 함수들을 제공합니다.
"""
import re
from typing import List

from libs.core.processor.doc_helpers.rtf_constants import (
    SHAPE_PROPERTY_NAMES,
)
from libs.core.processor.doc_helpers.rtf_decoder import (
    decode_bytes,
)


def clean_rtf_text(text: str, encoding: str = "cp949") -> str:
    """
    RTF 제어 코드를 안전하게 제거하고 순수 텍스트만 추출합니다.

    토큰 기반 파싱으로 내용 유실을 방지합니다.

    Args:
        text: RTF 텍스트
        encoding: 사용할 인코딩

    Returns:
        정리된 텍스트
    """
    if not text:
        return ""

    # 전처리: 이미지 태그 보호 (임시 마커로 치환)
    image_tags = []
    def save_image_tag(m):
        image_tags.append(m.group())
        return f'\x00IMG{len(image_tags)-1}\x00'

    text = re.sub(r'\[image:[^\]]+\]', save_image_tag, text)

    # 전처리: Shape 속성 제거 ({\sp{\sn name}{\sv value}} 형식)
    text = re.sub(r'\{\\sp\{\\sn\s*\w+\}\{\\sv\s*[^}]*\}\}', '', text)

    # Shape 속성이 직접 출력된 경우도 제거 (shapeType202fFlipH0... 형태)
    text = re.sub(r'shapeType\d+[a-zA-Z0-9]+(?:posrelh\d+posrelv\d+)?', '', text)

    # \shp 관련 제어 워드 제거
    text = re.sub(r'\\shp(?:inst|txt|left|right|top|bottom|bx\w+|by\w+|wr\d+|fblwtxt\d+|z\d+|lid\d+)\b\d*', '', text)

    result = []
    i = 0
    n = len(text)

    while i < n:
        ch = text[i]

        # 이미지 태그 마커 복원
        if ch == '\x00' and i + 3 < n and text[i+1:i+4] == 'IMG':
            # \x00IMGn\x00 패턴 찾기
            end_idx = text.find('\x00', i + 4)
            if end_idx != -1:
                try:
                    tag_idx = int(text[i+4:end_idx])
                    result.append(image_tags[tag_idx])
                    i = end_idx + 1
                    continue
                except (ValueError, IndexError):
                    pass

        if ch == '\\':
            # 제어 워드 또는 제어 기호
            if i + 1 < n:
                next_ch = text[i + 1]

                # 특수 이스케이프 처리
                if next_ch == '\\':
                    result.append('\\')
                    i += 2
                    continue
                elif next_ch == '{':
                    result.append('{')
                    i += 2
                    continue
                elif next_ch == '}':
                    result.append('}')
                    i += 2
                    continue
                elif next_ch == '~':
                    result.append('\u00A0')  # non-breaking space
                    i += 2
                    continue
                elif next_ch == '-':
                    result.append('\u00AD')  # soft hyphen
                    i += 2
                    continue
                elif next_ch == '_':
                    result.append('\u2011')  # non-breaking hyphen
                    i += 2
                    continue
                elif next_ch == "'":
                    # hex escape \'XX
                    if i + 3 < n:
                        try:
                            hex_val = text[i+2:i+4]
                            byte_val = int(hex_val, 16)
                            # 단일 바이트 디코딩 시도
                            try:
                                result.append(bytes([byte_val]).decode(encoding))
                            except:
                                try:
                                    result.append(bytes([byte_val]).decode('cp1252'))
                                except:
                                    pass
                            i += 4
                            continue
                        except (ValueError, IndexError):
                            pass
                    i += 1
                    continue
                elif next_ch == '*':
                    # \* - destination 마커, 건너뛰기
                    i += 2
                    continue
                elif next_ch.isalpha():
                    # 제어 워드: \word[N][delimiter]
                    j = i + 1
                    while j < n and text[j].isalpha():
                        j += 1

                    control_word = text[i+1:j]

                    # 숫자 파라미터 스킵
                    while j < n and (text[j].isdigit() or text[j] == '-'):
                        j += 1

                    # 구분자 처리 (공백은 제어 워드의 일부)
                    if j < n and text[j] == ' ':
                        j += 1

                    # 특별 처리가 필요한 제어 워드
                    if control_word in ('par', 'line'):
                        result.append('\n')
                    elif control_word == 'tab':
                        result.append('\t')
                    elif control_word == 'u':
                        # 유니코드: \uN?
                        # 이미 파라미터를 스킵했으므로 다시 파싱
                        um = re.match(r'\\u(-?\d+)\??', text[i:])
                        if um:
                            try:
                                code = int(um.group(1))
                                if code < 0:
                                    code += 65536
                                result.append(chr(code))
                            except:
                                pass
                            j = i + um.end()
                    # 다른 제어 워드는 무시

                    i = j
                    continue

            i += 1
        elif ch == '{' or ch == '}':
            # 중괄호는 건너뛰기
            i += 1
        elif ch == '\r' or ch == '\n':
            # RTF에서 줄바꿈 문자는 무시 (\par가 실제 줄바꿈)
            i += 1
        else:
            # 일반 텍스트
            result.append(ch)
            i += 1

    # 최종 정리
    text_result = ''.join(result)

    # Shape 속성 이름 제거
    shape_name_pattern = r'\b(' + '|'.join(SHAPE_PROPERTY_NAMES) + r')\b'
    text_result = re.sub(shape_name_pattern, '', text_result)

    # 숫자만 있는 쓰레기 제거 (예: -231, -1, -5 등)
    text_result = re.sub(r'\s*-\d+\s*', ' ', text_result)

    # Binary/Hex 데이터 제거
    text_result = _remove_hex_outside_image_tags(text_result)

    # 여러 공백을 하나로
    text_result = re.sub(r'\s+', ' ', text_result)

    return text_result.strip()


def _remove_hex_outside_image_tags(text: str) -> str:
    """이미지 태그 외부의 긴 hex 문자열만 제거"""
    # 이미지 태그 위치 찾기
    protected_ranges = []
    for m in re.finditer(r'\[image:[^\]]+\]', text):
        protected_ranges.append((m.start(), m.end()))

    if not protected_ranges:
        # 이미지 태그가 없으면 그냥 제거
        return re.sub(r'(?<![a-zA-Z])[0-9a-fA-F]{32,}(?![a-zA-Z])', '', text)

    # 이미지 태그 외부에서만 hex 제거
    result = []
    last_end = 0
    for start, end in protected_ranges:
        # 보호 영역 전 부분에서 hex 제거
        before = text[last_end:start]
        before = re.sub(r'(?<![a-zA-Z])[0-9a-fA-F]{32,}(?![a-zA-Z])', '', before)
        result.append(before)
        # 보호 영역(이미지 태그)은 그대로 유지
        result.append(text[start:end])
        last_end = end
    # 마지막 보호 영역 이후
    after = text[last_end:]
    after = re.sub(r'(?<![a-zA-Z])[0-9a-fA-F]{32,}(?![a-zA-Z])', '', after)
    result.append(after)
    return ''.join(result)


def remove_destination_groups(content: str) -> str:
    r"""
    RTF destination 그룹 {\*\destination...}을 제거합니다.

    문서 끝에 나타나는 themedata, colorschememapping, latentstyles, datastore 등을
    제거하여 메타데이터가 텍스트로 추출되는 것을 방지합니다.

    Args:
        content: RTF 콘텐츠

    Returns:
        destination 그룹이 제거된 콘텐츠
    """
    from libs.core.processor.doc_helpers.rtf_constants import (
        SKIP_DESTINATIONS,
        IMAGE_DESTINATIONS,
    )

    result = []
    i = 0
    n = len(content)

    while i < n:
        # {\* 패턴 감지
        if content[i:i+3] == '{\\*':
            # destination 이름 추출
            j = i + 3
            while j < n and content[j] in ' \t\r\n':
                j += 1

            if j < n and content[j] == '\\':
                # 제어 워드 추출
                k = j + 1
                while k < n and content[k].isalpha():
                    k += 1
                ctrl_word = content[j+1:k]

                if ctrl_word in SKIP_DESTINATIONS:
                    # 이 그룹 전체를 건너뛰기
                    depth = 1
                    i += 1  # '{' 다음으로
                    while i < n and depth > 0:
                        if content[i] == '{':
                            depth += 1
                        elif content[i] == '}':
                            depth -= 1
                        i += 1
                    continue

                if ctrl_word in IMAGE_DESTINATIONS:
                    # 이미지 태그는 보존하면서 그룹 제거
                    depth = 1
                    group_start = i
                    i += 1  # '{' 다음으로
                    while i < n and depth > 0:
                        if content[i] == '{':
                            depth += 1
                        elif content[i] == '}':
                            depth -= 1
                        i += 1

                    # 그룹 내에서 유효한 이미지 태그만 추출
                    group_content = content[group_start:i]
                    image_tag_match = re.search(r'\[image:[^\]]+\]', group_content)
                    if image_tag_match:
                        tag = image_tag_match.group()
                        # 유효한 태그인지 확인
                        if '/uploads/.' not in tag and 'uploads/.' not in tag:
                            result.append(tag)
                    continue

        result.append(content[i])
        i += 1

    return ''.join(result)


def remove_shape_groups(content: str) -> str:
    """
    Shape 그룹을 제거하되, shptxt 내의 텍스트는 보존합니다.

    RTF Shape 구조:
    {\\shp{\\*\\shpinst...{\\sp{\\sn xxx}{\\sv yyy}}...{\\shptxt 실제텍스트}}}

    Args:
        content: RTF 콘텐츠

    Returns:
        Shape 그룹이 정리된 콘텐츠
    """
    result = []
    i = 0

    while i < len(content):
        # \shp 시작 감지
        if content[i:i+5] == '{\\shp' or content[i:i+10] == '{\\*\\shpinst':
            # Shape 그룹 시작
            # shptxt 내용만 추출하고 나머지는 건너뛰기
            depth = 1
            start = i
            i += 1
            shptxt_content = []
            in_shptxt = False
            shptxt_depth = 0

            while i < len(content) and depth > 0:
                if content[i] == '{':
                    # \shptxt 시작 확인
                    if content[i:i+8] == '{\\shptxt':
                        in_shptxt = True
                        shptxt_depth = depth + 1
                        i += 8  # '{\\shptxt' 건너뛰기
                        continue
                    depth += 1
                elif content[i] == '}':
                    if in_shptxt and depth == shptxt_depth:
                        in_shptxt = False
                    depth -= 1
                elif in_shptxt:
                    shptxt_content.append(content[i])
                i += 1

            # shptxt 내용이 있으면 추가
            if shptxt_content:
                shptxt_text = ''.join(shptxt_content)
                result.append(shptxt_text)
        else:
            result.append(content[i])
            i += 1

    return ''.join(result)


def remove_shape_property_groups(content: str) -> str:
    """
    Shape 속성 그룹 {\\sp{\\sn xxx}{\\sv yyy}}를 제거합니다.

    Args:
        content: RTF 콘텐츠

    Returns:
        Shape 속성이 제거된 콘텐츠
    """
    # {\\sp{\\sn ...}{\\sv ...}} 패턴 제거
    content = re.sub(r'\{\\sp\{\\sn\s*[^}]*\}\{\\sv\s*[^}]*\}\}', '', content)

    # 개별 {\\sp ...} 패턴도 제거
    content = re.sub(r'\{\\sp\s*[^}]*\}', '', content)

    # {\\sn ...} 패턴 제거
    content = re.sub(r'\{\\sn\s*[^}]*\}', '', content)

    # {\\sv ...} 패턴 제거
    content = re.sub(r'\{\\sv\s*[^}]*\}', '', content)

    return content


def remove_shprslt_blocks(content: str) -> str:
    r"""
    \shprslt{...} 블록을 제거합니다.

    Word는 Shape (도형/테이블)를 \shp 블록으로 저장하고,
    이전 버전 호환성을 위해 \shprslt 블록에 동일한 내용을 중복 저장합니다.

    Args:
        content: RTF 콘텐츠

    Returns:
        \shprslt 블록이 제거된 콘텐츠
    """
    result = []
    i = 0
    pattern = '\\shprslt'

    while i < len(content):
        # \shprslt 찾기
        idx = content.find(pattern, i)
        if idx == -1:
            result.append(content[i:])
            break

        # \shprslt 전까지 추가
        result.append(content[i:idx])

        # \shprslt{ 다음의 중괄호 블록 건너뛰기
        brace_start = content.find('{', idx)
        if brace_start == -1:
            # 중괄호가 없으면 \shprslt만 건너뛰기
            i = idx + len(pattern)
            continue

        # 매칭되는 닫는 중괄호 찾기
        depth = 1
        j = brace_start + 1
        while j < len(content) and depth > 0:
            if content[j] == '{':
                depth += 1
            elif content[j] == '}':
                depth -= 1
            j += 1

        i = j

    return ''.join(result)
