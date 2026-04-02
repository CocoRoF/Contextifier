# Contextifier v0.2.5 — 성능, 보안, 에지 케이스 분석

> 분석 일자: 2025-04-01
> 분석 범위: 메모리 효율, 스레드 안전성, 보안 취약점, 인코딩 처리, 파일 손상 대응, 리소스 관리

---

## 1. 메모리 효율성 분석

### 1.1 현재 메모리 모델

모든 파이프라인 단계가 **전체 파일 인메모리(in-memory)** 전략을 사용한다.

```
파일 읽기 (bytes) → FileContext.file_data (전체 복사)
                  → FileContext.file_stream (BytesIO 래퍼, lazy)
                  → Converter.convert() → 포맷 객체 (추가 메모리)
                  → Preprocessor.preprocess() → PreprocessedData
                  → ContentExtractor.extract_all() → 텍스트 + 테이블 + 이미지
                  → Postprocessor.postprocess() → 최종 문자열
```

**메모리 피크 분석** (100MB PDF 기준):
```
file_data:      100 MB (원본 바이트)
file_stream:    100 MB (BytesIO — lazy이므로 미생성 가능)
fitz.Document:  ~200 MB (PyMuPDF 내부 표현)
텍스트 추출:     ~50 MB (문자열)
이미지 추출:     ~300 MB (디코딩된 이미지 바이트)
테이블 추출:     ~10 MB (TableData 객체)
──────────────────────────
예상 피크:      ~460-760 MB
```

### 1.2 포맷별 메모리 이슈

| 포맷 | 이슈 | 심각도 | 상세 |
|------|------|--------|------|
| **PDF Plus** | 전체 문서 + 복잡도 분석기 + 다중 라이브러리 | 🟠 | PyMuPDF + pdfplumber + pdfminer 동시 로딩 |
| **XLSX** | 전체 워크북 openpyxl 로딩 | 🟠 | 수만 행 스프레드시트에서 수 GB 가능 |
| **CSV/TSV** | 전체 파일 디코딩 | 🟡 | `content.decode()` 전체 수행 |
| **이미지** | 디코딩된 이미지 전체 보유 | 🟡 | TIFF/RAW 등 대형 이미지 |
| **DOCX** | python-docx Document 객체 | 🔵 | 일반적으로 작은 편 |

### 1.3 file_stream의 Lazy 개선

Phase 0에서 `file_stream`을 lazy로 변경하여 기본적으로 `None`. `BaseConverter._get_stream()`을 통해 필요 시에만 생성.

```python
# document_processor.py
file_context: FileContext = {
    "file_data": file_data,
    "file_stream": None,  # Lazy — 필요 시 생성
    ...
}

# pipeline/converter.py
def _get_stream(self, file_context: FileContext) -> io.BytesIO:
    stream = file_context.get("file_stream")
    if stream is None:
        stream = io.BytesIO(file_context["file_data"])
    else:
        stream.seek(0)
    return stream
```

✅ 이 개선으로 `file_data + file_stream` 이중 보유 문제는 해결됨.

### 1.4 개선 필요 영역

| 영역 | 현재 | 개선안 | 영향 |
|------|------|--------|------|
| 대형 PDF | 전체 로딩 | 페이지 단위 스트리밍 | 고 (PyMuPDF 지원) |
| 대형 XLSX | 전체 로딩 | openpyxl read_only 모드 | 고 |
| 대형 CSV | 전체 로딩 | 청크 단위 읽기 | 중 |
| 이미지 추출 | 전체 디코딩 보유 | 즉시 저장 후 해제 | 중 |
| file_data 보유 | 파이프라인 전체 | 변환 후 해제 (옵션) | 고 |

---

## 2. 스레드 안전성 분석

### 2.1 Thread-Safe 컴포넌트

| 컴포넌트 | 전략 | 안전성 |
|---------|------|--------|
| `ImageService` | `threading.local()` | ✅ 스레드별 독립 상태 |
| `TagService` | 불변 (regex 사전 컴파일) | ✅ 쓰기 없음 |
| `ChartService` | 불변 (차트 타입 맵) | ✅ 쓰기 없음 |
| `TableService` | 불변 (설정 참조만) | ✅ 쓰기 없음 |
| `MetadataService` | 불변 (라벨 딕셔너리) | ✅ 쓰기 없음 |
| `ProcessingConfig` | frozen dataclass | ✅ 불변 |

### 2.2 잠재적 Thread 이슈

| 컴포넌트 | 이슈 | 심각도 | 상세 |
|---------|------|--------|------|
| `LocalStorageBackend` | 동시 쓰기 Race Condition | 🟡 | 같은 파일명 동시 쓰기 시 |
| `_process_with_timeout()` | ThreadPoolExecutor 매 호출 생성 | 🟡 | `with TPE(max_workers=1)` 반복 생성 |
| `HandlerRegistry` | register() 비동기 안전하지 않음 | 🔵 | 초기화 시에만 호출되므로 실질적 이슈 아님 |
| `MemoryCacheBackend` | dict 동시 접근 | 🟡 | GIL 보호되지만 명시적 락 없음 |

### 2.3 _process_with_timeout() 문제

```python
def _process_with_timeout(self, file_context, *, timeout, **kwargs):
    # 매 호출마다 ThreadPoolExecutor 생성
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(self._execute_pipeline, ...)
        return future.result(timeout=timeout)
```

**문제점**:
- 타임아웃 사용 시 매번 새 ThreadPoolExecutor 생성/폐기
- 스레드 풀 생성 오버헤드 (~1-5ms)
- 고빈도 호출 시 GC 부하

**개선안**: 핸들러 또는 프로세서 레벨에서 단일 TPE 재사용

---

## 3. 보안 분석

### 3.1 HTML 인젝션

**🟠 TableService.format_as_html() — 이스케이프 누락**

```python
def format_as_html(self, table: TableData) -> str:
    content = self._clean_cell(cell.content)  # whitespace만 정리
    line_parts.append(f"<{tag}{attrs}>{content}</{tag}>")
    # → 셀 내용에 <script>alert(1)</script> 가능
```

**영향**:
- CSV 파일 등 사용자 입력이 포함된 문서에서 HTML 인젝션 가능
- 출력을 웹 페이지에 직접 삽입하는 경우 XSS 공격 벡터

**수정**: `_clean_cell()` 또는 `format_as_html()`에서 `html.escape()` 추가

### 3.2 HTML base64 이미지 크기 무제한

**🟠 HTML 핸들러 — 메모리 DoS**

```python
# html/content_extractor.py
# base64 이미지 디코딩 시 크기 제한 없음
# 수백 MB base64 이미지가 포함된 HTML → 메모리 폭발
```

**수정**: 크기 제한 + `max_image_size` 설정 도입

### 3.3 경로 순회 (Path Traversal)

| 위치 | 현재 상태 | 위험도 |
|------|----------|--------|
| `DocumentProcessor.extract_text()` | `os.path.exists()` 검사만 | 🟡 |
| `LocalStorageBackend.save()` | 경로 검증 없음 | 🟡 |
| `ImageService.save()` | 파일명에 `../` 가능 | 🟡 |

**수정**: 저장 경로를 base_directory 내로 제한 (`os.path.commonpath` 검증)

### 3.4 ZIP Bomb 방어

| 포맷 | ZIP 관련 | 현재 방어 |
|------|---------|----------|
| DOCX, PPTX, XLSX | ZIP 기반 | ❌ ZIP bomb 검사 없음 |
| HWPX | ZIP 기반 | ❌ ZIP bomb 검사 없음 |

**수정**: ZIP 해제 전 압축률 + 해제 크기 제한 검사

### 3.5 ReDoS (Regular Expression Denial of Service)

청킹 시스템에서 다수의 regex 패턴 사용. 사용자 입력이 regex에 직접 입력되지 않으므로 현재 위험은 낮음.
단, TagConfig의 prefix/suffix가 regex 특수문자를 포함할 경우 패턴 오류 가능.

**수정**: TagService에서 `re.escape()` 적용 여부 확인 필요

---

## 4. 인코딩 처리 분석

### 4.1 핸들러별 인코딩 전략

| 핸들러 | 감지 방법 | 기본 폴백 | 이슈 |
|--------|----------|----------|------|
| CSV | BOM 우선 → chardet → 설정 목록 | UTF-8 | 설정 순서에 따라 결과 변동 |
| TSV | CSV와 동일 | UTF-8 | 동일 |
| Text | BOM → chardet | UTF-8 | 신뢰도 낮은 chardet 결과 수용 |
| HTML | `<meta charset>` → BOM → chardet | UTF-8 | meta 태그 인코딩이 실제와 불일치 가능 |
| RTF | 제어 워드 인코딩 → CJK hex | - | 사이릴/아랍 인코딩 불확실 |
| DOC | UTF-16LE 가정 | - | **🟠 다른 인코딩 무시** |

### 4.2 chardet 한계

```python
# chardet 결과 예시
{'encoding': 'EUC-KR', 'confidence': 0.73}  # 73% 신뢰도
{'encoding': 'utf-8', 'confidence': 0.99}     # 99% 신뢰도
```

**이슈**: 신뢰도 임계값이 없음 → 73%짜리 결과도 그대로 사용
**수정**: 최소 신뢰도 설정 + 폴백 전략 체계화

### 4.3 인코딩 설정 표준화

```python
# 현재: 핸들러마다 다른 방식
csv_preprocessor:  config.format_options.get("csv", {}).get("encodings", [...])
text_converter:    config에서 읽는 방식 다름

# 필요: 통합 인코딩 설정
ProcessingConfig(
    encoding_detection=EncodingConfig(
        min_confidence=0.8,
        fallback_encodings=["utf-8", "utf-8-sig", "latin-1"],
        force_encoding=None,
    )
)
```

---

## 5. 파일 손상/에지 케이스 대응

### 5.1 손상 파일 처리

| 포맷 | 현재 대응 | 품질 |
|------|----------|------|
| PDF | fitz 예외 → ConversionError | ✅ |
| DOCX | python-docx 예외 → ConversionError | ✅ |
| DOC | OLE 실패 → raw text 스캐닝 폴백 | ✅ |
| PPTX | python-pptx 예외 → ConversionError | ✅ |
| PPT | OLE 실패 → raw text 스캐닝 폴백 | ✅ |
| XLSX | openpyxl 예외 → ConversionError | ✅ |
| XLS | xlrd 예외 → ConversionError | ✅ |
| HWP | 레코드 복구 (`_recovery.py`) | ✅ |
| CSV | 인코딩 폴백 체인 | ✅ |
| 기타 | handler-specific | 가변 |

### 5.2 비밀번호 보호 파일

| 포맷 | 비밀번호 지원 | 상세 |
|------|-------------|------|
| PDF | ✅ | `password` kwarg + PyMuPDF 인증 |
| DOCX | ❌ | python-docx 미지원 (Office encryption) |
| PPTX | ❌ | python-pptx 미지원 |
| XLSX | ❌ | openpyxl 미지원 |
| XLS | ❌ | xlrd 미지원 |
| HWP | ❌ | 구현 없음 |

**실무 영향**: 기업 문서의 상당수가 비밀번호 보호 → 처리 불가
**해결 방안**:
1. msoffcrypto-tool 라이브러리로 OOXML 복호화 전처리
2. HWP/HWPX는 암호화 플래그 검사 + 사용자 알림

### 5.3 빈 파일 / 최소 파일

| 시나리오 | 현재 대응 |
|---------|----------|
| 0바이트 파일 | ConversionError (file_data 비어있음) |
| 헤더만 있는 CSV | 빈 텍스트 반환 (정상) |
| 1페이지 빈 PDF | 페이지 태그만 반환 (정상) |
| 매직 바이트 불일치 | 위임 로직으로 올바른 핸들러 재시도 |

### 5.4 초대형 파일

| 제한 | 현재 상태 | 상세 |
|------|----------|------|
| 최대 파일 크기 | `_MAX_FILE_SIZE` 존재 | DocumentProcessor에서 사전 검사 |
| 최대 페이지 수 | 없음 | PDF 10,000 페이지도 전체 처리 시도 |
| 최대 시트 수 | 없음 | XLSX 100시트도 전체 처리 |
| 최대 행 수 | 없음 | CSV 백만 행도 전체 로딩 |
| 처리 타임아웃 | `timeout` 파라미터 존재 | BaseHandler.process(timeout=) |

### 5.5 무한 재귀 방어

| 위치 | 방어 수단 | 상태 |
|------|----------|------|
| PPTX 그룹 셰이프 | `MAX_GROUP_DEPTH = 20` | ✅ |
| BaseHandler 위임 | 위임 체인 깊이 제한 없음 | 🟡 |
| 청킹 재귀 | 재귀 없음 (반복 기반) | ✅ |

**🟡 위임 무한 루프 위험**:
```
DOC → RTF → ... → DOC  (이론적 가능성)
```
→ 위임 깊이 제한 또는 방문 핸들러 추적 필요

---

## 6. 리소스 관리 분석

### 6.1 파일 핸들 관리

```python
# BaseHandler._execute_pipeline()
try:
    converted = self._converter.convert(file_context, **kwargs)
    # ... 파이프라인 실행 ...
finally:
    if converted is not None:
        self._converter.close(converted)  # ✅ finally 보장
```

**현재 상태**: ✅ BaseHandler의 `_execute_pipeline()`에서 finally로 close() 보장

### 6.2 잠재적 리소스 누수

| 위치 | 이슈 | 심각도 |
|------|------|--------|
| Converter.convert() 내부 | 변환 중간 실패 시 임시 객체 미정리 | 🟡 |
| OLE/ZIP 스트림 | 일부 핸들러에서 명시적 close 없음 | 🟡 |
| ImageService 스레드 상태 | 장기 서비스에서 clear_state() 미호출 시 축적 | 🟡 |

### 6.3 GC 및 메모리 해제

- `file_data` (바이트): 파이프라인 완료 후 FileContext 참조가 해제되면 GC
- `fitz.Document`: `close()` 메서드로 명시적 해제 (finally에서 호출)
- 이미지 바이트: ImageService.save() 후 참조 해제 의존

---

## 7. 동시성 & 확장성

### 7.1 현재 동시성 모델

```
DocumentProcessor (단일 스레드)
    └── extract_text() — 블로킹

AsyncDocumentProcessor (asyncio)
    └── asyncio.to_thread() — GIL 제약 있음

CachedDocumentProcessor (단일 스레드 + 캐시)
    └── 캐시 미스 시 동기 처리
```

### 7.2 확장성 제한

| 제한 | 현재 | 영향 |
|------|------|------|
| GIL | Python GIL | CPU-bound 처리 병렬화 불가 |
| asyncio 제한 | `to_thread()` 기반 | 진정한 비동기 I/O 아님 |
| 배치 처리 | `extract_batch()` 있음 | Semaphore로 동시성 제한 가능 |
| 분산 처리 | 미지원 | 대규모 배치 시 분산 걸림돌 |

### 7.3 확장성 개선 방안

1. **ProcessPoolExecutor**: CPU-bound 처리 (PDF 파싱, 테이블 분석)
2. **Celery/RQ 통합**: 분산 큐 기반 배치 처리
3. **Streaming Pipeline**: 페이지 단위 처리로 메모리 분산
4. **Result Caching 확장**: process(), extract_chunks()도 캐싱

---

## 8. 종합 위험 매트릭스

| # | 영역 | 이슈 | 심각도 | 발생 확률 | 영향 |
|---|------|------|--------|----------|------|
| R1 | 보안 | format_as_html() 이스케이프 없음 | 🟠 | 높음 | HTML 인젝션 |
| R2 | 보안 | HTML base64 크기 무제한 | 🟠 | 중간 | 메모리 DoS |
| R3 | 보안 | 저장 경로 순회 가능 | 🟡 | 낮음 | 임의 경로 쓰기 |
| R4 | 보안 | ZIP bomb 미방어 | 🟡 | 낮음 | 메모리/디스크 폭발 |
| R5 | 성능 | 대형 파일 전체 인메모리 | 🟠 | 중간 | OOM |
| R6 | 성능 | TPE 매 호출 생성 | 🟡 | 낮음 | 오버헤드 |
| R7 | 안정성 | 위임 무한 루프 | 🟡 | 매우 낮음 | 스택 오버플로 |
| R8 | 안정성 | MemoryCacheBackend 동시접근 | 🟡 | 낮음 | 캐시 데이터 오염 |
| R9 | 안정성 | 비밀번호 파일 미지원 | 🟡 | 높음 | 처리 불가 |
| R10 | 인코딩 | DOC UTF-16LE 고정 가정 | 🟡 | 중간 | 텍스트 깨짐 |
