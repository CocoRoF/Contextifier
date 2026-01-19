# PyPI 배포 준비 완료! 🎉

Contextify가 PyPI 배포를 위해 준비되었습니다.

## 생성된 파일들

### 패키지 설정
- ✅ `pyproject.toml` - 프로젝트 메타데이터 및 의존성
- ✅ `MANIFEST.in` - 패키지에 포함할 파일 지정
- ✅ `LICENSE` - Apache 2.0 라이선스

### 문서
- ✅ `README.md` - 프로젝트 소개 및 사용법
- ✅ `QUICKSTART.md` - 빠른 시작 가이드
- ✅ `CHANGELOG.md` - 변경 이력
- ✅ `CONTRIBUTING.md` - 기여 가이드
- ✅ `BUILD_AND_PUBLISH.md` - 빌드 및 배포 가이드

### 빌드 결과
- ✅ `dist/contextify-1.0.0-py3-none-any.whl` (310 KB)
- ✅ `dist/contextify-1.0.0.tar.gz` (230 KB)
- ✅ 패키지 검증 완료 (twine check passed)

### 설정 예제
- ✅ `.pypirc.example` - PyPI 인증 설정 예제

## 배포 전 체크리스트

### 필수 작업
- [ ] **GitHub 저장소 생성** (아직 안 만들었다면)
  ```bash
  git init
  git add .
  git commit -m "Initial commit"
  git remote add origin https://github.com/yourusername/contextify.git
  git push -u origin main
  ```

- [ ] **pyproject.toml 업데이트**
  - `authors` 섹션에 실제 이름과 이메일 입력
  - `project.urls` 섹션에 실제 GitHub URL 입력

- [ ] **PyPI 계정 생성**
  - Production: https://pypi.org/account/register/
  - Test (권장): https://test.pypi.org/account/register/

- [ ] **API 토큰 생성**
  - PyPI 계정 설정 → API tokens → "Add API token"
  - 프로젝트별 또는 전체 계정용 토큰 생성
  - 토큰을 안전하게 보관

### Test PyPI에 먼저 배포 (권장)

```bash
# 1. Test PyPI에 업로드
C:/DOC_DMZ/Contextify/.venv/Scripts/python.exe -m twine upload --repository testpypi dist/*

# Username: __token__
# Password: pypi-... (your Test PyPI token)

# 2. Test PyPI에서 설치 테스트
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ contextify

# 3. 테스트
python -c "from libs.core.document_processor import DocumentProcessor; print('Success!')"
```

### PyPI에 배포 (프로덕션)

```bash
# 1. 최종 확인
C:/DOC_DMZ/Contextify/.venv/Scripts/python.exe -m twine check dist/*

# 2. PyPI에 업로드
C:/DOC_DMZ/Contextify/.venv/Scripts/python.exe -m twine upload dist/*

# Username: __token__
# Password: pypi-... (your PyPI token)

# 3. 설치 확인
pip install contextify

# 4. 테스트
python -c "from libs.core.document_processor import DocumentProcessor; print('Success!')"
```

## 사용자가 설치하는 방법

배포 후 사용자들은 다음과 같이 설치할 수 있습니다:

```bash
# pip으로 설치
pip install contextify

# uv로 설치
uv pip install contextify

# poetry로 설치
poetry add contextify
```

## 간단한 사용 예제

```python
from libs.core.document_processor import DocumentProcessor

processor = DocumentProcessor()

# 텍스트 추출
text = processor.extract_text("document.pdf")
print(text)

# 청킹
result = processor.extract_chunks("document.pdf", chunk_size=1000)
for chunk in result.chunks:
    print(chunk.text)
```

## 버전 업데이트 방법

새 버전을 배포할 때:

1. `pyproject.toml`에서 버전 변경
   ```toml
   version = "1.0.1"  # 또는 1.1.0, 2.0.0 등
   ```

2. `CHANGELOG.md` 업데이트

3. 빌드 및 배포
   ```bash
   # 이전 빌드 삭제
   Remove-Item -Recurse -Force dist

   # 새로 빌드
   C:/DOC_DMZ/Contextify/.venv/Scripts/python.exe -m build

   # 배포
   C:/DOC_DMZ/Contextify/.venv/Scripts/python.exe -m twine upload dist/*
   ```

## 문제 해결

### "Package already exists" 에러
- 버전 번호가 중복됨. `pyproject.toml`에서 버전을 올리고 다시 빌드

### Import 에러
- 패키지 구조 확인: `from libs.core.document_processor import DocumentProcessor`
- 전체 경로를 사용해야 함

### 의존성 에러
- `pyproject.toml`의 `dependencies` 섹션 확인
- 필요한 모든 패키지가 나열되어 있는지 확인

## 추가 자료

- [Python Packaging Guide](https://packaging.python.org/)
- [PyPI Help](https://pypi.org/help/)
- [Twine Documentation](https://twine.readthedocs.io/)
- [Semantic Versioning](https://semver.org/)

## 다음 단계

1. GitHub 저장소에 코드 푸시
2. Test PyPI에 배포하여 테스트
3. 문제 없으면 PyPI에 배포
4. PyPI 프로젝트 페이지 확인: https://pypi.org/project/contextify/
5. 사용자들이 `pip install contextify`로 설치할 수 있음!

축하합니다! 🎉
