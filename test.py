# test.py
"""
DocumentProcessor Test Script

HWP 파일을 사용하여 DocumentProcessor 기능 테스트
"""
import logging

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from libs.core.document_processor import DocumentProcessor


def main():
    # 테스트할 HWP 파일 경로
    file_path = r"C:\local_workspace\x2bee_github_push\Contextify\9.플래티어_품질관리_v1.1.hwp"

    # DocumentProcessor 인스턴스 생성 (OCR 엔진 없이)
    processor = DocumentProcessor()

    print("=" * 80)
    print("1. HWP -> Text 변환")
    print("=" * 80)

    # 1. HWP를 텍스트로 변환
    text = processor.extract_text(file_path, ocr_processing=False)
    print(text)

    print("\n" + "=" * 80)
    print("2. Chunk Text 수행 후 chunked_list 출력")
    print("=" * 80)

    # 2. 텍스트 청킹
    chunked_list = processor.chunk_text(
        text,
        chunk_size=1000,
        chunk_overlap=200,
        file_extension="hwp"
    )

    print(f"Total chunks: {len(chunked_list)}")
    print(f"Chunked list: {chunked_list}")

    print("\n" + "=" * 80)
    print("3. 각 청크 개별 출력")
    print("=" * 80)

    # 3. 청크 하나씩 출력
    for i, chunk in enumerate(chunked_list):
        print(f"\n--- Chunk {i + 1}/{len(chunked_list)} ---")
        print(f"Length: {len(chunk)} characters")
        print(chunk)
        print("-" * 40)


if __name__ == "__main__":
    main()
