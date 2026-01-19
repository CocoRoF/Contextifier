# test.py
"""
DocumentProcessor Test Script - extract_chunks 테스트
"""
import logging
import sys
sys.path.insert(0, r"C:\Users\USER\Desktop\xgen\Contextify")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def main():
    from libs.core.document_processor import DocumentProcessor
    
    file_path = r"C:\Users\USER\Desktop\xgen\Contextify\test\이미용 부속서_00.docx"
    processor = DocumentProcessor()

    print("=" * 80)
    print("extract_chunks 테스트")
    print("=" * 80)

    chunks = processor.extract_chunks(
        file_path,
        chunk_size=1000,
        chunk_overlap=200
    )

    print(f"Total chunks: {len(chunks)}")

    for i, chunk in enumerate(chunks):
        print(f"\n--- Chunk {i + 1}/{len(chunks)} ---")
        print(f"Length: {len(chunk)} characters")
        print(chunk)
        print("-" * 40)


if __name__ == "__main__":
    main()
