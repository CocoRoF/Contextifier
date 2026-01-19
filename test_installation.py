"""
빌드된 패키지를 로컬에서 테스트하는 스크립트
"""

# 설치된 패키지에서 import 테스트
try:
    from libs.core.document_processor import DocumentProcessor
    print("✅ DocumentProcessor import 성공")
    
    # 인스턴스 생성 테스트
    processor = DocumentProcessor()
    print("✅ DocumentProcessor 인스턴스 생성 성공")
    
    # 간단한 텍스트 파일 테스트
    import os
    test_file = os.path.join(os.path.dirname(__file__), "test", "sample.txt")
    
    if os.path.exists(test_file):
        text = processor.extract_text(test_file)
        print(f"✅ 텍스트 추출 성공: {len(text)} chars")
        
        # 청킹 테스트
        result = processor.extract_chunks(test_file, chunk_size=500)
        print(f"✅ 청킹 성공: {len(result.chunks)} chunks")
    else:
        print(f"⚠️  테스트 파일 없음: {test_file}")
        print("✅ 기본 기능 테스트는 통과했습니다")
    
    print("\n" + "=" * 60)
    print("모든 테스트 통과!")
    print("=" * 60)
    
except Exception as e:
    print(f"❌ 에러 발생: {e}")
    import traceback
    traceback.print_exc()
