"""PPT/PPTX를 PDF로 변환하는 유틸리티 함수"""

import os
import subprocess

class PptToPdfConversionError(Exception):
    """PPT to PDF 변환 중 발생하는 예외"""

    def __init__(self, message: str, status_code: int = 500):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)


def convert_ppt_to_pdf(input_path: str, output_dir: str) -> str:
    """
    PPT/PPTX 파일을 PDF로 변환합니다.

    Args:
        input_path: 변환할 PPT/PPTX 파일 경로
        output_dir: PDF 출력 디렉토리

    Returns:
        변환된 PDF 파일 경로

    Raises:
        PptToPdfConversionError: 변환 실패 시
    """
    input_name = os.path.basename(input_path)

    cmd = [
        "soffice",
        "--headless",
        "--convert-to",
        "pdf",
        "--outdir",
        output_dir,
        input_path,
    ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError as exc:
        raise PptToPdfConversionError(
            "LibreOffice(soffice) is not installed or not in PATH",
            status_code=500,
        ) from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="ignore").strip()
        raise PptToPdfConversionError(
            f"Conversion failed: {stderr}",
            status_code=500,
        ) from exc

    # LibreOffice는 기본적으로 확장자를 제거하고 .pdf를 붙임
    default_pdf_name = f"{os.path.splitext(input_name)[0]}.pdf"
    default_pdf_path = os.path.join(output_dir, default_pdf_name)

    if not os.path.exists(default_pdf_path):
        raise PptToPdfConversionError(
            "Converted PDF not found",
            status_code=500,
        )

    # 원본 파일명.pdf 형태로 변경 (예: test.pptx -> test.pptx.pdf)
    final_pdf_name = f"{input_name}.pdf"
    final_pdf_path = os.path.join(output_dir, final_pdf_name)

    if default_pdf_path != final_pdf_path:
        os.replace(default_pdf_path, final_pdf_path)

    return final_pdf_path
