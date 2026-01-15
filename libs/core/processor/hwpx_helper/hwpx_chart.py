# hwpx_helper/hwpx_chart.py
"""
HWPX 차트 추출

HWPX 문서에서 차트 데이터를 추출합니다.
OOXML 차트 XML 파일과 OLE 객체를 처리합니다.
"""
import logging
import os
import zlib
import zipfile
from typing import List, Set

from .hwpx_constants import SKIP_IMAGE_EXTENSIONS

# hwp_helper에서 ChartHelper import
try:
    from ..hwp_helper import ChartHelper
    CHART_HELPER_AVAILABLE = True
except ImportError:
    CHART_HELPER_AVAILABLE = False

logger = logging.getLogger("document-processor")


def extract_charts_from_hwpx(zf: zipfile.ZipFile, app_db=None) -> List[str]:
    """
    HWPX (ZIP) 파일에서 차트를 추출합니다.

    HWPX stores charts in various formats:
    1. OOXML chart XML files in Chart/ or Charts/ directory (preferred)
    2. OLE objects in BinData/ directory (fallback, may be duplicate)

    Args:
        zf: 열린 ZipFile 객체
        app_db: 데이터베이스 연결 (이미지 업로드용)

    Returns:
        포맷된 차트 데이터 문자열 목록
    """
    if not CHART_HELPER_AVAILABLE:
        logger.warning("ChartHelper not available, skipping chart extraction")
        return []

    chart_results = []
    processed_chart_hashes: Set[str] = set()  # 중복 방지용

    try:
        namelist = zf.namelist()

        # 1. Look for OOXML chart files first (Chart/chart*.xml pattern)
        chart_results.extend(
            _extract_ooxml_charts(zf, namelist, processed_chart_hashes, app_db)
        )

        # 2. Check BinData for OLE chart objects (fallback)
        chart_results.extend(
            _extract_ole_charts(zf, namelist, processed_chart_hashes, app_db)
        )

    except Exception as e:
        logger.warning(f"Error extracting charts from HWPX: {e}")

    return chart_results


def _extract_ooxml_charts(
    zf: zipfile.ZipFile,
    namelist: List[str],
    processed_chart_hashes: Set[str],
    app_db
) -> List[str]:
    """
    OOXML 형식의 차트 파일을 추출합니다.

    Args:
        zf: 열린 ZipFile 객체
        namelist: ZIP 파일의 파일 목록
        processed_chart_hashes: 처리된 차트 해시 집합
        app_db: 데이터베이스 연결

    Returns:
        포맷된 차트 데이터 문자열 목록
    """
    chart_results = []

    chart_files = [
        f for f in namelist
        if (f.startswith('Chart/') and f.endswith('.xml'))
        or (f.startswith('Contents/Charts/') and f.endswith('.xml'))
        or (f.startswith('Charts/') and f.endswith('.xml'))
    ]

    for chart_file in chart_files:
        try:
            with zf.open(chart_file) as f:
                chart_xml = f.read()

            # Try to parse as OOXML chart
            chart_data = ChartHelper.parse_ooxml_chart(chart_xml)

            if chart_data:
                # 중복 방지: 시리즈 데이터 해시
                chart_hash = str(chart_data.get('series', []))
                if chart_hash in processed_chart_hashes:
                    logger.debug(f"Skipping duplicate chart: {chart_file}")
                    continue
                processed_chart_hashes.add(chart_hash)

                # process_chart 사용 (테이블 우선, 실패 시 이미지)
                formatted = ChartHelper.process_chart(chart_data, app_db)
                if formatted:
                    chart_results.append(formatted)
                    logger.info(f"Extracted chart from HWPX: {chart_file}")

        except Exception as e:
            logger.debug(f"Error reading chart file {chart_file}: {e}")
            continue

    return chart_results


def _extract_ole_charts(
    zf: zipfile.ZipFile,
    namelist: List[str],
    processed_chart_hashes: Set[str],
    app_db
) -> List[str]:
    """
    BinData에서 OLE 차트 객체를 추출합니다 (폴백).

    Args:
        zf: 열린 ZipFile 객체
        namelist: ZIP 파일의 파일 목록
        processed_chart_hashes: 처리된 차트 해시 집합
        app_db: 데이터베이스 연결

    Returns:
        포맷된 차트 데이터 문자열 목록
    """
    chart_results = []

    bindata_files = [
        f for f in namelist
        if f.startswith('BinData/') and not f.endswith('/')
    ]

    for bindata_file in bindata_files:
        ext = os.path.splitext(bindata_file)[1].lower()

        # Skip known image formats
        if ext in SKIP_IMAGE_EXTENSIONS:
            continue

        try:
            with zf.open(bindata_file) as f:
                data = f.read()

            # Try to decompress if needed
            original_data = data
            try:
                data = zlib.decompress(data, -15)
            except:
                try:
                    data = zlib.decompress(data)
                except:
                    data = original_data

            # Check if this is an OLE chart object
            chart_data = ChartHelper.extract_chart_from_ole_stream(data)

            if chart_data:
                # 중복 방지: 시리즈 데이터 해시
                chart_hash = str(chart_data.get('series', []))
                if chart_hash in processed_chart_hashes:
                    logger.debug(f"Skipping duplicate chart from OLE: {bindata_file}")
                    continue
                processed_chart_hashes.add(chart_hash)

                # process_chart 사용 (테이블 우선, 실패 시 이미지)
                formatted = ChartHelper.process_chart(chart_data, app_db)
                if formatted:
                    chart_results.append(formatted)
                    logger.info(f"Extracted chart from HWPX BinData: {bindata_file}")

        except Exception as e:
            logger.debug(f"Error reading BinData file {bindata_file}: {e}")
            continue

    return chart_results
