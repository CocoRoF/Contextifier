from bs4 import BeautifulSoup
import os
from pathlib import Path

def clean_html_file(html_content, output_file_path=None):
    """
    HTML 파일을 읽어서 스타일을 제거하고 텍스트와 표만 남긴 후 저장
    """
    try:
        soup = BeautifulSoup(html_content, 'html.parser')

        # 1. 불필요한 태그들 완전 제거
        print("🧹 불필요한 태그 제거 중...")
        unwanted_tags = ['script', 'style', 'link', 'meta', 'noscript', 'iframe', 'img']
        for tag_name in unwanted_tags:
            for tag in soup.find_all(tag_name):
                tag.decompose()

        # 2. 모든 태그의 스타일 관련 속성 제거
        print("✨ 스타일 속성 제거 중...")
        for tag in soup.find_all(True):
            attrs_to_remove = ['style', 'class', 'id', 'width', 'height',
                             'bgcolor', 'color', 'font-family', 'font-size',
                             'margin', 'padding', 'border', 'background', 'face', 'size', 'align','lang']

            for attr in attrs_to_remove:
                if tag.has_attr(attr):
                    del tag[attr]

        # 3. 테이블 병합 셀 처리 및 빈 칸 채우기
        print("📊 테이블 병합 셀 처리 중...")
        for table in soup.find_all('table'):
            _process_table_merged_cells(table, soup)

        # 4. 빈 태그들 제거
        print("🗑️  빈 태그 제거 중...")
        for tag in soup.find_all():
            if (not tag.get_text(strip=True) and
                not tag.find_all() and
                tag.name not in ['br', 'hr', 'img']):
                tag.decompose()

        # 5. 불필요한 서식 태그만 제거 (공백은 보존)
        for tag_name in ['font', 'u', 'b']:
            for tag in soup.find_all(tag_name):
                tag.unwrap()  # 태그는 제거하되 내용은 보존

        # 6. HTML을 문자열로 변환 (prettify 사용하지 않음)
        cleaned_html = str(soup)

        # 7. 연속된 공백만 정리 (단일 공백은 보존)
        import re
        cleaned_html = re.sub(r'\s+', ' ', cleaned_html)
        cleaned_html = re.sub(r'>\s+<', '><', cleaned_html)  # 태그 사이 공백만 제거
        cleaned_html = cleaned_html.replace('<p>', '').replace('</p>', '').replace('</span>', '').replace('<span>', '')    # 빈 <p> 태그 제거

        return cleaned_html

    except Exception as e:
        print(f"❌ 오류 발생: {str(e)}")
        return None

def _process_table_merged_cells(table, soup):
    """
    테이블의 병합된 셀을 풀고 빈 셀을 'None'으로 채우는 함수
    """
    # 모든 행을 리스트로 변환
    rows = table.find_all('tr')
    if not rows:
        return

    # 테이블을 2차원 배열로 변환하여 처리
    grid = []
    max_cols = 0

    # 1단계: 기존 테이블을 그리드로 변환
    for row_idx, row in enumerate(rows):
        cells = row.find_all(['td', 'th'])
        grid.append([])
        col_idx = 0

        for cell in cells:
            # 이미 채워진 열은 건너뛰기
            while col_idx < len(grid[row_idx]) and grid[row_idx][col_idx] is not None:
                col_idx += 1

            # 현재 셀의 내용
            cell_text = cell.get_text(strip=True)
            if not cell_text or cell_text == '-':
                cell_text = 'None'

            # colspan, rowspan 값 가져오기
            colspan = int(cell.get('colspan', 1))
            rowspan = int(cell.get('rowspan', 1))

            # 그리드에 셀 내용 채우기
            for r in range(rowspan):
                target_row_idx = row_idx + r

                # 필요한 만큼 행을 추가
                while len(grid) <= target_row_idx:
                    grid.append([])

                # 필요한 만큼 열을 추가
                while len(grid[target_row_idx]) < col_idx + colspan:
                    grid[target_row_idx].append(None)

                # 셀 내용을 해당 영역에 복사
                for c in range(colspan):
                    if col_idx + c < len(grid[target_row_idx]):
                        grid[target_row_idx][col_idx + c] = cell_text

            col_idx += colspan

        # 최대 열 수 업데이트
        max_cols = max(max_cols, len(grid[row_idx]))

    # 2단계: 모든 행의 길이를 맞추고 빈 셀 채우기
    for row in grid:
        while len(row) < max_cols:
            row.append('None')

        # None인 셀들을 'None'으로 변경
        for i in range(len(row)):
            if row[i] is None:
                row[i] = 'None'

    # 3단계: 기존 테이블 내용을 새로운 그리드로 교체
    # 기존 행들 제거
    for row in table.find_all('tr'):
        row.decompose()

    # 새로운 행들 추가
    for grid_row in grid:
        new_row = soup.new_tag('tr')
        for cell_text in grid_row:
            new_cell = soup.new_tag('td')
            new_cell.string = cell_text
            new_row.append(new_cell)
        table.append(new_row)
