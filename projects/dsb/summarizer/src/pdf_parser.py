"""PDF text extraction functionality."""

import re
from pathlib import Path
from typing import Dict

import fitz  # PyMuPDF

from .models import PaperText


def extract_paper_text(pdf_path: str) -> PaperText:
    """
    PDF에서 논문 텍스트를 추출합니다.

    Args:
        pdf_path: PDF 파일 경로

    Returns:
        PaperText: 추출된 텍스트 구조

    Raises:
        FileNotFoundError: PDF 파일이 없는 경우
        ValueError: PDF 파싱에 실패한 경우
    """
    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        raise FileNotFoundError(f"PDF 파일을 찾을 수 없습니다: {pdf_path}")

    try:
        # PDF 열기
        doc = fitz.open(pdf_path)
        full_text = ""

        # 모든 페이지 텍스트 추출
        for page_num in range(len(doc)):
            page = doc[page_num]
            full_text += page.get_text()

        doc.close()

        # 제목 추출 (첫 문장에서 찾기)
        title = extract_title(full_text)

        # 초록 추출
        abstract = extract_abstract(full_text)

        # 방법론 섹션 추출
        method = extract_method_section(full_text)

        return PaperText(
            title=title,
            abstract=abstract,
            method=method,
            full_text=full_text
        )

    except Exception as e:
        raise ValueError(f"PDF 파싱 중 오류 발생: {e}")


def extract_title(text: str) -> str:
    """
    텍스트에서 논문 제목을 추출합니다.
    """
    lines = text.split('\n')

    # 처음 몇 줄에서 가장 긴 텍스트를 제목으로 추정
    for i, line in enumerate(lines[:10]):
        line = line.strip()
        if len(line) > 20 and not line.startswith(('Abstract', 'Introduction', 'References')):
            # 숫자나 특수문자로 시작하지 않는 경우 제목으로 간주
            if not re.match(r'^[\d\W]', line):
                return line

    # fallback: 첫 번째 의미 있는 줄
    for line in lines[:20]:
        line = line.strip()
        if len(line) > 10:
            return line

    return "Unknown Title"


def extract_abstract(text: str) -> str:
    """
    텍스트에서 초록(Abstract) 섹션을 추출합니다.
    """
    # Abstract 섹션 찾기
    abstract_match = re.search(
        r'(?:Abstract|ABSTRACT)[\s\n]*([^]+?)(?:\n\s*(?:1\.|Introduction|Keywords|I\.))',
        text,
        re.IGNORECASE | re.MULTILINE
    )

    if abstract_match:
        abstract = abstract_match.group(1).strip()
        # 불필요한 줄바꿈 정리
        abstract = re.sub(r'\s+', ' ', abstract)
        return abstract

    # fallback: Abstract 키워드 뒤의 내용 추출
    abstract_start = re.search(r'(?:Abstract|ABSTRACT)[\s\n]*', text, re.IGNORECASE)
    if abstract_start:
        start_pos = abstract_start.end()
        # 다음 섹션 시작 전까지 추출
        next_section = re.search(r'\n\s*(?:1\.|Introduction|Keywords|I\.)', text[start_pos:])
        if next_section:
            end_pos = start_pos + next_section.start()
            abstract = text[start_pos:end_pos].strip()
        else:
            abstract = text[start_pos:start_pos+2000].strip()  # 최대 2000자

        abstract = re.sub(r'\s+', ' ', abstract)
        return abstract

    return "Abstract not found"


def extract_method_section(text: str) -> str:
    """
    텍스트에서 방법론(Method) 섹션을 추출합니다.
    """
    # 다양한 방법론 섹션 키워드
    method_patterns = [
        r'(?:3\.|Method|METHOD|Methodology|METHODOLOGY|Approach|APPROACH)[\s\n]*([^]+?)(?:\n\s*(?:4\.|5\.|Experiment|EXPERIMENT|Results|RESULTS|Discussion|DISCUSSION|Conclusion|CONCLUSION))',
        r'(?:2\.|Method|METHOD|Methodology|METHODOLOGY)[\s\n]*([^]+?)(?:\n\s*(?:3\.|4\.|Experiment|EXPERIMENT|Results|RESULTS|Discussion|DISCUSSION|Conclusion|CONCLUSION))',
    ]

    for pattern in method_patterns:
        method_match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if method_match:
            method = method_match.group(1).strip()
            # 너무 긴 경우 제한
            if len(method) > 10000:
                method = method[:10000] + "..."
            return method

    # fallback: Method 키워드로 시작하는 섹션 찾기
    method_start = re.search(r'(?:Method|METHOD|Methodology|METHODOLOGY|Approach|APPROACH)[\s\n]*', text, re.IGNORECASE)
    if method_start:
        start_pos = method_start.end()
        # 다음 주요 섹션 전까지 추출
        next_section = re.search(r'\n\s*(?:[4-9]\.|Experiment|EXPERIMENT|Results|RESULTS|Discussion|DISCUSSION|Conclusion|CONCLUSION)', text[start_pos:])
        if next_section:
            end_pos = start_pos + next_section.start()
            method = text[start_pos:end_pos].strip()
        else:
            method = text[start_pos:start_pos+5000].strip()  # 최대 5000자

        return method

    return "Method section not found"