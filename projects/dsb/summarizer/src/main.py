"""Main execution file for paper visualizer."""

import asyncio
import os
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from .analyzer import analyze_paper
from .pdf_parser import extract_paper_text
from .visualizer import generate_markdown_report, validate_mermaid_syntax

console = Console()


@click.command()
@click.option(
    "--input",
    "-i",
    required=True,
    help="PDF 파일 경로 또는 arXiv URL"
)
@click.option(
    "--output",
    "-o",
    default="output",
    help="출력 디렉토리 (기본값: output)"
)
@click.option(
    "--format",
    "-f",
    default="markdown",
    type=click.Choice(["markdown", "mermaid", "both"]),
    help="출력 형식 (기본값: markdown)"
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="상세 출력"
)
def cli(input: str, output: str, format: str, verbose: bool) -> None:
    """
    논문 방법론 시각화 도구

    PDF 파일 또는 arXiv URL에서 논문을 분석하여 Mermaid 다이어그램을 생성합니다.
    """
    console.print("[bold blue]Paper Visualizer[/bold blue] - 논문 방법론 자동 시각화 도구")
    console.print()

    try:
        # 비동기 실행
        asyncio.run(main(input, output, format, verbose))
    except Exception as e:
        console.print(f"[bold red]오류 발생:[/bold red] {e}")
        raise click.ClickException(str(e))


async def main(input_path: str, output_dir: str, format_type: str, verbose: bool) -> None:
    """
    메인 실행 로직
    """
    # 입력 파일 확인
    if input_path.startswith("http"):
        console.print("[yellow]arXiv URL 지원은 아직 구현되지 않았습니다.[/yellow]")
        console.print("PDF 파일 경로를 사용해주세요.")
        return

    pdf_path = Path(input_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF 파일을 찾을 수 없습니다: {input_path}")

    # 출력 디렉토리 생성
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:

        # 1. PDF 파싱
        task1 = progress.add_task("PDF에서 텍스트 추출 중...", total=None)
        try:
            paper_text = extract_paper_text(str(pdf_path))
            progress.update(task1, description="✅ PDF 텍스트 추출 완료")

            if verbose:
                console.print(f"  - 논문 제목: {paper_text.title}")
                console.print(f"  - 초록 길이: {len(paper_text.abstract)}자")
                console.print(f"  - 방법론 길이: {len(paper_text.method)}자")
        except Exception as e:
            progress.update(task1, description=f"❌ PDF 파싱 실패: {e}")
            raise

        # 2. LLM 분석
        task2 = progress.add_task("논문 방법론 분석 중...", total=None)
        try:
            structure = await analyze_paper(paper_text)
            progress.update(task2, description="✅ 논문 분석 완료")

            if verbose:
                console.print(f"  - 컴포넌트 수: {len(structure.components)}")
                console.print(f"  - 연결 관계 수: {len(structure.connections)}")
        except Exception as e:
            progress.update(task2, description=f"❌ 분석 실패: {e}")
            console.print("[yellow]분석에 실패했습니다. OPENAI_API_KEY 환경 변수를 확인해주세요.[/yellow]")
            raise

        # 3. 결과 생성
        task3 = progress.add_task("시각화 결과 생성 중...", total=None)
        try:
            if format_type in ["markdown", "both"]:
                # 마크다운 리포트 생성
                markdown_report = generate_markdown_report(structure)
                markdown_file = output_path / f"{pdf_path.stem}_report.md"
                markdown_file.write_text(markdown_report, encoding='utf-8')

                if not validate_mermaid_syntax(markdown_report):
                    console.print("[yellow]경고: 생성된 Mermaid 다이어그램에 문법 오류가 있을 수 있습니다.[/yellow]")

            if format_type in ["mermaid", "both"]:
                # Mermaid 다이어그램만 별도 저장
                from .visualizer import generate_mermaid_diagram
                mermaid_diagram = generate_mermaid_diagram(structure)
                mermaid_file = output_path / f"{pdf_path.stem}_diagram.mmd"
                mermaid_file.write_text(mermaid_diagram, encoding='utf-8')

            progress.update(task3, description="✅ 결과 생성 완료")

        except Exception as e:
            progress.update(task3, description=f"❌ 결과 생성 실패: {e}")
            raise

    # 결과 요약
    console.print()
    console.print("[bold green]✅ 작업 완료![/bold green]")
    console.print(f"출력 위치: {output_path.absolute()}")

    if format_type in ["markdown", "both"]:
        console.print(f"  - 마크다운 리포트: {pdf_path.stem}_report.md")
    if format_type in ["mermaid", "both"]:
        console.print(f"  - Mermaid 다이어그램: {pdf_path.stem}_diagram.mmd")

    console.print()
    console.print("[bold]생성된 다이어그램 미리보기:[/bold]")

    # 다이어그램 미리보기
    from .visualizer import generate_mermaid_diagram
    diagram = generate_mermaid_diagram(structure)
    console.print(diagram)


def main_direct(pdf_path: str) -> None:
    """
    직접 실행을 위한 간단한 진입점
    """
    asyncio.run(main(pdf_path, "output", "markdown", True))


if __name__ == "__main__":
    # 환경 변수 확인
    if not os.getenv("OPENAI_API_KEY"):
        console.print("[bold red]오류:[/bold red] OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
        console.print("API 키를 설정해주세요:")
        console.print("export OPENAI_API_KEY=your_api_key_here")
        exit(1)

    cli()