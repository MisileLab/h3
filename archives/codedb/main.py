import typer
from pathlib import Path
from src import core

app = typer.Typer()

@app.command()
def index(path: Path = typer.Argument(..., exists=True, help="지정된 경로의 코드/문서를 인덱싱합니다.")):
    """지정된 경로의 코드/문서를 인덱싱합니다."""
    typer.echo(f"'{path}'의 내용을 인덱싱합니다...")
    core.index_directory(path)

@app.command()
def search(
    query_text: str = typer.Argument(..., help="인덱싱된 데이터에서 검색을 수행합니다."),
    n_results: int = typer.Option(5, "--top-k", "-k", help="반환할 결과의 수")
):
    """인덱싱된 데이터에서 검색을 수행합니다."""
    typer.echo(f"질의: '{query_text}'")
    results = core.search_in_index(query_text, n_results=n_results)
    
    if not results or not results.get('documents') or not results['documents'][0]:
        typer.echo("검색 결과가 없습니다.")
        return

    typer.echo("\n--- 검색 결과 ---")
    for i, doc in enumerate(results['documents'][0]):
        distance = results['distances'][0][i]
        metadata = results['metadatas'][0][i]
        
        typer.echo(f"\n결과 {i+1}:")
        typer.echo(f"  - 소스: {metadata.get('source', 'N/A')}")
        typer.echo(f"  - 유사도 (거리): {distance:.4f}") # ChromaDB는 거리를 반환합니다. 낮을수록 유사합니다.
        typer.echo("  - 내용:")
        typer.echo(f"  ```\n{doc}\n  ```")
    typer.echo("\n--- 검색 완료 ---")


if __name__ == "__main__":
    app()