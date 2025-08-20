import chromadb
from chromadb.utils import embedding_functions
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pathlib import Path
import typer
import subprocess

# 1. 상수 정의
CHROMA_DB_PATH = "./.chroma_db"
COLLECTION_NAME = "codedb_collection"
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"

# 2. ChromaDB 클라이언트 및 임베딩 함수 초기화
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)

# 3. 텍스트 분할기 초기화
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)

def get_or_create_collection():
    """컬렉션을 가져오거나 새로 생성합니다."""
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=sentence_transformer_ef
    )
    return collection

def is_text_file(file_path: Path):
    """파일이 텍스트 파일인지 확인합니다."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            f.read(1024)  # Read a small portion to check for text content
        return True
    except (UnicodeDecodeError, IOError):
        return False

def get_files_from_git(path: Path):
    """git ls-files를 사용하여 git이 추적하는 파일 목록을 가져옵니다."""
    try:
        result = subprocess.run(
            ['git', 'ls-files', '--cached', '--others', '--exclude-standard', str(path)],
            capture_output=True,
            text=True,
            check=True
        )
        return [Path(line) for line in result.stdout.strip().split('\n')]
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Git이 설치되지 않았거나, git 저장소가 아닌 경우 rglob으로 대체
        return list(path.rglob('*'))


def index_directory(path: Path):
    """지정된 경로의 모든 텍스트 파일을 순회하고 인덱싱합니다."""
    collection = get_or_create_collection()

    files_to_index = [p for p in get_files_from_git(path) if p.is_file() and is_text_file(p)]
    
    if not files_to_index:
        typer.echo("인덱싱할 파일을 찾을 수 없습니다.")
        return

    with typer.progressbar(files_to_index, label="파일 인덱싱 중") as progress:
        for file_path in progress:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                
                chunks = text_splitter.split_text(content)
                
                if not chunks:
                    continue

                # 각 청크에 대한 메타데이터 및 ID 생성
                metadatas = [{"source": str(file_path)} for _ in chunks]
                ids = [f"{file_path}:{i}" for i, _ in enumerate(chunks)]

                collection.add(
                    documents=chunks,
                    metadatas=metadatas,
                    ids=ids
                )
            except Exception as e:
                typer.echo(f"\n'{file_path}' 처리 중 오류 발생: {e}", err=True)

    typer.echo(f"\n총 {len(files_to_index)}개의 파일에서 문서들을 성공적으로 인덱싱했습니다.")


def search_in_index(query_text: str, n_results: int = 5):
    """인덱스에서 질의에 대한 검색을 수행합니다."""
    collection = get_or_create_collection()
    results = collection.query(
        query_texts=[query_text],
        n_results=n_results,
    )
    return results

