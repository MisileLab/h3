# CodeDB

CodeDB is a command-line tool that allows you to index and search your local codebase using natural language queries. It leverages vector embeddings to find relevant code snippets, even if they don't contain the exact keywords from your search.

## Features

*   **Code Indexing:** Recursively indexes a directory of code files.
*   **Natural Language Search:** Search your code using natural language questions or descriptions.
*   **Semantic Search:** Finds code based on meaning and context, not just keywords.
*   **Supports Multiple File Types:** Indexes a wide range of common programming language files.
*   **Easy to Use:** Simple command-line interface.

## Requirements

*   Python 3.13+
*   The dependencies listed in `pyproject.toml`. The main ones are:
    *   `typer`
    *   `chromadb`
    *   `langchain`
    *   `sentence-transformers`

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd codedb
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    pip install -e .
    ```

## Usage

### 1. Indexing your Code

Before you can search, you need to index a directory containing your code.

```bash
codedb index /path/to/your/code
```

This command will traverse the directory, find all supported files, and store their vector embeddings in a local ChromaDB database (in the `.chroma_db` directory).

### 2. Searching your Code

Once your code is indexed, you can search it using the `search` command.

```bash
codedb search "your natural language query"
```

**Example:**

```bash
codedb search "function to read a file in python"
```

You can also specify the number of results to return using the `-k` or `--top-k` option:

```bash
codedb search "how to make a GET request in javascript" -k 10
```

## Supported File Types

CodeDB supports indexing the following file extensions:

- `.py`
- `.md`
- `.txt`
- `.js`
- `.ts`
- `.html`
- `.css`
- `.java`
- `.c`
- `.cpp`
- `.go`
- `.rs`
