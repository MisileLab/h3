# Code Knowledge Graph

A prototype for building and querying code knowledge graphs using Python and Rust. This project demonstrates how to transform multi-language codebases into structured knowledge graphs that enable AI systems to understand code relationships and answer complex questions about code structure.

## 🎯 Project Goal

This prototype aims to solve the limitations of current LLM-based code generation tools by:
- Creating a structured representation of code relationships
- Enabling deep codebase understanding beyond context window limitations
- Providing a foundation for next-generation AI pair programmers

## 🛠️ Technical Stack

- **Languages**: Python 3.10+ and Rust
- **Package Manager**: `uv`
- **Code Parsing**: `tree-sitter` with `tree-sitter-python` and `tree-sitter-rust`
- **Graph Storage & Navigation**: `NetworkX`
- **LLM Integration**: `openai` (API placeholder)
- **MCP Server**: `mcp` for Cursor IDE integration

## 📦 Installation

### Prerequisites
- Python 3.10 or higher
- [uv](https://github.com/astral-sh/uv) package manager
- [Cursor IDE](https://cursor.sh/) (for MCP integration)

### Setup

1. Clone or navigate to the project directory:
```bash
cd code_knowledge_graph
```

2. Install dependencies using uv:
```bash
uv sync
```

Or install dependencies manually:
```bash
uv add tree-sitter tree-sitter-python tree-sitter-rust networkx openai
```

3. Activate the virtual environment (if needed):
```bash
source .venv/bin/activate  # On Unix/macOS
# or
.venv\Scripts\activate     # On Windows
```

## 🚀 Quick Start

Run the demo to see the complete workflow:

```bash
uv run python demo/demo.py
```

This will:
1. Parse the demo Python and Rust files in the `demo/` directory
2. Build a knowledge graph of the code structure
3. Answer sample questions about the codebase using the graph

## 🔌 Cursor IDE Integration (MCP Server)

This project includes an MCP (Model Context Protocol) server that allows you to use the code knowledge graph directly in Cursor IDE!

### Setup for Cursor

1. **Install dependencies**:
```bash
uv sync
```

2. **Run the setup script**:
```bash
uv run python setup_cursor.py
```

This will automatically configure the MCP server for Cursor IDE.

3. **Restart Cursor IDE** - The MCP tools will be available in the chat interface.

### Available MCP Tools in Cursor

- **`analyze_project`** - Analyze a multi-language project (Python and Rust) and build its knowledge graph
- **`ask_about_code`** - Ask natural language questions about your codebase
- **`get_project_structure`** - Get detailed project overview
- **`find_functions`** - Find functions by criteria (module, parameter, class)
- **`get_function_details`** - Get detailed information about a specific function
- **`trace_function_calls`** - Trace call chains for functions
- **`get_class_hierarchy`** - Get class inheritance relationships
- **`find_rust_items`** - Find Rust-specific items (structs, enums, traits, impls)
- **`get_trait_implementations`** - Get all implementations of a specific trait

### Example Usage in Cursor

1. **First, analyze your project**:
   ```
   Use the analyze_project tool with your project path
   ```

2. **Then ask questions**:
   - "What functions call the authenticate_user function?"
   - "Show me all methods of the User class"
   - "What functions are defined in the auth module?"
   - "Trace the call chain for process_payment function"
   - "Find all functions that take a user_id parameter"
   - "Show me all Rust structs in the project"
   - "What traits are implemented for the User struct?"
   - "Find all enum variants in the codebase"

### Manual MCP Configuration

If the setup script doesn't work, you can manually configure Cursor:

1. Create/edit `~/.cursor/mcp_servers.json`:
```json
{
  "mcpServers": {
    "code-knowledge-graph": {
      "command": "uv",
      "args": [
        "run",
        "python",
        "/path/to/your/code_knowledge_graph/mcp_server.py"
      ],
      "cwd": "/path/to/your/code_knowledge_graph",
      "env": {}
    }
  }
}
```

2. Replace the paths with your actual project path
3. Restart Cursor IDE

## 📁 Project Structure

```
code_knowledge_graph/
├── README.md              # This file
├── pyproject.toml         # Project configuration and dependencies
├── SCHEMA.md              # Graph schema definition
├── FUTURE_WORK.md         # Research directions and future improvements
├── code_parser.py         # Code parsing using tree-sitter
├── graph_builder.py       # Knowledge graph construction
├── query_translator.py    # Natural language to graph query translation
├── main_pipeline.py       # Complete RAG pipeline integration
├── mcp_server.py          # MCP server for Cursor IDE integration
├── setup_cursor.py        # Setup script for Cursor configuration
└── demo/                  # Demo files and examples
    ├── demo.py            # Demo script showing the workflow
    ├── auth.py            # Sample authentication module
    ├── payment.py         # Sample payment module
    ├── rust_demo.rs       # Sample Rust module
    └── auth_rust.rs       # Sample Rust authentication module
```

## 🔧 Core Components

### 1. Code Parser (`code_parser.py` and `rust_parser.py`)
Extracts structural information from Python and Rust files using tree-sitter:
- Function definitions
- Class/Struct definitions  
- Function call relationships
- Import/Use statements
- Rust-specific: Enums, Traits, Impl blocks

### 2. Graph Builder (`graph_builder.py`)
Constructs a NetworkX directed graph from parsed code:
- Nodes: Functions, Classes, Modules, Structs, Enums, Traits, Impl blocks
- Edges: CALLS, INHERITS, DEFINED_IN, HAS_METHOD relationships

### 3. Query Translator (`query_translator.py`)
Translates natural language questions to NetworkX graph queries using few-shot prompting.

### 4. Main Pipeline (`main_pipeline.py`)
Integrates all components into a complete RAG pipeline for answering code questions.

## 🎯 Example Usage

```python
from main_pipeline import ask_question_about_code

# Ask questions about a codebase
answer = ask_question_about_code(
    project_path="demo/",
    question="What functions call the authenticate_user function?"
)
print(answer)
```

## 🔮 Future Work

See [FUTURE_WORK.md](FUTURE_WORK.md) for research directions and advanced features planned for this prototype.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.