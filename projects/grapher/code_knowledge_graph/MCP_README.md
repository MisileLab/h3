# Code Knowledge Graph MCP Server

This document provides detailed information about the MCP (Model Context Protocol) server implementation for the Code Knowledge Graph project.

## 🎯 What is the MCP Server?

The MCP server allows Cursor IDE (and other MCP-compatible tools) to directly interact with the code knowledge graph functionality. Instead of running the demo script, you can now use the knowledge graph capabilities directly within your IDE.

## 🚀 Quick Setup

### Automatic Setup (Recommended)

```bash
uv run python setup_cursor.py
```

This script will:
1. Detect your Cursor configuration directory
2. Create/update the MCP server configuration
3. Provide detailed setup instructions

### Manual Setup

1. Create/edit `~/.cursor/mcp_servers.json`
2. Add the following configuration:

```json
{
  "mcpServers": {
    "code-knowledge-graph": {
      "command": "uv",
      "args": [
        "run",
        "python",
        "/absolute/path/to/code_knowledge_graph/mcp_server.py"
      ],
      "cwd": "/absolute/path/to/code_knowledge_graph",
      "env": {}
    }
  }
}
```

3. Restart Cursor IDE

## 🛠️ Available Tools

### 1. analyze_project
Analyzes a Python project and builds its knowledge graph.

**Parameters:**
- `project_path` (string, required): Path to the Python project directory

**Example:**
```
Analyze the project at /path/to/my/python/project
```

### 2. ask_about_code
Asks natural language questions about the analyzed codebase.

**Parameters:**
- `question` (string, required): Natural language question about the code

**Example Questions:**
- "What functions call the authenticate_user function?"
- "Show me all methods of the User class"
- "What functions are defined in the auth module?"
- "Find all functions that take a user_id parameter"

### 3. get_project_structure
Gets detailed information about the current project structure.

**Parameters:** None

**Returns:**
- Project overview with node/edge counts
- Graph analysis with most connected nodes
- Node and edge type distributions

### 4. find_functions
Finds functions matching specific criteria.

**Parameters:**
- `module` (string, optional): Filter by module name
- `parameter` (string, optional): Filter by parameter name
- `class` (string, optional): Filter by class name

**Examples:**
- Find all functions in the auth module
- Find functions that take a user_id parameter
- Find methods of the User class

### 5. get_function_details
Gets detailed information about a specific function.

**Parameters:**
- `function_name` (string, required): Name of the function
- `module` (string, optional): Module name for disambiguation

**Returns:**
- Function signature and metadata
- Call relationships (what it calls, what calls it)
- Class/module membership

### 6. trace_function_calls
Traces the call chain for a specific function.

**Parameters:**
- `function_name` (string, required): Name of the function
- `direction` (string, optional): "both", "incoming", or "outgoing" (default: "both")
- `max_depth` (integer, optional): Maximum depth of tracing (default: 3)

**Examples:**
- Trace all calls made by process_payment
- Find all functions that call authenticate_user
- Get bidirectional call chain for User.__init__

### 7. get_class_hierarchy
Gets the inheritance hierarchy for a class.

**Parameters:**
- `class_name` (string, required): Name of the class
- `direction` (string, optional): "both", "parents", or "children" (default: "both")

**Returns:**
- Parent classes (inheritance)
- Child classes (classes that inherit from this)
- Methods and their visibility

## 💡 Usage Workflow in Cursor

### Step 1: Analyze Your Project
First, you need to analyze your project to build the knowledge graph:

```
Use the analyze_project tool with the path to your Python project
```

### Step 2: Ask Questions
Once the project is analyzed, you can ask various questions:

**Code Exploration:**
- "What functions are defined in the auth module?"
- "Show me all classes in the payment module"
- "Find all functions that handle user authentication"

**Relationship Analysis:**
- "What functions call the authenticate_user function?"
- "Show me the call chain for process_payment"
- "Which classes inherit from BaseModel?"

**Detailed Information:**
- "Get details about the User class"
- "Show me the function signature of hash_password"
- "What parameters does the validate_session function take?"

## 🔍 Advanced Features

### Cross-Module Analysis
The MCP server can analyze relationships across different modules:

```
"How does the payment module depend on the auth module?"
"What functions from auth are used in payment processing?"
```

### Call Chain Tracing
Trace complex call relationships:

```
"Trace the complete call chain from login to payment processing"
"Find all functions involved in user registration flow"
```

### Class Hierarchy Analysis
Understand inheritance relationships:

```
"Show me the complete inheritance hierarchy for PaymentProvider"
"What classes implement the process_payment method?"
```

## 🛠️ Troubleshooting

### Common Issues

1. **Tools don't appear in Cursor**
   - Restart Cursor IDE
   - Check that the MCP configuration is correct
   - Verify the paths in the configuration

2. **"uv command not found"**
   - Install uv: `curl -LsSf https://astral.sh/uv/install.sh | sh`
   - Make sure uv is in your PATH

3. **"Project not found" errors**
   - Use absolute paths when analyzing projects
   - Check that the project directory exists and contains Python files

4. **Permission errors**
   - Make sure the MCP server script has execute permissions
   - Check that the project directory is readable

### Debug Mode

To enable debug logging, set the environment variable:

```bash
export CODE_KG_DEBUG=1
```

Then restart Cursor. The MCP server will output detailed logs to the console.

## 📊 Performance Considerations

- **Large Projects**: For projects with many files, the initial analysis may take a few seconds
- **Memory Usage**: The knowledge graph is kept in memory for fast queries
- **Caching**: The graph is cached after the first analysis for subsequent queries

## 🔧 Configuration Options

The MCP server can be configured through environment variables:

- `CODE_KG_MAX_DEPTH`: Maximum depth for call tracing (default: 3)
- `CODE_KG_CACHE_SIZE`: Maximum number of cached graphs (default: 5)
- `CODE_KG_LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)

## 🚀 Future Enhancements

Planned improvements to the MCP server:

1. **Real-time Updates**: Automatically update the graph when files change
2. **Multi-language Support**: Support for JavaScript, TypeScript, and other languages
3. **Advanced Queries**: More sophisticated natural language understanding
4. **Visualization**: Generate visual representations of code relationships
5. **Performance Optimization**: Incremental parsing and smarter caching

## 📚 Additional Resources

- [MCP Protocol Specification](https://modelcontextprotocol.io/)
- [Cursor IDE Documentation](https://cursor.sh/docs)
- [Project README](./README.md)
- [Graph Schema](./SCHEMA.md)
- [Future Work](./FUTURE_WORK.md)