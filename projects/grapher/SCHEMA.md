# Code Knowledge Graph Schema

This document defines the schema for the code knowledge graph used in this project. The graph represents code elements as nodes and their relationships as edges, enabling structured querying and analysis of codebases.

## Graph Overview

The knowledge graph is a **directed graph** where:
- **Nodes** represent code elements (functions, classes, modules)
- **Edges** represent relationships between code elements
- **Properties** provide additional metadata about nodes and edges

## Node Types

### 1. Function
Represents a function or method in the codebase.

| Property | Type | Description | Example |
|----------|------|-------------|---------|
| `type` | string | Always "function" | `"function"` |
| `name` | string | Function name | `"authenticate_user"` |
| `file_path` | string | Path to the containing file | `"auth.py"` |
| `line_number` | int | Starting line number | `15` |
| `parameters` | list | List of parameter names | `["username", "password"]` |
| `return_type` | string | Return type annotation | `"bool"` |
| `docstring` | string | Function docstring | `"Authenticates a user"` |
| `is_async` | bool | Whether function is async | `false` |

### 2. Class
Represents a class definition in the codebase.

| Property | Type | Description | Example |
|----------|------|-------------|---------|
| `type` | string | Always "class" | `"class"` |
| `name` | string | Class name | `"User"` |
| `file_path` | string | Path to the containing file | `"auth.py"` |
| `line_number` | int | Starting line number | `5` |
| `base_classes` | list | List of parent classes | `["BaseModel"]` |
| `docstring` | string | Class docstring | `"User model"` |

### 3. Module
Represents a Python module/file.

| Property | Type | Description | Example |
|----------|------|-------------|---------|
| `type` | string | Always "module" | `"module"` |
| `name` | string | Module name (filename without extension) | `"auth"` |
| `file_path` | string | Full path to the file | `"auth.py"` |
| `imports` | list | List of imported modules | `["typing", "datetime"]` |
| `docstring` | string | Module docstring | `"Authentication utilities"` |

## Edge Types

### 1. CALLS
Represents a function calling another function.

| Property | Type | Description | Example |
|----------|------|-------------|---------|
| `type` | string | Always "CALLS" | `"CALLS"` |
| `line_number` | int | Line where call occurs | `25` |
| `call_type` | string | Type of call ("direct", "method", "async") | `"direct"` |

**Direction**: `caller_function` → `callee_function`

### 2. INHERITS
Represents class inheritance relationships.

| Property | Type | Description | Example |
|----------|------|-------------|---------|
| `type` | string | Always "INHERITS" | `"INHERITS"` |
| `inheritance_type` | string | Type of inheritance ("single", "multiple") | `"single"` |

**Direction**: `child_class` → `parent_class`

### 3. DEFINED_IN
Represents containment relationships (functions/classes defined in modules).

| Property | Type | Description | Example |
|----------|------|-------------|---------|
| `type` | string | Always "DEFINED_IN" | `"DEFINED_IN"` |
| `definition_type` | string | Type of definition ("function", "class") | `"function"` |

**Direction**: `defined_element` → `containing_module`

### 4. HAS_METHOD
Represents class-method relationships.

| Property | Type | Description | Example |
|----------|------|-------------|---------|
| `type` | string | Always "HAS_METHOD" | `"HAS_METHOD"` |
| `method_type` | string | Method type ("instance", "class", "static") | `"instance"` |
| `visibility` | string | Method visibility ("public", "private", "protected") | `"public"` |

**Direction**: `class` → `method`

### 5. IMPORTS
Represents module import relationships.

| Property | Type | Description | Example |
|----------|------|-------------|---------|
| `type` | string | Always "IMPORTS" | `"IMPORTS"` |
| `import_type` | string | Import type ("direct", "from_import", "alias") | `"from_import"` |
| `alias` | string | Import alias if any | `"auth"` |

**Direction**: `importing_module` → `imported_module`

## Graph Query Patterns

### Common Queries

1. **Find all callers of a function**:
   ```python
   list(G.predecessors('function_name'))
   ```

2. **Find all methods of a class**:
   ```python
   [n for n in G.successors('class_name') 
    if G.nodes[n]['type'] == 'function']
   ```

3. **Find all functions in a module**:
   ```python
   [n for n in G.successors('module_name') 
    if G.nodes[n]['type'] == 'function']
   ```

4. **Find inheritance hierarchy**:
   ```python
   list(nx.descendants(G, 'class_name'))
   ```

5. **Find call chain between functions**:
   ```python
   list(nx.all_simple_paths(G, 'caller', 'callee'))
   ```

## Schema Evolution

This schema is designed to be extensible. Future additions may include:

- **Variable nodes**: Representing global variables and constants
- **Parameter edges**: Representing parameter passing relationships
- **Exception edges**: Representing exception handling relationships
- **Test edges**: Representing test coverage relationships
- **Documentation edges**: Linking code to documentation

## Implementation Notes

- All node names are **unique** within the graph using the format: `module_name.element_name`
- The graph is **directed** to preserve relationship semantics
- Properties are stored as **key-value pairs** in NetworkX node/edge attributes
- The schema supports **multi-graph** features for parallel relationships