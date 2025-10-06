"""
Code Parser Module

This module uses tree-sitter to parse Python files and extract structural information
including functions, classes, and their relationships.
"""

import os
from typing import List, Dict, Any, Optional
from tree_sitter import Language, Parser, Node
import tree_sitter_python


def initialize_parser() -> Parser:
    """
    Initialize and return a tree-sitter parser for Python.
    
    Returns:
        Parser: Configured tree-sitter parser for Python
    """
    # Build the Python language library
    PY_LANGUAGE = Language(tree_sitter_python.language())
    
    # Create parser
    parser = Parser(PY_LANGUAGE)
    
    return parser


def extract_node_text(node: Node, source_code: bytes) -> str:
    """
    Extract text content from a tree-sitter node.
    
    Args:
        node: Tree-sitter node
        source_code: Source code as bytes
        
    Returns:
        str: Text content of the node
    """
    return source_code[node.start_byte:node.end_byte].decode('utf-8')


def extract_function_info(node: Node, source_code: bytes, file_path: str) -> Dict[str, Any]:
    """
    Extract information about a function definition.
    
    Args:
        node: Function definition node
        source_code: Source code as bytes
        file_path: Path to the source file
        
    Returns:
        Dict: Function information
    """
    function_info = {
        'type': 'function',
        'name': '',
        'file_path': file_path,
        'line_number': node.start_point[0] + 1,
        'parameters': [],
        'return_type': '',
        'docstring': '',
        'is_async': False,
        'calls': []
    }
    
    # Check if it's an async function
    if node.type == 'async_function_definition':
        function_info['is_async'] = True
    
    # Extract function name
    for child in node.children:
        if child.type == 'identifier':
            function_info['name'] = extract_node_text(child, source_code)
            break
    
    # Extract parameters
    for child in node.children:
        if child.type == 'parameters':
            for param_child in child.children:
                if param_child.type == 'identifier':
                    function_info['parameters'].append(extract_node_text(param_child, source_code))
                elif param_child.type == 'typed_parameter':
                    for typed_child in param_child.children:
                        if typed_child.type == 'identifier':
                            function_info['parameters'].append(extract_node_text(typed_child, source_code))
                            break
    
    # Extract return type
    for child in node.children:
        if child.type == 'type':
            function_info['return_type'] = extract_node_text(child, source_code)
            break
    
    # Extract docstring
    for child in node.children:
        if child.type == 'block':
            for grandchild in child.children:
                if grandchild.type == 'string':
                    function_info['docstring'] = extract_node_text(grandchild, source_code).strip('"\'')
                    break
    
    # Extract function calls within the function
    function_calls = extract_function_calls(node, source_code)
    function_info['calls'] = function_calls
    
    return function_info


def extract_class_info(node: Node, source_code: bytes, file_path: str) -> Dict[str, Any]:
    """
    Extract information about a class definition.
    
    Args:
        node: Class definition node
        source_code: Source code as bytes
        file_path: Path to the source file
        
    Returns:
        Dict: Class information
    """
    class_info = {
        'type': 'class',
        'name': '',
        'file_path': file_path,
        'line_number': node.start_point[0] + 1,
        'base_classes': [],
        'docstring': '',
        'methods': []
    }
    
    # Extract class name
    for child in node.children:
        if child.type == 'identifier':
            class_info['name'] = extract_node_text(child, source_code)
            break
    
    # Extract base classes
    for child in node.children:
        if child.type == 'argument_list':
            for arg_child in child.children:
                if arg_child.type == 'identifier':
                    class_info['base_classes'].append(extract_node_text(arg_child, source_code))
    
    # Extract docstring
    for child in node.children:
        if child.type == 'block':
            for grandchild in child.children:
                if grandchild.type == 'string':
                    class_info['docstring'] = extract_node_text(grandchild, source_code).strip('"\'')
                    break
    
    # Extract methods
    for child in node.children:
        if child.type == 'block':
            for grandchild in child.children:
                if grandchild.type in ['function_definition', 'async_function_definition']:
                    method_info = extract_function_info(grandchild, source_code, file_path)
                    class_info['methods'].append(method_info)
    
    return class_info


def extract_function_calls(node: Node, source_code: bytes) -> List[str]:
    """
    Extract all function calls within a given node.
    
    Args:
        node: Tree-sitter node to search within
        source_code: Source code as bytes
        
    Returns:
        List[str]: List of function names being called
    """
    calls = []
    
    def traverse(node: Node):
        if node.type == 'call':
            # Get the function being called
            for child in node.children:
                if child.type == 'identifier':
                    calls.append(extract_node_text(child, source_code))
                    break
                elif child.type == 'attribute':
                    # Handle method calls like obj.method()
                    attr_parts = []
                    for attr_child in child.children:
                        if attr_child.type == 'identifier':
                            attr_parts.append(extract_node_text(attr_child, source_code))
                    if attr_parts:
                        calls.append('.'.join(attr_parts))
                    break
        
        # Recursively traverse children
        for child in node.children:
            traverse(child)
    
    traverse(node)
    return calls


def extract_imports(node: Node, source_code: bytes) -> List[Dict[str, str]]:
    """
    Extract import statements from the AST.
    
    Args:
        node: Root node of the AST
        source_code: Source code as bytes
        
    Returns:
        List[Dict]: List of import information
    """
    imports = []
    
    def traverse(node: Node):
        if node.type == 'import_statement':
            for child in node.children:
                if child.type == 'dotted_name':
                    import_name = extract_node_text(child, source_code)
                    imports.append({
                        'type': 'import',
                        'module': import_name,
                        'alias': None
                    })
                elif child.type == 'aliased_import':
                    for aliased_child in child.children:
                        if aliased_child.type == 'dotted_name':
                            import_name = extract_node_text(aliased_child, source_code)
                        elif aliased_child.type == 'identifier':
                            alias = extract_node_text(aliased_child, source_code)
                            imports[-1]['alias'] = alias
        
        elif node.type == 'import_from_statement':
            module_name = ''
            for child in node.children:
                if child.type == 'dotted_name':
                    module_name = extract_node_text(child, source_code)
                elif child.type == 'import_name':
                    for import_child in child.children:
                        if import_child.type == 'dotted_name':
                            name = extract_node_text(import_child, source_code)
                            imports.append({
                                'type': 'from_import',
                                'module': module_name,
                                'name': name,
                                'alias': None
                            })
                        elif import_child.type == 'aliased_import':
                            for aliased_child in import_child.children:
                                if aliased_child.type == 'dotted_name':
                                    name = extract_node_text(aliased_child, source_code)
                                elif aliased_child.type == 'identifier':
                                    alias = extract_node_text(aliased_child, source_code)
                                    imports[-1]['alias'] = alias
        
        # Recursively traverse children
        for child in node.children:
            traverse(child)
    
    traverse(node)
    return imports


def parse_python_file(file_path: str) -> List[Dict[str, Any]]:
    """
    Parse a Python file and extract structural information.
    
    Args:
        file_path: Path to the Python file to parse
        
    Returns:
        List[Dict]: List containing information about functions, classes, and modules
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Read the source code
    with open(file_path, 'rb') as f:
        source_code = f.read()
    
    # Initialize parser
    parser = initialize_parser()
    
    # Parse the source code
    tree = parser.parse(source_code)
    root_node = tree.root_node
    
    results = []
    
    # Extract module information
    module_name = os.path.splitext(os.path.basename(file_path))[0]
    imports = extract_imports(root_node, source_code)
    
    module_info = {
        'type': 'module',
        'name': module_name,
        'file_path': file_path,
        'imports': [imp['module'] for imp in imports],
        'docstring': ''
    }
    
    # Extract module docstring
    for child in root_node.children:
        if child.type == 'expression_statement':
            for grandchild in child.children:
                if grandchild.type == 'string':
                    module_info['docstring'] = extract_node_text(grandchild, source_code).strip('"\'')
                    break
    
    results.append(module_info)
    
    # Extract functions and classes
    def traverse(node: Node):
        if node.type in ['function_definition', 'async_function_definition']:
            func_info = extract_function_info(node, source_code, file_path)
            results.append(func_info)
        elif node.type == 'class_definition':
            class_info = extract_class_info(node, source_code, file_path)
            results.append(class_info)
        
        # Recursively traverse children
        for child in node.children:
            traverse(child)
    
    traverse(root_node)
    
    return results


if __name__ == "__main__":
    # Test the parser with a simple example
    test_file = "demo/auth.py"
    if os.path.exists(test_file):
        try:
            parsing_results = parse_python_file(test_file)
            print(f"Parsed {len(parsing_results)} elements from {test_file}")
            for result in parsing_results:
                print(f"  {result['type']}: {result['name']}")
        except Exception as e:
            print(f"Error parsing {test_file}: {e}")
    else:
        print(f"Test file {test_file} not found")