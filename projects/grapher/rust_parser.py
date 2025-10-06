"""
Rust Parser Module

This module uses tree-sitter to parse Rust files and extract structural information
including functions, structs, enums, traits, and their relationships.
"""

import os
from typing import List, Dict, Any, Optional
from tree_sitter import Language, Parser, Node
import tree_sitter_rust


def initialize_rust_parser() -> Parser:
    """
    Initialize and return a tree-sitter parser for Rust.
    
    Returns:
        Parser: Configured tree-sitter parser for Rust
    """
    # Build the Rust language library
    RUST_LANGUAGE = Language(tree_sitter_rust.language())
    
    # Create parser
    parser = Parser(RUST_LANGUAGE)
    
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
    Extract information about a Rust function definition.
    
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
        'is_unsafe': False,
        'visibility': 'private',
        'calls': []
    }
    
    # Check if it's an async function
    if node.type == 'async_function':
        function_info['is_async'] = True
    
    # Check if it's an unsafe function
    for child in node.children:
        if child.type == 'async':
            function_info['is_async'] = True
        elif child.type == 'unsafe':
            function_info['is_unsafe'] = True
        elif child.type == 'visibility_modifier':
            function_info['visibility'] = extract_node_text(child, source_code)
        elif child.type == 'identifier':
            function_info['name'] = extract_node_text(child, source_code)
        elif child.type == 'parameters':
            # Extract parameters
            for param_child in child.children:
                if param_child.type == 'parameter':
                    param_name = ''
                    param_type = ''
                    for param_part in param_child.children:
                        if param_part.type == 'identifier':
                            param_name = extract_node_text(param_part, source_code)
                        elif param_part.type == 'type_identifier':
                            param_type = extract_node_text(param_part, source_code)
                    if param_name:
                        function_info['parameters'].append(f"{param_name}: {param_type}" if param_type else param_name)
        elif child.type == 'type_identifier':
            function_info['return_type'] = extract_node_text(child, source_code)
    
    # Extract docstring (Rust uses /// comments)
    for child in node.children:
        if child.type == 'line_comment' and extract_node_text(child, source_code).strip().startswith('///'):
            function_info['docstring'] = extract_node_text(child, source_code).strip('///').strip()
            break
    
    # Extract function calls within the function
    function_calls = extract_function_calls(node, source_code)
    function_info['calls'] = function_calls
    
    return function_info


def extract_struct_info(node: Node, source_code: bytes, file_path: str) -> Dict[str, Any]:
    """
    Extract information about a Rust struct definition.
    
    Args:
        node: Struct definition node
        source_code: Source code as bytes
        file_path: Path to the source file
        
    Returns:
        Dict: Struct information
    """
    struct_info = {
        'type': 'struct',
        'name': '',
        'file_path': file_path,
        'line_number': node.start_point[0] + 1,
        'fields': [],
        'docstring': '',
        'visibility': 'private',
        'is_unit_struct': False,
        'is_tuple_struct': False
    }
    
    # Extract struct name and type
    for child in node.children:
        if child.type == 'visibility_modifier':
            struct_info['visibility'] = extract_node_text(child, source_code)
        elif child.type == 'type_identifier':
            struct_info['name'] = extract_node_text(child, source_code)
        elif child.type == 'line_comment' and extract_node_text(child, source_code).strip().startswith('///'):
            struct_info['docstring'] = extract_node_text(child, source_code).strip('///').strip()
        elif child.type == 'field_declaration_list':
            # Regular struct with named fields
            for field_child in child.children:
                if field_child.type == 'field_declaration':
                    field_name = ''
                    field_type = ''
                    for field_part in field_child.children:
                        if field_part.type == 'field_identifier':
                            field_name = extract_node_text(field_part, source_code)
                        elif field_part.type == 'type_identifier':
                            field_type = extract_node_text(field_part, source_code)
                    if field_name:
                        struct_info['fields'].append(f"{field_name}: {field_type}" if field_type else field_name)
        elif child.type == 'tuple_type':
            # Tuple struct
            struct_info['is_tuple_struct'] = True
            for tuple_child in child.children:
                if tuple_child.type == 'type_identifier':
                    struct_info['fields'].append(extract_node_text(tuple_child, source_code))
        elif child.type == 'semicolon':
            # Unit struct
            struct_info['is_unit_struct'] = True
    
    return struct_info


def extract_enum_info(node: Node, source_code: bytes, file_path: str) -> Dict[str, Any]:
    """
    Extract information about a Rust enum definition.
    
    Args:
        node: Enum definition node
        source_code: Source code as bytes
        file_path: Path to the source file
        
    Returns:
        Dict: Enum information
    """
    enum_info = {
        'type': 'enum',
        'name': '',
        'file_path': file_path,
        'line_number': node.start_point[0] + 1,
        'variants': [],
        'docstring': '',
        'visibility': 'private'
    }
    
    # Extract enum name and variants
    for child in node.children:
        if child.type == 'visibility_modifier':
            enum_info['visibility'] = extract_node_text(child, source_code)
        elif child.type == 'type_identifier':
            enum_info['name'] = extract_node_text(child, source_code)
        elif child.type == 'line_comment' and extract_node_text(child, source_code).strip().startswith('///'):
            enum_info['docstring'] = extract_node_text(child, source_code).strip('///').strip()
        elif child.type == 'enum_variant':
            variant_name = ''
            variant_data = ''
            for variant_child in child.children:
                if variant_child.type == 'identifier':
                    variant_name = extract_node_text(variant_child, source_code)
                elif variant_child.type in ['tuple_type', 'struct']:
                    variant_data = extract_node_text(variant_child, source_code)
            if variant_name:
                enum_info['variants'].append(f"{variant_name}{variant_data}" if variant_data else variant_name)
    
    return enum_info


def extract_trait_info(node: Node, source_code: bytes, file_path: str) -> Dict[str, Any]:
    """
    Extract information about a Rust trait definition.
    
    Args:
        node: Trait definition node
        source_code: Source code as bytes
        file_path: Path to the source file
        
    Returns:
        Dict: Trait information
    """
    trait_info = {
        'type': 'trait',
        'name': '',
        'file_path': file_path,
        'line_number': node.start_point[0] + 1,
        'methods': [],
        'docstring': '',
        'visibility': 'private',
        'supertraits': []
    }
    
    # Extract trait name and methods
    for child in node.children:
        if child.type == 'visibility_modifier':
            trait_info['visibility'] = extract_node_text(child, source_code)
        elif child.type == 'type_identifier':
            trait_info['name'] = extract_node_text(child, source_code)
        elif child.type == 'line_comment' and extract_node_text(child, source_code).strip().startswith('///'):
            trait_info['docstring'] = extract_node_text(child, source_code).strip('///').strip()
        elif child.type == 'trait_bounds':
            # Extract supertraits
            for bound_child in child.children:
                if bound_child.type == 'type_identifier':
                    trait_info['supertraits'].append(extract_node_text(bound_child, source_code))
        elif child.type in ['function_item', 'async_function']:
            method_info = extract_function_info(child, source_code, file_path)
            trait_info['methods'].append(method_info)
    
    return trait_info


def extract_impl_info(node: Node, source_code: bytes, file_path: str) -> Dict[str, Any]:
    """
    Extract information about a Rust impl block.
    
    Args:
        node: Impl block node
        source_code: Source code as bytes
        file_path: Path to the source file
        
    Returns:
        Dict: Impl block information
    """
    impl_info = {
        'type': 'impl',
        'trait_name': '',
        'type_name': '',
        'file_path': file_path,
        'line_number': node.start_point[0] + 1,
        'methods': [],
        'is_trait_impl': False
    }
    
    # Extract trait and type information
    for child in node.children:
        if child.type == 'trait_bounds':
            # This is a trait implementation
            impl_info['is_trait_impl'] = True
            for bound_child in child.children:
                if bound_child.type == 'type_identifier':
                    impl_info['trait_name'] = extract_node_text(bound_child, source_code)
        elif child.type == 'type_identifier':
            impl_info['type_name'] = extract_node_text(child, source_code)
        elif child.type in ['function_item', 'async_function']:
            method_info = extract_function_info(child, source_code, file_path)
            impl_info['methods'].append(method_info)
    
    return impl_info


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
        if node.type == 'call_expression':
            # Get the function being called
            for child in node.children:
                if child.type == 'identifier':
                    calls.append(extract_node_text(child, source_code))
                    break
                elif child.type == 'field_expression':
                    # Handle method calls like obj.method()
                    field_parts = []
                    for field_child in child.children:
                        if field_child.type == 'identifier':
                            field_parts.append(extract_node_text(field_child, source_code))
                    if field_parts:
                        calls.append('.'.join(field_parts))
                    break
        
        # Recursively traverse children
        for child in node.children:
            traverse(child)
    
    traverse(node)
    return calls


def extract_use_statements(node: Node, source_code: bytes) -> List[Dict[str, str]]:
    """
    Extract use statements from the AST.
    
    Args:
        node: Root node of the AST
        source_code: Source code as bytes
        
    Returns:
        List[Dict]: List of use statement information
    """
    uses = []
    
    def traverse(node: Node):
        if node.type == 'use_declaration':
            for child in node.children:
                if child.type == 'scoped_use_list':
                    # use std::collections::{HashMap, HashSet};
                    for list_child in child.children:
                        if list_child.type == 'identifier':
                            uses.append({
                                'type': 'scoped_use',
                                'module': 'std::collections',
                                'name': extract_node_text(list_child, source_code),
                                'alias': None
                            })
                elif child.type == 'use_list':
                    # use std::io::{self, Write};
                    for list_child in child.children:
                        if list_child.type == 'identifier':
                            uses.append({
                                'type': 'use_list',
                                'module': 'std::io',
                                'name': extract_node_text(list_child, source_code),
                                'alias': None
                            })
                elif child.type == 'scoped_identifier':
                    # use std::collections::HashMap;
                    parts = []
                    for part_child in child.children:
                        if part_child.type == 'identifier':
                            parts.append(extract_node_text(part_child, source_code))
                    if parts:
                        module = '::'.join(parts[:-1]) if len(parts) > 1 else ''
                        name = parts[-1]
                        uses.append({
                            'type': 'scoped_use',
                            'module': module,
                            'name': name,
                            'alias': None
                        })
        
        # Recursively traverse children
        for child in node.children:
            traverse(child)
    
    traverse(node)
    return uses


def parse_rust_file(file_path: str) -> List[Dict[str, Any]]:
    """
    Parse a Rust file and extract structural information.
    
    Args:
        file_path: Path to the Rust file to parse
        
    Returns:
        List[Dict]: List containing information about functions, structs, enums, traits, and modules
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Read the source code
    with open(file_path, 'rb') as f:
        source_code = f.read()
    
    # Initialize parser
    parser = initialize_rust_parser()
    
    # Parse the source code
    tree = parser.parse(source_code)
    root_node = tree.root_node
    
    results = []
    
    # Extract module information
    module_name = os.path.splitext(os.path.basename(file_path))[0]
    uses = extract_use_statements(root_node, source_code)
    
    module_info = {
        'type': 'module',
        'name': module_name,
        'file_path': file_path,
        'imports': [use['module'] for use in uses if use['module']],
        'docstring': ''
    }
    
    # Extract module docstring
    for child in root_node.children:
        if child.type == 'line_comment' and extract_node_text(child, source_code).strip().startswith('//!'):
            module_info['docstring'] = extract_node_text(child, source_code).strip('//!').strip()
            break
    
    results.append(module_info)
    
    # Extract functions, structs, enums, traits, and impl blocks
    def traverse(node: Node):
        if node.type in ['function_item', 'async_function']:
            func_info = extract_function_info(node, source_code, file_path)
            results.append(func_info)
        elif node.type == 'struct_item':
            struct_info = extract_struct_info(node, source_code, file_path)
            results.append(struct_info)
        elif node.type == 'enum_item':
            enum_info = extract_enum_info(node, source_code, file_path)
            results.append(enum_info)
        elif node.type == 'trait_item':
            trait_info = extract_trait_info(node, source_code, file_path)
            results.append(trait_info)
        elif node.type == 'impl_item':
            impl_info = extract_impl_info(node, source_code, file_path)
            results.append(impl_info)
        
        # Recursively traverse children
        for child in node.children:
            traverse(child)
    
    traverse(root_node)
    
    return results


if __name__ == "__main__":
    # Test the parser with a simple example
    test_file = "demo/rust_demo.rs"
    if os.path.exists(test_file):
        try:
            parsing_results = parse_rust_file(test_file)
            print(f"Parsed {len(parsing_results)} elements from {test_file}")
            for result in parsing_results:
                print(f"  {result['type']}: {result['name']}")
        except Exception as e:
            print(f"Error parsing {test_file}: {e}")
    else:
        print(f"Test file {test_file} not found")
