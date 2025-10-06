"""
Graph Builder Module

This module builds a NetworkX knowledge graph from parsed code information.
It creates nodes for functions, classes, and modules, and edges for their relationships.
"""

import networkx as nx
from typing import List, Dict, Any, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_node_id(element_type: str, name: str, module_name: str = "") -> str:
    """
    Create a unique node ID for a code element.
    
    Args:
        element_type: Type of element (function, class, module)
        name: Name of the element
        module_name: Name of the containing module (for functions and classes)
        
    Returns:
        str: Unique node ID
    """
    if element_type == "module":
        return f"module.{name}"
    else:
        return f"{module_name}.{name}"


def add_module_node(graph: nx.DiGraph, module_info: Dict[str, Any]) -> str:
    """
    Add a module node to the graph.
    
    Args:
        graph: NetworkX directed graph
        module_info: Module information dictionary
        
    Returns:
        str: Node ID of the added module
    """
    node_id = create_node_id("module", module_info["name"])
    
    # Add node with module properties
    graph.add_node(node_id, **{
        'type': 'module',
        'name': module_info['name'],
        'file_path': module_info['file_path'],
        'imports': module_info.get('imports', []),
        'docstring': module_info.get('docstring', '')
    })
    
    logger.info(f"Added module node: {node_id}")
    return node_id


def add_function_node(graph: nx.DiGraph, func_info: Dict[str, Any], module_name: str) -> str:
    """
    Add a function node to the graph.
    
    Args:
        graph: NetworkX directed graph
        func_info: Function information dictionary
        module_name: Name of the containing module
        
    Returns:
        str: Node ID of the added function
    """
    node_id = create_node_id("function", func_info["name"], module_name)
    
    # Add node with function properties
    graph.add_node(node_id, **{
        'type': 'function',
        'name': func_info['name'],
        'file_path': func_info['file_path'],
        'line_number': func_info['line_number'],
        'parameters': func_info.get('parameters', []),
        'return_type': func_info.get('return_type', ''),
        'docstring': func_info.get('docstring', ''),
        'is_async': func_info.get('is_async', False)
    })
    
    logger.info(f"Added function node: {node_id}")
    return node_id


def add_class_node(graph: nx.DiGraph, class_info: Dict[str, Any], module_name: str) -> str:
    """
    Add a class node to the graph.
    
    Args:
        graph: NetworkX directed graph
        class_info: Class information dictionary
        module_name: Name of the containing module
        
    Returns:
        str: Node ID of the added class
    """
    node_id = create_node_id("class", class_info["name"], module_name)
    
    # Add node with class properties
    graph.add_node(node_id, **{
        'type': 'class',
        'name': class_info['name'],
        'file_path': class_info['file_path'],
        'line_number': class_info['line_number'],
        'base_classes': class_info.get('base_classes', []),
        'docstring': class_info.get('docstring', '')
    })
    
    logger.info(f"Added class node: {node_id}")
    return node_id


def add_struct_node(graph: nx.DiGraph, struct_info: Dict[str, Any], module_name: str) -> str:
    """
    Add a Rust struct node to the graph.
    
    Args:
        graph: NetworkX directed graph
        struct_info: Struct information dictionary
        module_name: Name of the containing module
        
    Returns:
        str: Node ID of the added struct
    """
    node_id = create_node_id("struct", struct_info["name"], module_name)
    
    # Add node with struct properties
    graph.add_node(node_id, **{
        'type': 'struct',
        'name': struct_info['name'],
        'file_path': struct_info['file_path'],
        'line_number': struct_info['line_number'],
        'fields': struct_info.get('fields', []),
        'docstring': struct_info.get('docstring', ''),
        'visibility': struct_info.get('visibility', 'private'),
        'is_unit_struct': struct_info.get('is_unit_struct', False),
        'is_tuple_struct': struct_info.get('is_tuple_struct', False)
    })
    
    logger.info(f"Added struct node: {node_id}")
    return node_id


def add_enum_node(graph: nx.DiGraph, enum_info: Dict[str, Any], module_name: str) -> str:
    """
    Add a Rust enum node to the graph.
    
    Args:
        graph: NetworkX directed graph
        enum_info: Enum information dictionary
        module_name: Name of the containing module
        
    Returns:
        str: Node ID of the added enum
    """
    node_id = create_node_id("enum", enum_info["name"], module_name)
    
    # Add node with enum properties
    graph.add_node(node_id, **{
        'type': 'enum',
        'name': enum_info['name'],
        'file_path': enum_info['file_path'],
        'line_number': enum_info['line_number'],
        'variants': enum_info.get('variants', []),
        'docstring': enum_info.get('docstring', ''),
        'visibility': enum_info.get('visibility', 'private')
    })
    
    logger.info(f"Added enum node: {node_id}")
    return node_id


def add_trait_node(graph: nx.DiGraph, trait_info: Dict[str, Any], module_name: str) -> str:
    """
    Add a Rust trait node to the graph.
    
    Args:
        graph: NetworkX directed graph
        trait_info: Trait information dictionary
        module_name: Name of the containing module
        
    Returns:
        str: Node ID of the added trait
    """
    node_id = create_node_id("trait", trait_info["name"], module_name)
    
    # Add node with trait properties
    graph.add_node(node_id, **{
        'type': 'trait',
        'name': trait_info['name'],
        'file_path': trait_info['file_path'],
        'line_number': trait_info['line_number'],
        'methods': trait_info.get('methods', []),
        'docstring': trait_info.get('docstring', ''),
        'visibility': trait_info.get('visibility', 'private'),
        'supertraits': trait_info.get('supertraits', [])
    })
    
    logger.info(f"Added trait node: {node_id}")
    return node_id


def add_impl_node(graph: nx.DiGraph, impl_info: Dict[str, Any], module_name: str) -> str:
    """
    Add a Rust impl block node to the graph.
    
    Args:
        graph: NetworkX directed graph
        impl_info: Impl block information dictionary
        module_name: Name of the containing module
        
    Returns:
        str: Node ID of the added impl block
    """
    node_id = create_node_id("impl", f"{impl_info['type_name']}_impl", module_name)
    
    # Add node with impl properties
    graph.add_node(node_id, **{
        'type': 'impl',
        'name': f"{impl_info['type_name']}_impl",
        'file_path': impl_info['file_path'],
        'line_number': impl_info['line_number'],
        'trait_name': impl_info.get('trait_name', ''),
        'type_name': impl_info.get('type_name', ''),
        'methods': impl_info.get('methods', []),
        'is_trait_impl': impl_info.get('is_trait_impl', False)
    })
    
    logger.info(f"Added impl node: {node_id}")
    return node_id


def add_defined_in_edge(graph: nx.DiGraph, element_node_id: str, module_node_id: str, element_type: str):
    """
    Add a DEFINED_IN relationship between an element and its module.
    
    Args:
        graph: NetworkX directed graph
        element_node_id: Node ID of the element (function/class)
        module_node_id: Node ID of the module
        element_type: Type of the element (function/class)
    """
    graph.add_edge(element_node_id, module_node_id, **{
        'type': 'DEFINED_IN',
        'definition_type': element_type
    })
    
    logger.info(f"Added DEFINED_IN edge: {element_node_id} -> {module_node_id}")


def add_calls_edge(graph: nx.DiGraph, caller_id: str, callee_id: str, line_number: int = 0):
    """
    Add a CALLS relationship between functions.
    
    Args:
        graph: NetworkX directed graph
        caller_id: Node ID of the calling function
        callee_id: Node ID of the called function
        line_number: Line number where the call occurs
    """
    graph.add_edge(caller_id, callee_id, **{
        'type': 'CALLS',
        'line_number': line_number,
        'call_type': 'direct'
    })
    
    logger.info(f"Added CALLS edge: {caller_id} -> {callee_id}")


def add_inherits_edge(graph: nx.DiGraph, child_class_id: str, parent_class_id: str):
    """
    Add an INHERITS relationship between classes.
    
    Args:
        graph: NetworkX directed graph
        child_class_id: Node ID of the child class
        parent_class_id: Node ID of the parent class
    """
    graph.add_edge(child_class_id, parent_class_id, **{
        'type': 'INHERITS',
        'inheritance_type': 'single'
    })
    
    logger.info(f"Added INHERITS edge: {child_class_id} -> {parent_class_id}")


def add_has_method_edge(graph: nx.DiGraph, class_id: str, method_id: str, method_type: str = "instance"):
    """
    Add a HAS_METHOD relationship between a class and its method.
    
    Args:
        graph: NetworkX directed graph
        class_id: Node ID of the class
        method_id: Node ID of the method
        method_type: Type of method (instance, class, static)
    """
    # Determine visibility based on method name
    method_name = graph.nodes[method_id]['name']
    visibility = 'private' if method_name.startswith('_') else 'public'
    
    graph.add_edge(class_id, method_id, **{
        'type': 'HAS_METHOD',
        'method_type': method_type,
        'visibility': visibility
    })
    
    logger.info(f"Added HAS_METHOD edge: {class_id} -> {method_id}")


def add_imports_edge(graph: nx.DiGraph, importing_module_id: str, imported_module_id: str):
    """
    Add an IMPORTS relationship between modules.
    
    Args:
        graph: NetworkX directed graph
        importing_module_id: Node ID of the importing module
        imported_module_id: Node ID of the imported module
    """
    graph.add_edge(importing_module_id, imported_module_id, **{
        'type': 'IMPORTS',
        'import_type': 'direct',
        'alias': None
    })
    
    logger.info(f"Added IMPORTS edge: {importing_module_id} -> {imported_module_id}")


def resolve_function_reference(func_name: str, graph: nx.DiGraph, caller_module: str) -> Optional[str]:
    """
    Resolve a function reference to a node ID in the graph.
    
    Args:
        func_name: Name of the function being called
        graph: NetworkX directed graph
        caller_module: Module name of the caller
        
    Returns:
        Optional[str]: Node ID of the called function, or None if not found
    """
    # First try to find the function in the same module
    same_module_id = f"{caller_module}.{func_name}"
    if same_module_id in graph.nodes:
        return same_module_id
    
    # Try to find the function in any module
    for node_id in graph.nodes:
        if (graph.nodes[node_id]['type'] == 'function' and 
            graph.nodes[node_id]['name'] == func_name):
            return node_id
    
    # Handle method calls (e.g., "obj.method")
    if '.' in func_name:
        parts = func_name.split('.')
        method_name = parts[-1]
        for node_id in graph.nodes:
            if (graph.nodes[node_id]['type'] == 'function' and 
                graph.nodes[node_id]['name'] == method_name):
                return node_id
    
    return None


def build_knowledge_graph(parsing_results: List[Dict[str, Any]]) -> nx.DiGraph:
    """
    Build a knowledge graph from parsing results.
    
    Args:
        parsing_results: List of parsed code elements
        
    Returns:
        nx.DiGraph: Constructed knowledge graph
    """
    graph = nx.DiGraph()
    
    # Separate elements by type
    modules = [elem for elem in parsing_results if elem['type'] == 'module']
    functions = [elem for elem in parsing_results if elem['type'] == 'function']
    classes = [elem for elem in parsing_results if elem['type'] == 'class']
    structs = [elem for elem in parsing_results if elem['type'] == 'struct']
    enums = [elem for elem in parsing_results if elem['type'] == 'enum']
    traits = [elem for elem in parsing_results if elem['type'] == 'trait']
    impls = [elem for elem in parsing_results if elem['type'] == 'impl']
    
    # Create a mapping from file path to module
    file_to_module = {}
    for module in modules:
        file_to_module[module['file_path']] = module['name']
    
    # Add module nodes
    module_nodes = {}
    for module in modules:
        node_id = add_module_node(graph, module)
        module_nodes[module['name']] = node_id
    
    # Add function nodes and DEFINED_IN relationships
    function_nodes = {}
    for func in functions:
        module_name = file_to_module.get(func['file_path'], 'unknown')
        node_id = add_function_node(graph, func, module_name)
        function_nodes[f"{module_name}.{func['name']}"] = node_id
        
        # Add DEFINED_IN relationship
        if module_name in module_nodes:
            add_defined_in_edge(graph, node_id, module_nodes[module_name], 'function')
    
    # Add class nodes and DEFINED_IN relationships
    class_nodes = {}
    for cls in classes:
        module_name = file_to_module.get(cls['file_path'], 'unknown')
        node_id = add_class_node(graph, cls, module_name)
        class_nodes[f"{module_name}.{cls['name']}"] = node_id
        
        # Add DEFINED_IN relationship
        if module_name in module_nodes:
            add_defined_in_edge(graph, node_id, module_nodes[module_name], 'class')
        
        # Add inheritance relationships
        for base_class in cls.get('base_classes', []):
            # Try to resolve base class reference
            base_class_id = resolve_function_reference(base_class, graph, module_name)
            if base_class_id and graph.nodes[base_class_id]['type'] == 'class':
                add_inherits_edge(graph, node_id, base_class_id)
        
        # Add method relationships
        for method in cls.get('methods', []):
            method_node_id = add_function_node(graph, method, module_name)
            method_full_id = f"{module_name}.{method['name']}"
            function_nodes[method_full_id] = method_node_id
            add_has_method_edge(graph, node_id, method_node_id)
    
    # Add struct nodes and DEFINED_IN relationships
    struct_nodes = {}
    for struct in structs:
        module_name = file_to_module.get(struct['file_path'], 'unknown')
        node_id = add_struct_node(graph, struct, module_name)
        struct_nodes[f"{module_name}.{struct['name']}"] = node_id
        
        # Add DEFINED_IN relationship
        if module_name in module_nodes:
            add_defined_in_edge(graph, node_id, module_nodes[module_name], 'struct')
    
    # Add enum nodes and DEFINED_IN relationships
    enum_nodes = {}
    for enum in enums:
        module_name = file_to_module.get(enum['file_path'], 'unknown')
        node_id = add_enum_node(graph, enum, module_name)
        enum_nodes[f"{module_name}.{enum['name']}"] = node_id
        
        # Add DEFINED_IN relationship
        if module_name in module_nodes:
            add_defined_in_edge(graph, node_id, module_nodes[module_name], 'enum')
    
    # Add trait nodes and DEFINED_IN relationships
    trait_nodes = {}
    for trait in traits:
        module_name = file_to_module.get(trait['file_path'], 'unknown')
        node_id = add_trait_node(graph, trait, module_name)
        trait_nodes[f"{module_name}.{trait['name']}"] = node_id
        
        # Add DEFINED_IN relationship
        if module_name in module_nodes:
            add_defined_in_edge(graph, node_id, module_nodes[module_name], 'trait')
        
        # Add supertrait relationships
        for supertrait in trait.get('supertraits', []):
            supertrait_id = resolve_function_reference(supertrait, graph, module_name)
            if supertrait_id and graph.nodes[supertrait_id]['type'] == 'trait':
                add_inherits_edge(graph, node_id, supertrait_id)
        
        # Add method relationships
        for method in trait.get('methods', []):
            method_node_id = add_function_node(graph, method, module_name)
            method_full_id = f"{module_name}.{method['name']}"
            function_nodes[method_full_id] = method_node_id
            add_has_method_edge(graph, node_id, method_node_id)
    
    # Add impl nodes and DEFINED_IN relationships
    impl_nodes = {}
    for impl in impls:
        module_name = file_to_module.get(impl['file_path'], 'unknown')
        node_id = add_impl_node(graph, impl, module_name)
        impl_nodes[f"{module_name}.{impl['type_name']}_impl"] = node_id
        
        # Add DEFINED_IN relationship
        if module_name in module_nodes:
            add_defined_in_edge(graph, node_id, module_nodes[module_name], 'impl')
        
        # Add method relationships
        for method in impl.get('methods', []):
            method_node_id = add_function_node(graph, method, module_name)
            method_full_id = f"{module_name}.{method['name']}"
            function_nodes[method_full_id] = method_node_id
            add_has_method_edge(graph, node_id, method_node_id)
    
    # Add function call relationships
    for func in functions:
        module_name = file_to_module.get(func['file_path'], 'unknown')
        caller_id = f"{module_name}.{func['name']}"
        
        if caller_id in function_nodes:
            for called_func in func.get('calls', []):
                callee_id = resolve_function_reference(called_func, graph, module_name)
                if callee_id:
                    add_calls_edge(graph, function_nodes[caller_id], callee_id, func['line_number'])
    
    # Add import relationships
    for module in modules:
        module_id = module_nodes[module['name']]
        for imported_module in module.get('imports', []):
            # Try to find the imported module in our graph
            imported_module_id = f"module.{imported_module}"
            if imported_module_id in module_nodes:
                add_imports_edge(graph, module_id, module_nodes[imported_module])
    
    logger.info(f"Built knowledge graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
    return graph


if __name__ == "__main__":
    # Test with dummy data
    dummy_data = [
        {
            'type': 'module',
            'name': 'test_module',
            'file_path': 'test.py',
            'imports': ['os', 'sys'],
            'docstring': 'Test module'
        },
        {
            'type': 'function',
            'name': 'test_function',
            'file_path': 'test.py',
            'line_number': 5,
            'parameters': ['arg1', 'arg2'],
            'return_type': 'str',
            'docstring': 'Test function',
            'is_async': False,
            'calls': ['print', 'len']
        },
        {
            'type': 'class',
            'name': 'TestClass',
            'file_path': 'test.py',
            'line_number': 10,
            'base_classes': ['BaseClass'],
            'docstring': 'Test class',
            'methods': [
                {
                    'type': 'function',
                    'name': 'method1',
                    'file_path': 'test.py',
                    'line_number': 12,
                    'parameters': ['self'],
                    'return_type': '',
                    'docstring': 'Test method',
                    'is_async': False,
                    'calls': []
                }
            ]
        }
    ]
    
    try:
        graph = build_knowledge_graph(dummy_data)
        print(f"Successfully built graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
        
        # Print some basic info
        print("\nNodes:")
        for node_id in graph.nodes():
            node_data = graph.nodes[node_id]
            print(f"  {node_id}: {node_data['type']} - {node_data['name']}")
        
        print("\nEdges:")
        for edge in graph.edges(data=True):
            print(f"  {edge[0]} -> {edge[1]} ({edge[2]['type']})")
            
    except Exception as e:
        print(f"Error building graph: {e}")
        import traceback
        traceback.print_exc()