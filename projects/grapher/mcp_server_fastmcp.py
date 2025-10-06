"""
MCP Server for Code Knowledge Graph using FastMCP

This MCP server exposes the code knowledge graph functionality through the Model Context Protocol,
allowing Cursor IDE and other MCP-compatible tools to query codebases using natural language.
"""

import json
import logging
from typing import Optional

from fastmcp import FastMCP

# Import our code knowledge graph modules
from main_pipeline import ask_question_about_code, analyze_project_structure, build_project_graph

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables to cache the graph
current_graph = None
current_project_path = None

# Create FastMCP server
mcp = FastMCP("code-knowledge-graph")


@mcp.tool()
def analyze_project(project_path: str) -> str:
    """Analyze the structure of a multi-language project (Python and Rust) and build its knowledge graph"""
    global current_graph, current_project_path
    
    try:
        # Build the knowledge graph
        logger.info(f"Analyzing project: {project_path}")
        current_graph = build_project_graph(project_path)
        current_project_path = project_path
        
        # Get project structure
        structure = analyze_project_structure(project_path)
        
        # Count nodes by type for better reporting
        node_counts = {}
        for node_data in current_graph.nodes.values():
            node_type = node_data.get('type', 'unknown')
            node_counts[node_type] = node_counts.get(node_type, 0) + 1
        
        result = {
            "status": "success",
            "message": f"Successfully analyzed multi-language project at {project_path}",
            "project_structure": structure,
            "graph_info": {
                "nodes": current_graph.number_of_nodes(),
                "edges": current_graph.number_of_edges(),
                "node_types": node_counts
            }
        }
        
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Error analyzing project: {e}")
        return f"Error: {str(e)}"


@mcp.tool()
def ask_about_code(question: str) -> str:
    """Ask a natural language question about the analyzed codebase"""
    global current_graph, current_project_path
    
    if not current_graph:
        return "Error: No project analyzed yet. Use analyze_project first."
    
    try:
        logger.info(f"Answering question: {question}")
        answer = ask_question_about_code(current_project_path, question)
        return answer
    except Exception as e:
        logger.error(f"Error answering question: {e}")
        return f"Error: {str(e)}"


@mcp.tool()
def get_project_structure() -> str:
    """Get detailed information about the current project structure"""
    global current_graph, current_project_path
    
    if not current_graph:
        return "Error: No project analyzed yet. Use analyze_project first."
    
    try:
        structure = analyze_project_structure(current_project_path)
        
        # Add additional graph-specific information
        graph_info = {
            "nodes_by_type": {},
            "edges_by_type": {},
            "most_connected_nodes": []
        }
        
        # Count nodes by type
        for node_data in current_graph.nodes.values():
            node_type = node_data.get('type', 'unknown')
            graph_info["nodes_by_type"][node_type] = graph_info["nodes_by_type"].get(node_type, 0) + 1
        
        # Count edges by type
        for _, _, edge_data in current_graph.edges(data=True):
            edge_type = edge_data.get('type', 'unknown')
            graph_info["edges_by_type"][edge_type] = graph_info["edges_by_type"].get(edge_type, 0) + 1
        
        # Find most connected nodes
        node_degrees = {}
        for node in current_graph.nodes():
            node_degrees[node] = len(list(current_graph.neighbors(node))) + len(list(current_graph.predecessors(node)))
        
        graph_info["most_connected_nodes"] = sorted(node_degrees.items(), key=lambda x: x[1], reverse=True)[:10]
        
        result = {
            "project_structure": structure,
            "graph_analysis": graph_info
        }
        
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Error getting project structure: {e}")
        return f"Error: {str(e)}"


@mcp.tool()
def find_functions(module: Optional[str] = None, parameter: Optional[str] = None, class_name: Optional[str] = None) -> str:
    """Find functions matching specific criteria"""
    global current_graph
    
    if not current_graph:
        return "Error: No project analyzed yet. Use analyze_project first."
    
    try:
        functions = []
        
        for node_id, node_data in current_graph.nodes(data=True):
            if node_data['type'] != 'function':
                continue
            
            # Apply filters
            if module:
                if not any(current_graph.nodes.get(pred, {}).get('name') == module 
                         for pred in current_graph.predecessors(node_id)
                         if current_graph.nodes.get(pred, {}).get('type') == 'module'):
                    continue
            
            if parameter:
                if parameter not in node_data.get('parameters', []):
                    continue
            
            if class_name:
                if not any(current_graph.nodes.get(pred, {}).get('name') == class_name
                         for pred in current_graph.predecessors(node_id)
                         if current_graph.nodes.get(pred, {}).get('type') == 'class'):
                    continue
            
            functions.append({
                "name": node_data['name'],
                "file_path": node_data['file_path'],
                "line_number": node_data['line_number'],
                "parameters": node_data.get('parameters', []),
                "docstring": node_data.get('docstring', ''),
                "is_async": node_data.get('is_async', False)
            })
        
        result = {
            "functions": functions,
            "count": len(functions),
            "filters_applied": {
                "module": module,
                "parameter": parameter,
                "class": class_name
            }
        }
        
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Error finding functions: {e}")
        return f"Error: {str(e)}"


@mcp.tool()
def get_function_details(function_name: str, module: Optional[str] = None) -> str:
    """Get detailed information about a specific function"""
    global current_graph
    
    if not current_graph:
        return "Error: No project analyzed yet. Use analyze_project first."
    
    try:
        # Find the function node
        function_node = None
        for node_id, node_data in current_graph.nodes(data=True):
            if (node_data['type'] == 'function' and 
                node_data['name'] == function_name):
                
                if module:
                    # Check if it's in the specified module
                    for pred in current_graph.predecessors(node_id):
                        pred_data = current_graph.nodes.get(pred, {})
                        if (pred_data.get('type') == 'module' and 
                            pred_data.get('name') == module):
                            function_node = node_id
                            break
                else:
                    function_node = node_id
                    break
        
        if not function_node:
            return f"Function '{function_name}' not found"
        
        node_data = current_graph.nodes[function_node]
        
        # Get relationships
        calls = []
        called_by = []
        belongs_to_class = None
        belongs_to_module = None
        
        for succ in current_graph.successors(function_node):
            edge_data = current_graph.edges[function_node, succ]
            if edge_data.get('type') == 'CALLS':
                succ_data = current_graph.nodes[succ]
                calls.append({
                    "name": succ_data['name'],
                    "type": succ_data['type'],
                    "line_number": edge_data.get('line_number')
                })
        
        for pred in current_graph.predecessors(function_node):
            edge_data = current_graph.edges[pred, function_node]
            pred_data = current_graph.nodes[pred]
            
            if edge_data.get('type') == 'CALLS':
                called_by.append({
                    "name": pred_data['name'],
                    "type": pred_data['type']
                })
            elif edge_data.get('type') == 'HAS_METHOD':
                belongs_to_class = pred_data['name']
            elif edge_data.get('type') == 'DEFINED_IN' and pred_data['type'] == 'module':
                belongs_to_module = pred_data['name']
        
        result = {
            "function": {
                "name": node_data['name'],
                "file_path": node_data['file_path'],
                "line_number": node_data['line_number'],
                "parameters": node_data.get('parameters', []),
                "return_type": node_data.get('return_type', ''),
                "docstring": node_data.get('docstring', ''),
                "is_async": node_data.get('is_async', False)
            },
            "relationships": {
                "calls": calls,
                "called_by": called_by,
                "belongs_to_class": belongs_to_class,
                "belongs_to_module": belongs_to_module
            }
        }
        
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Error getting function details: {e}")
        return f"Error: {str(e)}"


@mcp.tool()
def trace_function_calls(function_name: str, direction: str = "both", max_depth: int = 3) -> str:
    """Trace the call chain for a specific function"""
    global current_graph
    
    if not current_graph:
        return "Error: No project analyzed yet. Use analyze_project first."
    
    try:
        # Find the function node
        function_node = None
        for node_id, node_data in current_graph.nodes(data=True):
            if node_data['type'] == 'function' and node_data['name'] == function_name:
                function_node = node_id
                break
        
        if not function_node:
            return f"Function '{function_name}' not found"
        
        call_chain = []
        visited = set()
        
        def trace_calls(node, depth, path_type):
            if depth > max_depth or node in visited:
                return
            
            visited.add(node)
            node_data = current_graph.nodes[node]
            
            if direction in ["outgoing", "both"]:
                for succ in current_graph.successors(node):
                    edge_data = current_graph.edges[node, succ]
                    if edge_data.get('type') == 'CALLS':
                        succ_data = current_graph.nodes[succ]
                        call_chain.append({
                            "from": node_data['name'],
                            "to": succ_data['name'],
                            "type": succ_data['type'],
                            "direction": "outgoing",
                            "depth": depth,
                            "line_number": edge_data.get('line_number')
                        })
                        trace_calls(succ, depth + 1, "outgoing")
            
            if direction in ["incoming", "both"]:
                for pred in current_graph.predecessors(node):
                    edge_data = current_graph.edges[pred, node]
                    if edge_data.get('type') == 'CALLS':
                        pred_data = current_graph.nodes[pred]
                        call_chain.append({
                            "from": pred_data['name'],
                            "to": node_data['name'],
                            "type": node_data['type'],
                            "direction": "incoming",
                            "depth": depth
                        })
                        trace_calls(pred, depth + 1, "incoming")
        
        trace_calls(function_node, 0, "start")
        
        result = {
            "function": function_name,
            "call_chain": call_chain,
            "total_calls": len(call_chain),
            "max_depth": max_depth,
            "direction": direction
        }
        
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Error tracing function calls: {e}")
        return f"Error: {str(e)}"


@mcp.tool()
def get_class_hierarchy(class_name: str, direction: str = "both") -> str:
    """Get the inheritance hierarchy for a class"""
    global current_graph
    
    if not current_graph:
        return "Error: No project analyzed yet. Use analyze_project first."
    
    try:
        # Find the class node
        class_node = None
        for node_id, node_data in current_graph.nodes(data=True):
            if node_data['type'] == 'class' and node_data['name'] == class_name:
                class_node = node_id
                break
        
        if not class_node:
            return f"Class '{class_name}' not found"
        
        hierarchy = {
            "class": class_name,
            "parents": [],
            "children": [],
            "methods": []
        }
        
        class_data = current_graph.nodes[class_node]
        
        # Get parents (inheritance)
        if direction in ["parents", "both"]:
            for succ in current_graph.successors(class_node):
                edge_data = current_graph.edges[class_node, succ]
                if edge_data.get('type') == 'INHERITS':
                    succ_data = current_graph.nodes[succ]
                    hierarchy["parents"].append({
                        "name": succ_data['name'],
                        "file_path": succ_data['file_path']
                    })
        
        # Get children (classes that inherit from this class)
        if direction in ["children", "both"]:
            for pred in current_graph.predecessors(class_node):
                edge_data = current_graph.edges[pred, class_node]
                if edge_data.get('type') == 'INHERITS':
                    pred_data = current_graph.nodes[pred]
                    hierarchy["children"].append({
                        "name": pred_data['name'],
                        "file_path": pred_data['file_path']
                    })
        
        # Get methods
        for succ in current_graph.successors(class_node):
            edge_data = current_graph.edges[class_node, succ]
            if edge_data.get('type') == 'HAS_METHOD':
                succ_data = current_graph.nodes[succ]
                hierarchy["methods"].append({
                    "name": succ_data['name'],
                    "parameters": succ_data.get('parameters', []),
                    "line_number": succ_data['line_number'],
                    "visibility": edge_data.get('visibility', 'public')
                })
        
        return json.dumps(hierarchy, indent=2)
    except Exception as e:
        logger.error(f"Error getting class hierarchy: {e}")
        return f"Error: {str(e)}"


@mcp.tool()
def find_rust_items(item_type: str = "all", visibility: Optional[str] = None) -> str:
    """Find Rust-specific items (structs, enums, traits, impls)"""
    global current_graph
    
    if not current_graph:
        return "Error: No project analyzed yet. Use analyze_project first."
    
    try:
        rust_items = []
        
        for node_id, node_data in current_graph.nodes(data=True):
            if node_data['type'] in ['struct', 'enum', 'trait', 'impl']:
                # Apply type filter
                if item_type != "all" and node_data['type'] != item_type:
                    continue
                
                # Apply visibility filter
                if visibility and node_data.get('visibility') != visibility:
                    continue
                
                item_info = {
                    "type": node_data['type'],
                    "name": node_data['name'],
                    "file_path": node_data['file_path'],
                    "line_number": node_data['line_number'],
                    "visibility": node_data.get('visibility', 'private')
                }
                
                # Add type-specific information
                if node_data['type'] == 'struct':
                    item_info.update({
                        "fields": node_data.get('fields', []),
                        "is_unit_struct": node_data.get('is_unit_struct', False),
                        "is_tuple_struct": node_data.get('is_tuple_struct', False)
                    })
                elif node_data['type'] == 'enum':
                    item_info["variants"] = node_data.get('variants', [])
                elif node_data['type'] == 'trait':
                    item_info.update({
                        "methods": [method['name'] for method in node_data.get('methods', [])],
                        "supertraits": node_data.get('supertraits', [])
                    })
                elif node_data['type'] == 'impl':
                    item_info.update({
                        "trait_name": node_data.get('trait_name', ''),
                        "type_name": node_data.get('type_name', ''),
                        "is_trait_impl": node_data.get('is_trait_impl', False),
                        "methods": [method['name'] for method in node_data.get('methods', [])]
                    })
                
                rust_items.append(item_info)
        
        result = {
            "rust_items": rust_items,
            "count": len(rust_items),
            "filters": {
                "item_type": item_type,
                "visibility": visibility
            }
        }
        
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Error finding Rust items: {e}")
        return f"Error: {str(e)}"


@mcp.tool()
def get_trait_implementations(trait_name: str) -> str:
    """Get all implementations of a specific trait"""
    global current_graph
    
    if not current_graph:
        return "Error: No project analyzed yet. Use analyze_project first."
    
    try:
        implementations = []
        
        for node_id, node_data in current_graph.nodes(data=True):
            if node_data['type'] == 'impl' and node_data.get('trait_name') == trait_name:
                implementations.append({
                    "type_name": node_data.get('type_name', ''),
                    "file_path": node_data['file_path'],
                    "line_number": node_data['line_number'],
                    "methods": [method['name'] for method in node_data.get('methods', [])]
                })
        
        result = {
            "trait": trait_name,
            "implementations": implementations,
            "count": len(implementations)
        }
        
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Error getting trait implementations: {e}")
        return f"Error: {str(e)}"


if __name__ == "__main__":
    # Run the FastMCP server with stdio protocol for MCP
    mcp.run()
