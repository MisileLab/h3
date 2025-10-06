"""
MCP Server for Code Knowledge Graph

This MCP server exposes the code knowledge graph functionality through the Model Context Protocol,
allowing Cursor IDE and other MCP-compatible tools to query codebases using natural language.
"""

import asyncio
import json
import logging
import sys
from typing import Any, Dict, List, Optional
from pathlib import Path

# MCP imports
try:
    import mcp.server.stdio
    import mcp.types as types
    from mcp.server import Server
except ImportError:
    print("MCP not installed. Install with: uv add mcp")
    sys.exit(1)

# Import our code knowledge graph modules
from main_pipeline import ask_question_about_code, analyze_project_structure, build_project_graph
from code_parser import parse_python_file
from graph_builder import build_knowledge_graph
import networkx as nx

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables to cache the graph
current_graph = None
current_project_path = None

# Create MCP server
server = Server("code-knowledge-graph")


@server.list_tools()
async def handle_list_tools() -> List[types.Tool]:
    """List available tools."""
    return [
        types.Tool(
            name="analyze_project",
            description="Analyze the structure of a Python project and build its knowledge graph",
            inputSchema={
                "type": "object",
                "properties": {
                    "project_path": {
                        "type": "string",
                        "description": "Path to the Python project directory to analyze"
                    }
                },
                "required": ["project_path"]
            }
        ),
        types.Tool(
            name="ask_about_code",
            description="Ask a natural language question about the analyzed codebase",
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "Natural language question about the code (e.g., 'What functions call the authenticate_user function?')"
                    }
                },
                "required": ["question"]
            }
        ),
        types.Tool(
            name="get_project_structure",
            description="Get detailed information about the current project structure",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        types.Tool(
            name="find_functions",
            description="Find functions matching specific criteria",
            inputSchema={
                "type": "object",
                "properties": {
                    "module": {
                        "type": "string",
                        "description": "Filter by module name (optional)"
                    },
                    "parameter": {
                        "type": "string",
                        "description": "Filter by parameter name (optional)"
                    },
                    "class": {
                        "type": "string",
                        "description": "Filter by class name (optional)"
                    }
                },
                "required": []
            }
        ),
        types.Tool(
            name="get_function_details",
            description="Get detailed information about a specific function",
            inputSchema={
                "type": "object",
                "properties": {
                    "function_name": {
                        "type": "string",
                        "description": "Name of the function to analyze"
                    },
                    "module": {
                        "type": "string",
                        "description": "Module name (optional, helps with disambiguation)"
                    }
                },
                "required": ["function_name"]
            }
        ),
        types.Tool(
            name="trace_function_calls",
            description="Trace the call chain for a specific function",
            inputSchema={
                "type": "object",
                "properties": {
                    "function_name": {
                        "type": "string",
                        "description": "Name of the function to trace"
                    },
                    "direction": {
                        "type": "string",
                        "enum": ["both", "incoming", "outgoing"],
                        "description": "Direction of call tracing (default: both)",
                        "default": "both"
                    },
                    "max_depth": {
                        "type": "integer",
                        "description": "Maximum depth of call tracing (default: 3)",
                        "default": 3
                    }
                },
                "required": ["function_name"]
            }
        ),
        types.Tool(
            name="get_class_hierarchy",
            description="Get the inheritance hierarchy for a class",
            inputSchema={
                "type": "object",
                "properties": {
                    "class_name": {
                        "type": "string",
                        "description": "Name of the class to analyze"
                    },
                    "direction": {
                        "type": "string",
                        "enum": ["both", "parents", "children"],
                        "description": "Direction of hierarchy tracing (default: both)",
                        "default": "both"
                    }
                },
                "required": ["class_name"]
            }
        )
    ]


@server.call_tool()
async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> List[types.TextContent]:
    """Handle tool calls."""
    global current_graph, current_project_path
    
    try:
        if name == "analyze_project":
            project_path = arguments.get("project_path")
            if not project_path:
                return [types.TextContent(type="text", text="Error: project_path is required")]
            
            # Build the knowledge graph
            logger.info(f"Analyzing project: {project_path}")
            current_graph = build_project_graph(project_path)
            current_project_path = project_path
            
            # Get project structure
            structure = analyze_project_structure(project_path)
            
            result = {
                "status": "success",
                "message": f"Successfully analyzed project at {project_path}",
                "project_structure": structure,
                "graph_info": {
                    "nodes": current_graph.number_of_nodes(),
                    "edges": current_graph.number_of_edges()
                }
            }
            
            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
        
        elif name == "ask_about_code":
            if not current_graph:
                return [types.TextContent(type="text", text="Error: No project analyzed yet. Use analyze_project first.")]
            
            question = arguments.get("question")
            if not question:
                return [types.TextContent(type="text", text="Error: question is required")]
            
            logger.info(f"Answering question: {question}")
            answer = ask_question_about_code(current_project_path, question)
            
            return [types.TextContent(type="text", text=answer)]
        
        elif name == "get_project_structure":
            if not current_graph:
                return [types.TextContent(type="text", text="Error: No project analyzed yet. Use analyze_project first.")]
            
            structure = analyze_project_structure(current_project_path)
            
            # Add additional graph-specific information
            graph_info = {
                "nodes_by_type": {},
                "edges_by_type": {},
                "most_connected_nodes": []
            }
            
            # Count nodes by type
            for node_data in current_graph.nodes.values():
                node_type = node_data['type']
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
            
            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
        
        elif name == "find_functions":
            if not current_graph:
                return [types.TextContent(type="text", text="Error: No project analyzed yet. Use analyze_project first.")]
            
            module_filter = arguments.get("module")
            parameter_filter = arguments.get("parameter")
            class_filter = arguments.get("class")
            
            functions = []
            
            for node_id, node_data in current_graph.nodes(data=True):
                if node_data['type'] != 'function':
                    continue
                
                # Apply filters
                if module_filter:
                    if not any(current_graph.nodes.get(pred, {}).get('name') == module_filter 
                             for pred in current_graph.predecessors(node_id)
                             if current_graph.nodes.get(pred, {}).get('type') == 'module'):
                        continue
                
                if parameter_filter:
                    if parameter_filter not in node_data.get('parameters', []):
                        continue
                
                if class_filter:
                    if not any(current_graph.nodes.get(pred, {}).get('name') == class_filter
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
                    "module": module_filter,
                    "parameter": parameter_filter,
                    "class": class_filter
                }
            }
            
            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
        
        elif name == "get_function_details":
            if not current_graph:
                return [types.TextContent(type="text", text="Error: No project analyzed yet. Use analyze_project first.")]
            
            function_name = arguments.get("function_name")
            module_filter = arguments.get("module")
            
            # Find the function node
            function_node = None
            for node_id, node_data in current_graph.nodes(data=True):
                if (node_data['type'] == 'function' and 
                    node_data['name'] == function_name):
                    
                    if module_filter:
                        # Check if it's in the specified module
                        for pred in current_graph.predecessors(node_id):
                            pred_data = current_graph.nodes.get(pred, {})
                            if (pred_data.get('type') == 'module' and 
                                pred_data.get('name') == module_filter):
                                function_node = node_id
                                break
                    else:
                        function_node = node_id
                        break
            
            if not function_node:
                return [types.TextContent(type="text", text=f"Function '{function_name}' not found")]
            
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
            
            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
        
        elif name == "trace_function_calls":
            if not current_graph:
                return [types.TextContent(type="text", text="Error: No project analyzed yet. Use analyze_project first.")]
            
            function_name = arguments.get("function_name")
            direction = arguments.get("direction", "both")
            max_depth = arguments.get("max_depth", 3)
            
            # Find the function node
            function_node = None
            for node_id, node_data in current_graph.nodes(data=True):
                if node_data['type'] == 'function' and node_data['name'] == function_name:
                    function_node = node_id
                    break
            
            if not function_node:
                return [types.TextContent(type="text", text=f"Function '{function_name}' not found")]
            
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
            
            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
        
        elif name == "get_class_hierarchy":
            if not current_graph:
                return [types.TextContent(type="text", text="Error: No project analyzed yet. Use analyze_project first.")]
            
            class_name = arguments.get("class_name")
            direction = arguments.get("direction", "both")
            
            # Find the class node
            class_node = None
            for node_id, node_data in current_graph.nodes(data=True):
                if node_data['type'] == 'class' and node_data['name'] == class_name:
                    class_node = node_id
                    break
            
            if not class_node:
                return [types.TextContent(type="text", text=f"Class '{class_name}' not found")]
            
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
            
            return [types.TextContent(type="text", text=json.dumps(hierarchy, indent=2))]
        
        else:
            return [types.TextContent(type="text", text=f"Unknown tool: {name}")]
    
    except Exception as e:
        logger.error(f"Error in tool {name}: {str(e)}")
        return [types.TextContent(type="text", text=f"Error: {str(e)}")]


async def main():
    """Main function to run the MCP server."""
    # Use stdio transport
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())