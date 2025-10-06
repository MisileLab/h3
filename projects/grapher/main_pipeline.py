"""
Main Pipeline Module

This module integrates all components to create a complete RAG pipeline
for answering questions about codebases using the knowledge graph.
"""

import os
import glob
from typing import List, Dict, Any, Optional
import networkx as nx
import logging

from code_parser import parse_python_file
from rust_parser import parse_rust_file
from graph_builder import build_knowledge_graph
from query_translator import translate_natural_language_to_query

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def find_code_files(project_path: str) -> Dict[str, List[str]]:
    """
    Find all code files (Python and Rust) in the given project path.
    
    Args:
        project_path: Path to the project directory
        
    Returns:
        Dict[str, List[str]]: Dictionary with 'python' and 'rust' keys containing file paths
    """
    python_files = []
    rust_files = []
    
    # Use glob to find all .py files recursively
    python_pattern = os.path.join(project_path, "**", "*.py")
    python_files = glob.glob(python_pattern, recursive=True)
    
    # Use glob to find all .rs files recursively
    rust_pattern = os.path.join(project_path, "**", "*.rs")
    rust_files = glob.glob(rust_pattern, recursive=True)
    
    logger.info(f"Found {len(python_files)} Python files and {len(rust_files)} Rust files in {project_path}")
    return {
        'python': python_files,
        'rust': rust_files
    }


def parse_project_files(project_path: str) -> List[Dict[str, Any]]:
    """
    Parse all code files (Python and Rust) in the project.
    
    Args:
        project_path: Path to the project directory
        
    Returns:
        List[Dict]: List of all parsed code elements
    """
    code_files = find_code_files(project_path)
    all_results = []
    
    # Parse Python files
    for file_path in code_files['python']:
        try:
            logger.info(f"Parsing Python file: {file_path}")
            parsing_results = parse_python_file(file_path)
            all_results.extend(parsing_results)
        except Exception as e:
            logger.error(f"Error parsing Python file {file_path}: {e}")
            continue
    
    # Parse Rust files
    for file_path in code_files['rust']:
        try:
            logger.info(f"Parsing Rust file: {file_path}")
            parsing_results = parse_rust_file(file_path)
            all_results.extend(parsing_results)
        except Exception as e:
            logger.error(f"Error parsing Rust file {file_path}: {e}")
            continue
    
    total_files = len(code_files['python']) + len(code_files['rust'])
    logger.info(f"Parsed {len(all_results)} code elements from {total_files} files ({len(code_files['python'])} Python, {len(code_files['rust'])} Rust)")
    return all_results


def build_project_graph(project_path: str) -> nx.DiGraph:
    """
    Build a knowledge graph for the entire project.
    
    Args:
        project_path: Path to the project directory
        
    Returns:
        nx.DiGraph: Knowledge graph of the project
    """
    logger.info(f"Building knowledge graph for project: {project_path}")
    
    # Parse all files in the project
    parsing_results = parse_project_files(project_path)
    
    # Build the knowledge graph
    graph = build_knowledge_graph(parsing_results)
    
    logger.info(f"Built graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
    return graph


def execute_graph_query(query: str, graph: nx.DiGraph) -> List[Any]:
    """
    Execute a NetworkX graph query safely.
    
    Args:
        query: NetworkX query string
        graph: NetworkX graph
        
    Returns:
        List[Any]: Query results
    """
    try:
        # Create a safe execution environment
        safe_globals = {
            'G': graph,
            'nx': nx,
            'list': list,
            'set': set,
            'len': len,
            'str': str,
            'int': int,
            'float': float,
            'bool': bool,
        }
        
        # Execute the query
        result = eval(query, safe_globals)
        
        # Ensure result is a list
        if result is None:
            return []
        elif isinstance(result, (list, set)):
            return list(result)
        else:
            return [result]
            
    except Exception as e:
        logger.error(f"Error executing query '{query}': {e}")
        return []


def extract_code_context(node_ids: List[str], graph: nx.DiGraph) -> str:
    """
    Extract code context for the given node IDs.
    
    Args:
        node_ids: List of node IDs
        graph: NetworkX graph
        
    Returns:
        str: Formatted code context
    """
    if not node_ids:
        return "No relevant code found."
    
    context_parts = []
    
    for node_id in node_ids:
        if node_id not in graph.nodes:
            continue
            
        node_data = graph.nodes[node_id]
        context = ""
        
        # Format the node information
        if node_data['type'] == 'function':
            context = f"Function: {node_data['name']}\n"
            context += f"  File: {node_data['file_path']}:{node_data['line_number']}\n"
            if node_data['parameters']:
                context += f"  Parameters: {', '.join(node_data['parameters'])}\n"
            if node_data['docstring']:
                context += f"  Docstring: {node_data['docstring']}\n"
            context += f"  Async: {node_data['is_async']}\n"
            
        elif node_data['type'] == 'class':
            context = f"Class: {node_data['name']}\n"
            context += f"  File: {node_data['file_path']}:{node_data['line_number']}\n"
            if node_data['base_classes']:
                context += f"  Inherits from: {', '.join(node_data['base_classes'])}\n"
            if node_data['docstring']:
                context += f"  Docstring: {node_data['docstring']}\n"
                
        elif node_data['type'] == 'module':
            context = f"Module: {node_data['name']}\n"
            context += f"  File: {node_data['file_path']}\n"
            if node_data['imports']:
                context += f"  Imports: {', '.join(node_data['imports'])}\n"
            if node_data['docstring']:
                context += f"  Docstring: {node_data['docstring']}\n"
        
        context_parts.append(context)
        
        # Add related information (e.g., what this node calls or is called by)
        related_info = []
        
        # Find outgoing edges (what this node calls/contains)
        for _, target, edge_data in graph.out_edges(node_id, data=True):
            if target in graph.nodes:
                target_data = graph.nodes[target]
                relation = edge_data.get('type', 'unknown')
                related_info.append(f"  -> {relation} -> {target_data['type']}:{target_data['name']}")
        
        # Find incoming edges (what calls this node)
        for source, _, edge_data in graph.in_edges(node_id, data=True):
            if source in graph.nodes:
                source_data = graph.nodes[source]
                relation = edge_data.get('type', 'unknown')
                related_info.append(f"  <- {relation} <- {source_data['type']}:{source_data['name']}")
        
        if related_info:
            context_parts.append("Relationships:\n" + "\n".join(related_info))
        
        context_parts.append("-" * 50)
    
    return "\n".join(context_parts)


def call_llm(prompt: str) -> str:
    """
    Placeholder function for calling an LLM.
    In a real implementation, this would call OpenAI, Claude, or another LLM.
    
    Args:
        prompt: The prompt to send to the LLM
        
    Returns:
        str: LLM response
    """
    # This is a placeholder - in production, you'd use the actual OpenAI API
    # For demonstration, we'll return a simple response
    
    logger.info("LLM call (placeholder) - would send prompt to actual LLM")
    
    # Simple rule-based responses for demo purposes
    prompt_lower = prompt.lower()
    
    if "function" in prompt_lower and "call" in prompt_lower:
        return "Based on the code analysis, here are the functions that call the specified function."
    elif "method" in prompt_lower and "class" in prompt_lower:
        return "Based on the code analysis, here are the methods of the specified class."
    elif "module" in prompt_lower and "function" in prompt_lower:
        return "Based on the code analysis, here are the functions defined in the specified module."
    elif "inherit" in prompt_lower:
        return "Based on the code analysis, here are the inheritance relationships."
    else:
        return "Based on the code analysis, here's what I found about your question."


def ask_question_about_code(project_path: str, question: str) -> str:
    """
    Main function to answer questions about a codebase.
    
    Args:
        project_path: Path to the project directory
        question: Natural language question about the code
        
    Returns:
        str: Answer to the question
    """
    logger.info(f"Question: {question}")
    logger.info(f"Project path: {project_path}")
    
    try:
        # Step 1: Build the knowledge graph
        logger.info("Step 1: Building knowledge graph...")
        graph = build_project_graph(project_path)
        
        # Step 2: Translate the question to a graph query
        logger.info("Step 2: Translating question to graph query...")
        query = translate_natural_language_to_query(question, graph)
        logger.info(f"Generated query: {query}")
        
        # Step 3: Execute the query to get relevant nodes
        logger.info("Step 3: Executing graph query...")
        relevant_nodes = execute_graph_query(query, graph)
        logger.info(f"Found {len(relevant_nodes)} relevant nodes")
        
        # Step 4: Extract code context
        logger.info("Step 4: Extracting code context...")
        code_context = extract_code_context(relevant_nodes, graph)
        
        # Step 5: Generate final answer using LLM
        logger.info("Step 5: Generating answer...")
        
        # Create the prompt for the LLM
        prompt = f"""You are a code analysis assistant. Based on the following code context, answer the user's question.

Question: {question}

Code Context:
{code_context}

Please provide a clear and helpful answer based on the code information above."""
        
        answer = call_llm(prompt)
        
        # Add the raw query results for transparency
        final_answer = f"{answer}\n\n---\nQuery executed: {query}\nRelevant nodes found: {len(relevant_nodes)}"
        
        return final_answer
        
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        return f"Sorry, I encountered an error while processing your question: {str(e)}"


def analyze_project_structure(project_path: str) -> Dict[str, Any]:
    """
    Analyze the overall structure of a project.
    
    Args:
        project_path: Path to the project directory
        
    Returns:
        Dict[str, Any]: Project structure analysis
    """
    logger.info(f"Analyzing project structure: {project_path}")
    
    try:
        graph = build_project_graph(project_path)
        
        # Count different types of nodes
        node_counts = {}
        for node_data in graph.nodes.values():
            node_type = node_data['type']
            node_counts[node_type] = node_counts.get(node_type, 0) + 1
        
        # Count different types of edges
        edge_counts = {}
        for _, _, edge_data in graph.edges(data=True):
            edge_type = edge_data.get('type', 'unknown')
            edge_counts[edge_type] = edge_counts.get(edge_type, 0) + 1
        
        # Find most connected nodes
        node_degrees = {}
        for node in graph.nodes():
            node_degrees[node] = len(list(graph.neighbors(node))) + len(list(graph.predecessors(node)))
        most_connected = sorted(node_degrees.items(), key=lambda x: x[1], reverse=True)[:5]
        
        analysis = {
            'total_nodes': graph.number_of_nodes(),
            'total_edges': graph.number_of_edges(),
            'node_counts': node_counts,
            'edge_counts': edge_counts,
            'most_connected_nodes': most_connected,
            'modules': [data['name'] for data in graph.nodes.values() if data['type'] == 'module'],
            'classes': [data['name'] for data in graph.nodes.values() if data['type'] == 'class'],
            'functions': [data['name'] for data in graph.nodes.values() if data['type'] == 'function']
        }
        
        return analysis
        
    except Exception as e:
        logger.error(f"Error analyzing project structure: {e}")
        return {'error': str(e)}


if __name__ == "__main__":
    # Test the pipeline with the demo directory
    demo_path = "demo"
    
    if os.path.exists(demo_path):
        print("Testing pipeline with demo directory...")
        
        # Test project structure analysis
        print("\n=== Project Structure Analysis ===")
        structure = analyze_project_structure(demo_path)
        for key, value in structure.items():
            print(f"{key}: {value}")
        
        # Test some sample questions
        sample_questions = [
            "What functions are defined in the auth module?",
            "Show me all methods of the User class",
            "What functions call the authenticate_user function?"
        ]
        
        for question in sample_questions:
            print(f"\n=== Question: {question} ===")
            answer = ask_question_about_code(demo_path, question)
            print(answer)
            
    else:
        print(f"Demo directory {demo_path} not found. Please create demo files first.")