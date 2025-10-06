"""
Query Translator Module

This module translates natural language questions into NetworkX graph queries
using few-shot prompting techniques.
"""

import re
import networkx as nx
from typing import List, Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class QueryTranslator:
    """
    Translates natural language questions to NetworkX graph queries.
    """
    
    def __init__(self):
        """Initialize the query translator with example patterns."""
        self.examples = [
            {
                "question": "What functions call the function_a?",
                "query": "list(G.predecessors('function_a'))",
                "description": "Find all callers of a specific function"
            },
            {
                "question": "Show me all methods of ClassX",
                "query": "[n for n in G.successors('ClassX') if G.nodes[n]['type'] == 'function']",
                "description": "Find all methods of a specific class"
            },
            {
                "question": "What functions are defined in the auth module?",
                "query": "[n for n in G.successors('module.auth') if G.nodes[n]['type'] == 'function']",
                "description": "Find all functions in a specific module"
            },
            {
                "question": "Which classes inherit from BaseModel?",
                "query": "[n for n in G.predecessors('BaseModel') if G.nodes[n]['type'] == 'class']",
                "description": "Find all child classes of a specific parent class"
            },
            {
                "question": "What does the authenticate_user function call?",
                "query": "list(G.successors('auth.authenticate_user'))",
                "description": "Find all functions called by a specific function"
            },
            {
                "question": "Find all classes in the payment module",
                "query": "[n for n in G.successors('module.payment') if G.nodes[n]['type'] == 'class']",
                "description": "Find all classes in a specific module"
            },
            {
                "question": "What modules are imported by the main module?",
                "query": "[n for n in G.successors('module.main') if G.nodes[n]['type'] == 'module']",
                "description": "Find all modules imported by a specific module"
            },
            {
                "question": "Show me the inheritance hierarchy of User class",
                "query": "list(nx.descendants(G, 'auth.User'))",
                "description": "Find all descendants (inheritance hierarchy) of a class"
            },
            {
                "question": "What is the call path from function_a to function_c?",
                "query": "list(nx.all_simple_paths(G, 'function_a', 'function_c'))",
                "description": "Find all paths between two functions"
            },
            {
                "question": "Find all functions that take a user_id parameter",
                "query": "[n for n in G.nodes() if G.nodes[n]['type'] == 'function' and 'user_id' in G.nodes[n]['parameters']]",
                "description": "Find functions with specific parameter"
            }
        ]
    
    def extract_entity_names(self, question: str) -> List[str]:
        """
        Extract potential entity names (functions, classes, modules) from the question.
        
        Args:
            question: Natural language question
            
        Returns:
            List[str]: List of potential entity names
        """
        entities = []
        
        # Look for quoted names
        quoted_matches = re.findall(r'["\']([^"\']+)["\']', question)
        entities.extend(quoted_matches)
        
        # Look for function-like patterns (e.g., function_name, method_name)
        function_patterns = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)\b(?:\s+function|method|class)?', question)
        entities.extend(function_patterns)
        
        # Look for class patterns (usually capitalized)
        class_patterns = re.findall(r'\b([A-Z][a-zA-Z0-9_]*)\b', question)
        entities.extend(class_patterns)
        
        # Look for module patterns
        module_patterns = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s+module\b', question)
        entities.extend([f"module.{module}" for module in module_patterns])
        
        # Remove duplicates and common words
        common_words = {'what', 'show', 'find', 'all', 'the', 'in', 'by', 'from', 'to', 'of', 'me', 'does', 'do', 'are', 'is'}
        entities = [e for e in entities if e.lower() not in common_words and len(e) > 1]
        
        return list(set(entities))
    
    def identify_query_type(self, question: str) -> str:
        """
        Identify the type of query based on question patterns.
        
        Args:
            question: Natural language question
            
        Returns:
            str: Query type (callers, methods, functions_in_module, etc.)
        """
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['call', 'calls', 'calling']):
            if 'by' in question_lower or 'what does' in question_lower:
                return 'called_by_function'
            else:
                return 'callers_of_function'
        
        elif any(word in question_lower for word in ['method', 'methods']):
            return 'methods_of_class'
        
        elif any(word in question_lower for word in ['function', 'functions']):
            if 'module' in question_lower:
                return 'functions_in_module'
            elif 'parameter' in question_lower:
                return 'functions_with_parameter'
        
        elif any(word in question_lower for word in ['class', 'classes']):
            if 'module' in question_lower:
                return 'classes_in_module'
            elif 'inherit' in question_lower:
                return 'child_classes_of'
            elif 'hierarchy' in question_lower:
                return 'inheritance_hierarchy'
        
        elif any(word in question_lower for word in ['import', 'imports', 'imported']):
            return 'imports_of_module'
        
        elif any(word in question_lower for word in ['path', 'paths', 'between']):
            return 'path_between'
        
        return 'unknown'
    
    def generate_query_for_type(self, query_type: str, entities: List[str], graph: nx.DiGraph) -> Optional[str]:
        """
        Generate a NetworkX query based on query type and entities.
        
        Args:
            query_type: Type of query
            entities: List of entity names
            graph: NetworkX graph
            
        Returns:
            Optional[str]: Generated query string
        """
        if not entities:
            return None
        
        primary_entity = entities[0]
        
        # Try to find the actual node ID in the graph
        node_id = self.find_node_id(primary_entity, graph)
        if node_id:
            primary_entity = node_id
        
        query_templates = {
            'callers_of_function': f"list(G.predecessors('{primary_entity}'))",
            'called_by_function': f"list(G.successors('{primary_entity}'))",
            'methods_of_class': f"[n for n in G.successors('{primary_entity}') if G.nodes[n]['type'] == 'function']",
            'functions_in_module': f"[n for n in G.successors('{primary_entity}') if G.nodes[n]['type'] == 'function']",
            'classes_in_module': f"[n for n in G.successors('{primary_entity}') if G.nodes[n]['type'] == 'class']",
            'child_classes_of': f"[n for n in G.predecessors('{primary_entity}') if G.nodes[n]['type'] == 'class']",
            'inheritance_hierarchy': f"list(nx.descendants(G, '{primary_entity}'))",
            'imports_of_module': f"[n for n in G.successors('{primary_entity}') if G.nodes[n]['type'] == 'module']",
            'functions_with_parameter': f"[n for n in G.nodes() if G.nodes[n]['type'] == 'function' and '{entities[1] if len(entities) > 1 else ''}' in G.nodes[n]['parameters']]",
            'path_between': f"list(nx.all_simple_paths(G, '{primary_entity}', '{entities[1] if len(entities) > 1 else ''}'))"
        }
        
        return query_templates.get(query_type)
    
    def find_node_id(self, entity_name: str, graph: nx.DiGraph) -> Optional[str]:
        """
        Find the actual node ID for an entity name in the graph.
        
        Args:
            entity_name: Entity name from the question
            graph: NetworkX graph
            
        Returns:
            Optional[str]: Actual node ID if found
        """
        # Direct match
        if entity_name in graph.nodes:
            return entity_name
        
        # Try with module prefix
        for node_id in graph.nodes:
            node_data = graph.nodes[node_id]
            if node_data['name'] == entity_name:
                return node_id
        
        # Try partial match
        for node_id in graph.nodes:
            if entity_name in node_id:
                return node_id
        
        return None
    
    def translate_natural_language_to_query(self, question: str, graph: nx.DiGraph) -> str:
        """
        Translate a natural language question to a NetworkX graph query.
        
        Args:
            question: Natural language question
            graph: NetworkX graph
            
        Returns:
            str: NetworkX query string
        """
        logger.info(f"Translating question: {question}")
        
        # Extract entities from the question
        entities = self.extract_entity_names(question)
        logger.info(f"Extracted entities: {entities}")
        
        # Identify query type
        query_type = self.identify_query_type(question)
        logger.info(f"Identified query type: {query_type}")
        
        # Generate query based on type and entities
        query = self.generate_query_for_type(query_type, entities, graph)
        
        if query:
            logger.info(f"Generated query: {query}")
            return query
        
        # If no specific pattern matched, try to find the best matching example
        best_match = self.find_best_matching_example(question)
        if best_match:
            # Adapt the example query with the extracted entities
            adapted_query = self.adapt_example_query(best_match, entities, graph)
            if adapted_query:
                logger.info(f"Adapted query from example: {adapted_query}")
                return adapted_query
        
        # Fallback: return a simple node search
        if entities:
            entity = entities[0]
            node_id = self.find_node_id(entity, graph)
            if node_id:
                return f"[n for n in G.nodes() if n == '{node_id}']"
        
        logger.warning("Could not generate query for question")
        return "[]  # Could not parse question"
    
    def find_best_matching_example(self, question: str) -> Optional[Dict[str, str]]:
        """
        Find the best matching example for the given question.
        
        Args:
            question: Natural language question
            
        Returns:
            Optional[Dict]: Best matching example
        """
        question_lower = question.lower()
        best_match = None
        best_score = 0
        
        for example in self.examples:
            example_lower = example['question'].lower()
            
            # Simple keyword matching score
            score = 0
            question_words = set(question_lower.split())
            example_words = set(example_lower.split())
            
            common_words = question_words & example_words
            score = len(common_words)
            
            if score > best_score:
                best_score = score
                best_match = example
        
        return best_match if best_score > 0 else None
    
    def adapt_example_query(self, example: Dict[str, str], entities: List[str], graph: nx.DiGraph) -> Optional[str]:
        """
        Adapt an example query with the extracted entities.
        
        Args:
            example: Example query with template
            entities: Extracted entities
            graph: NetworkX graph
            
        Returns:
            Optional[str]: Adapted query
        """
        if not entities:
            return None
        
        query = example['query']
        
        # Replace placeholder entity names with actual entities
        # This is a simple adaptation - in practice, you'd want more sophisticated logic
        
        # Look for common patterns in the example query
        if 'function_a' in query and entities:
            node_id = self.find_node_id(entities[0], graph)
            if node_id:
                query = query.replace('function_a', node_id)
        
        if 'ClassX' in query and entities:
            node_id = self.find_node_id(entities[0], graph)
            if node_id:
                query = query.replace('ClassX', node_id)
        
        if 'BaseModel' in query and entities:
            node_id = self.find_node_id(entities[0], graph)
            if node_id:
                query = query.replace('BaseModel', node_id)
        
        if 'module.auth' in query and entities:
            if entities[0].startswith('module.'):
                query = query.replace('module.auth', entities[0])
            else:
                query = query.replace('module.auth', f'module.{entities[0]}')
        
        if 'authenticate_user' in query and entities:
            node_id = self.find_node_id(entities[0], graph)
            if node_id:
                query = query.replace('authenticate_user', node_id.split('.')[-1])
        
        return query


def translate_natural_language_to_query(question: str, G: nx.DiGraph) -> str:
    """
    Convenience function to translate natural language to graph query.
    
    Args:
        question: Natural language question
        G: NetworkX graph
        
    Returns:
        str: NetworkX query string
    """
    translator = QueryTranslator()
    return translator.translate_natural_language_to_query(question, G)


if __name__ == "__main__":
    # Test the query translator
    import networkx as nx
    
    # Create a simple test graph
    G = nx.DiGraph()
    G.add_node('module.auth', type='module', name='auth')
    G.add_node('auth.authenticate_user', type='function', name='authenticate_user', parameters=['username', 'password'])
    G.add_node('auth.User', type='class', name='User')
    G.add_node('auth.login', type='function', name='login', parameters=['user_id'])
    G.add_edge('auth.authenticate_user', 'module.auth', type='DEFINED_IN')
    G.add_edge('auth.User', 'module.auth', type='DEFINED_IN')
    G.add_edge('auth.login', 'module.auth', type='DEFINED_IN')
    G.add_edge('auth.login', 'auth.authenticate_user', type='CALLS')
    G.add_edge('auth.User', 'auth.authenticate_user', type='HAS_METHOD')
    
    translator = QueryTranslator()
    
    test_questions = [
        "What functions call the authenticate_user function?",
        "Show me all methods of User class",
        "What functions are defined in the auth module?",
        "Find functions that take a user_id parameter"
    ]
    
    for question in test_questions:
        print(f"\nQuestion: {question}")
        query = translator.translate_natural_language_to_query(question, G)
        print(f"Query: {query}")
        
        try:
            result = eval(query)
            print(f"Result: {result}")
        except Exception as e:
            print(f"Error executing query: {e}")