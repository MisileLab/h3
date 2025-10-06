"""
Demo script showing the complete code knowledge graph workflow.

This script demonstrates how to use the code knowledge graph system
to parse code, build a graph, and answer questions about the codebase.
"""

import sys
import os

# Add the parent directory to the path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Add parent directory to path for imports
import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from main_pipeline import ask_question_about_code, analyze_project_structure


def print_separator(title: str):
    """Print a formatted separator."""
    print("\n" + "=" * 60)
    print(f" {title} ")
    print("=" * 60)


def demo_project_analysis():
    """Demonstrate project structure analysis."""
    print_separator("PROJECT STRUCTURE ANALYSIS")
    
    # Analyze the demo project structure
    structure = analyze_project_structure("demo")
    
    print("Project Overview:")
    print(f"  Total nodes: {structure.get('total_nodes', 0)}")
    print(f"  Total edges: {structure.get('total_edges', 0)}")
    
    print("\nNode Types:")
    for node_type, count in structure.get('node_counts', {}).items():
        print(f"  {node_type}: {count}")
    
    print("\nEdge Types:")
    for edge_type, count in structure.get('edge_counts', {}).items():
        print(f"  {edge_type}: {count}")
    
    print("\nModules found:")
    for module in structure.get('modules', []):
        print(f"  - {module}")
    
    print("\nClasses found:")
    for cls in structure.get('classes', []):
        print(f"  - {cls}")
    
    print("\nFunctions found:")
    for func in structure.get('functions', []):
        print(f"  - {func}")
    
    print("\nMost connected nodes:")
    for node_id, degree in structure.get('most_connected_nodes', []):
        print(f"  {node_id}: {degree} connections")


def demo_question_answering():
    """Demonstrate question answering capabilities."""
    print_separator("QUESTION ANSWERING DEMO")
    
    # Sample questions to demonstrate the system
    questions = [
        "What functions are defined in the auth module?",
        "Show me all methods of the User class",
        "What functions call the authenticate_user function?",
        "What does the process_payment_for_user function call?",
        "Find all classes in the payment module",
        "What modules are imported by the payment module?",
        "Show me functions that take a username parameter",
        "What classes inherit from User class?",
        "Find all functions related to payment processing",
        "What is the relationship between AuthManager and User?"
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n--- Question {i} ---")
        print(f"Q: {question}")
        print("A: ", end="")
        
        try:
            answer = ask_question_about_code("demo", question)
            print(answer)
        except Exception as e:
            print(f"Error: {e}")
        
        print("-" * 40)


def demo_specific_analysis():
    """Demonstrate specific code analysis scenarios."""
    print_separator("SPECIFIC ANALYSIS SCENARIOS")
    
    scenarios = [
        {
            "title": "Authentication Flow Analysis",
            "questions": [
                "What functions are involved in user authentication?",
                "How does the authenticate_user function work?",
                "What classes are related to user management?"
            ]
        },
        {
            "title": "Payment Processing Flow",
            "questions": [
                "What functions are involved in payment processing?",
                "How does the process_payment_for_user function work?",
                "What payment providers are available?"
            ]
        },
        {
            "title": "Cross-Module Dependencies",
            "questions": [
                "How does the payment module depend on the auth module?",
                "What functions from auth are used in payment?",
                "Show me the relationship between User and Transaction classes"
            ]
        }
    ]
    
    for scenario in scenarios:
        print(f"\n### {scenario['title']} ###")
        for question in scenario['questions']:
            print(f"\nQ: {question}")
            try:
                answer = ask_question_about_code("demo", question)
                print(f"A: {answer[:200]}..." if len(answer) > 200 else f"A: {answer}")
            except Exception as e:
                print(f"A: Error - {e}")


def demo_error_handling():
    """Demonstrate error handling with edge cases."""
    print_separator("ERROR HANDLING DEMO")
    
    edge_case_questions = [
        "What functions call a non_existent_function?",
        "Show me methods of NonExistentClass",
        "What modules are imported by non_existent_module?",
        "",  # Empty question
        "Find functions with parameter xyz123"  # Non-existent parameter
    ]
    
    for question in edge_case_questions:
        print(f"\nQ: '{question}'")
        try:
            answer = ask_question_about_code("demo", question)
            print(f"A: {answer}")
        except Exception as e:
            print(f"A: Caught exception - {e}")


def main():
    """Main demo function."""
    print("🚀 Code Knowledge Graph Demo")
    print("This demo showcases the complete workflow of the code knowledge graph system.")
    
    try:
        # Demo 1: Project structure analysis
        demo_project_analysis()
        
        # Demo 2: Question answering
        demo_question_answering()
        
        # Demo 3: Specific analysis scenarios
        demo_specific_analysis()
        
        # Demo 4: Error handling
        demo_error_handling()
        
        print_separator("DEMO COMPLETED")
        print("✅ All demos completed successfully!")
        print("\nKey takeaways:")
        print("1. The system can parse Python code and extract structural information")
        print("2. It builds a knowledge graph representing code relationships")
        print("3. Natural language questions are translated to graph queries")
        print("4. The system provides meaningful answers about code structure")
        print("5. Error handling ensures robustness with edge cases")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()