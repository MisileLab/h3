
import json
import os
from src.agent import run_agent

def run_scenarios(scenario_file: str):
    with open(scenario_file, 'r', encoding='utf-8') as f:
        scenarios = json.load(f)
    
    for scenario in scenarios:
        query = scenario['query']
        print(f"--- Running scenario: {query} ---")
        run_agent(query)
        print("--- Scenario complete ---\n")

if __name__ == "__main__":
    # For now, just run the research tasks
    run_scenarios("scenarios/research_tasks.json")
    run_scenarios("scenarios/multi_step_reasoning.json")
    run_scenarios("scenarios/complex_tasks.json")
