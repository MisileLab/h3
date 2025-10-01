
import os
import json
import uuid
import copy
from datetime import datetime, timezone
from typing import TypedDict, List, Any, Dict
from functools import wraps

from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

load_dotenv()

# --- LLM Initialization ---
# Using gpt-5 as the main model
llm = ChatOpenAI(model="gpt-5", temperature=0)

# --- State Definition ---
class AgentState(TypedDict):
    query: str
    messages: list # Changed from List[BaseMessage] for broader compatibility
    intermediate_steps: list
    final_answer: str
    trace_log: List[dict]
    session_id: str

# --- Tools ---
# (Placeholders for now)
def web_search(query: str):
    """Simulates a web search."""
    return f"Search results for: {query}"

def calculator(expression: str):
    """Simulates a calculator."""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"

# --- Trace Decorator ---
def trace_node(node_func):
    @wraps(node_func)
    def wrapper(state: AgentState) -> AgentState:
        node_name = node_func.__name__
        turn_id = len(state.get('trace_log', [])) + 1
        start_time = datetime.now(timezone.utc)
        
        # Deep copy inputs to prevent mutation
        input_data = {
            "query": state['query'], 
            "messages": copy.deepcopy(state['messages'])
        }

        # Execute the actual node function
        result_state = node_func(state)

        end_time = datetime.now(timezone.utc)
        latency_ms = (end_time - start_time).total_seconds() * 1000

        # Determine action type and output based on node
        action_type = "reasoning" # Default
        output_data = {}
        if node_name == "planning_node":
            action_type = "planning"
            output_data = {"result": result_state['messages'][-1], "success": True}
        elif node_name == "tool_execution_node":
            action_type = "tool_call"
            output_data = {"result": result_state['intermediate_steps'][-1], "success": True}
        elif node_name == "synthesis_node":
            action_type = "synthesis"
            output_data = {"result": result_state['final_answer'], "success": True}

        turn_trace = {
            "turn_id": turn_id,
            "node_name": node_name,
            "action_type": action_type,
            "input": input_data,
            "output": output_data,
            "tokens_used": 0,  # Placeholder
            "latency_ms": latency_ms,
            "timestamp": start_time.isoformat()
        }
        
        if 'trace_log' not in result_state:
            result_state['trace_log'] = []
        result_state['trace_log'].append(turn_trace)
        
        return result_state
    return wrapper

# --- Nodes ---
@trace_node
def planning_node(state: AgentState):
    """Decides the next action by calling the LLM."""
    print("---PLANNING---")
    response = llm.invoke(state['messages'])
    state['messages'].append(response)
    return state

@trace_node
def tool_execution_node(state: AgentState):
    """Executes the chosen tool (simulation)."""
    print("---TOOL EXECUTION---")
    # This remains a simulation for now
    result = web_search(state['query'])
    state['intermediate_steps'].append(("web_search", result))
    # Append a simulated tool output message for the next step
    state['messages'].append({"role": "tool", "content": result})
    return state

@trace_node
def synthesis_node(state: AgentState):
    """Synthesizes the final answer by calling the LLM."""
    print("---SYNTHESIS---")
    # Create a new prompt for synthesis
    synthesis_prompt = state['messages'] + [{
        "role": "user", 
        "content": "Based on the previous steps, provide a comprehensive final answer."
    }]
    response = llm.invoke(synthesis_prompt)
    state['final_answer'] = response.content
    state['messages'].append(response)
    return state

# --- Graph Definition ---
workflow = StateGraph(AgentState)

workflow.add_node("planning", planning_node)
workflow.add_node("tool_execution", tool_execution_node)
workflow.add_node("synthesis", synthesis_node)

workflow.set_entry_point("planning")

# A simple router function could be added here for more complex logic
workflow.add_edge("planning", "tool_execution")
workflow.add_edge("tool_execution", "synthesis")
workflow.add_edge("synthesis", END)

app = workflow.compile()

# --- Agent Executor ---
def save_trace(trace_data: Dict[str, Any]):
    session_id = trace_data["session_id"]
    traces_dir = "traces"
    if not os.path.exists(traces_dir):
        os.makedirs(traces_dir)
    filename = os.path.join(traces_dir, f"trace_{session_id}.json")
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(trace_data, f, ensure_ascii=False, indent=2)
    print(f"\nTrace saved to {filename}")

def run_agent(query: str):
    session_id = str(uuid.uuid4())
    
    initial_state = {
        "query": query,
        "messages": [{"role": "user", "content": query}],
        "intermediate_steps": [],
        "final_answer": "",
        "trace_log": [],
        "session_id": session_id,
    }
    
    start_time = datetime.now(timezone.utc)
    
    # Execute the graph
    final_state = app.invoke(initial_state)
    
    end_time = datetime.now(timezone.utc)
    total_time_ms = (end_time - start_time).total_seconds() * 1000

    trace_data = {
        "session_id": session_id,
        "query": query,
        "timestamp": start_time.isoformat(),
        "turns": final_state['trace_log'],
        "final_answer": final_state['final_answer'],
        "total_cost": 0.0, # Placeholder
        "total_time_ms": total_time_ms
    }

    save_trace(trace_data)
    
    print("\n--- FINAL ANSWER ---")
    print(final_state['final_answer'])
    
    return final_state

if __name__ == "__main__":
    run_agent("2024년 AI 분야 주요 발전 3가지를 찾아서 각각 요약해줘")
