
import logging
from typing import List, Tuple, Any, Dict
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Online LLM Judge Optimizer ---

LLM_JUDGE_PROMPT_ONLINE = """
You are an expert in analyzing AI agent execution traces. Your task is to review the agent's scratchpad (intermediate steps) and identify only the steps that are ESSENTIAL for deciding the NEXT action.
A step is NOT essential if it was an error, a dead end, redundant, or provides no useful information for the immediate next step.

**The User's Goal:**
{query}

**Agent's Scratchpad (Previous Steps):**
{steps_json}

**Your Task:**
Review the scratchpad and return a JSON object containing a list of indices for the steps that are absolutely essential for the agent to consider for its next reasoning cycle. The last step is almost always essential.

Output your response as a JSON object with a single key "essential_step_indices" containing a list of the integer indices to keep.

Example Input:
[
  {"action": "search", "action_input": "python list length", "observation": "len() function"},
  {"action": "search", "action_input": "how to open file python", "observation": "with open(...) as f:"},
  {"action": "write_file", "action_input": "script.py, print(len([1,2]))", "observation": "File written successfully"},
  {"action": "run_script", "action_input": "script.py", "observation": "Error: script has syntax error"}
]

Example Response:
{{
  "essential_step_indices": [1, 2]
}}
"""

def optimize_steps_with_llm_judge(
    steps: List[Tuple[AgentAction, Any]],
    query: str
) -> List[Tuple[AgentAction, Any]]:
    """
    Uses gpt-5-nano as a "judge" to decide which intermediate steps are essential for the next reasoning cycle.
    """
    if not steps:
        return []

    logging.info("Optimizing steps with LLM judge (gpt-5-nano)...")

    try:
        judge_llm = ChatOpenAI(model="gpt-5-nano", temperature=0).bind(stop=None)
        prompt = ChatPromptTemplate.from_template(LLM_JUDGE_PROMPT_ONLINE)
        parser = JsonOutputParser()
        chain = prompt | judge_llm | parser
        
        # Format steps for the prompt
        steps_json = [
            {"action": action.tool, "action_input": action.tool_input, "observation": str(obs)}
            for action, obs in steps
        ]

        response = chain.invoke({
            "query": query,
            "steps_json": str(steps_json)
        })
        
        essential_indices = set(response.get("essential_step_indices", []))
        
        if not essential_indices:
            logging.warning("LLM judge did not return any essential steps. Returning original steps.")
            return steps
            
        essential_steps = [steps[i] for i in essential_indices if i < len(steps)]
        
        logging.info(f"LLM judge reduced {len(steps)} steps to {len(essential_steps)} essential steps.")
        
        return essential_steps

    except Exception as e:
        logging.error(f"An error occurred during LLM judge optimization: {e}")
        return steps

# --- Online Rule-Based Optimizer ---

def is_bad_step(step: Tuple[AgentAction, Any]) -> bool:
    """
    Determines if a single agent step is "bad" based on simple rules.
    A "bad" step is one that resulted in an error or produced no useful output.
    """
    action, observation = step
    obs_str = str(observation).lower()

    # Rule 1: Observation contains error messages
    error_keywords = ["error", "failed", "not found", "exception"]
    if any(keyword in obs_str for keyword in error_keywords):
        return True

    # Rule 2: Observation is empty or indicates no result
    if not obs_str.strip():
        return True
        
    return False

def optimize_steps_with_rules(
    steps: List[Tuple[AgentAction, Any]]
) -> List[Tuple[AgentAction, Any]]:
    """
    Optimizes the list of intermediate steps by removing "bad" steps based on rules.
    """
    if not steps:
        return []
    
    logging.info("Optimizing steps with rule-based filter...")
    
    # We keep the last step regardless, as it's the most recent context.
    # We only filter the steps before the last one.
    last_step = steps[-1]
    previous_steps = steps[:-1]
    
    essential_steps = [step for step in previous_steps if not is_bad_step(step)]
    essential_steps.append(last_step)
    
    if len(essential_steps) < len(steps):
        logging.info(f"Rule-based optimizer reduced {len(steps)} steps to {len(essential_steps)}.")
        
    return essential_steps
