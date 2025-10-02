
import os
import argparse
import subprocess
import time
import json
from dotenv import load_dotenv
from datasets import load_dataset
from langchain_openai import ChatOpenAI
from langchain.agents import Tool, create_react_agent
from langchain_core.agents import AgentAction, AgentFinish
from langchain import hub
from langchain_community.callbacks import get_openai_callback
from src.tools import coding_tools
from src.optimizer import optimize_steps_with_llm_judge
from typing import List, Tuple, Any, Union

# Load environment variables
load_dotenv()

# --- Constants ---
AGENT_WORKSPACE = "agent_workspace"
SOLUTION_FILENAME = "solution.py"
TEST_RUNNER_FILENAME = "test_runner.py"
MAX_ITERATIONS = 25

# --- Helper Functions ---
def print_section_header(title):
    print("\n" + "="*80)
    print(f"# {title}")
    print("="*80)

def setup_workspace():
    if not os.path.exists(AGENT_WORKSPACE):
        os.makedirs(AGENT_WORKSPACE)
    for f in os.listdir(AGENT_WORKSPACE):
        os.remove(os.path.join(AGENT_WORKSPACE, f))

def create_agent_prompt(problem: dict) -> str:
    problem_description = problem['problem_description_main']
    
    prompt = (
        "You are an expert Python programmer. Your task is to solve a coding problem by writing a single Python script.\n"
        "You have access to the following tools: write_file, read_file, run_python_script, list_directory, run_tests, mark_as_done.\n\n"
        "write_file(filename: str, content: str) -> str: Writes content to a file in the agent_workspace directory.\n"
        "read_file(filename: str) -> str: Reads and returns the content of a file from the agent_workspace directory.\n"
        "run_python_script(filename: str) -> str: Executes a Python script from the agent_workspace directory and returns its output.\n"
        "list_directory(path: str) -> str: Lists files in the specified directory (relative to agent_workspace).\n"
        "run_tests() -> str: Runs the test suite against the code in agent_workspace/solution.py. Returns PASS or FAIL with details.\n"
        "mark_as_done() -> str: Signals that you have finished your work and are ready for the solution to be tested.\n\n"
        "IMPORTANT: When providing JSON for tool inputs, you must provide ONLY the raw JSON object, without any surrounding markdown formatting like ```json or ```.\n\n"
        "Follow these instructions carefully:\n"
        f"1. Create a single Python script named `agent_workspace/{SOLUTION_FILENAME}`.\n"
        "2. Implement the solution for the problem in this single script.\n"
        "3. Ensure the script includes all necessary imports (like numpy, math, etc.).\n"
        "4. After writing the code, use the `run_tests` tool to check your solution.\n"
        "5. If the tests fail, read the error, modify the code in `agent_workspace/{SOLUTION_FILENAME}`, and run tests again.\n"
        "6. Once the tests pass, use the `mark_as_done` tool to indicate you are finished.\n\n"
        "--- MAIN PROBLEM ---\n"
        f"{problem_description}\n\n"
        f"\nBegin by creating the `agent_workspace/{SOLUTION_FILENAME}` file and implementing the solution."
    )
    return prompt

def run_llm_test(problem: dict) -> str:
    """Runs the test suite against the code in solution.py and returns a detailed string output."""
    solution_path = os.path.join(AGENT_WORKSPACE, SOLUTION_FILENAME)
    if not os.path.exists(solution_path):
        return "FAIL: No solution file found at {}".format(solution_path)

    solution_code = open(solution_path, 'r').read()
    
    test_code = problem.get('required_dependencies', '') + "\n\n"
    test_code += solution_code
    test_code += "\n\n# --- Running Tests ---\n"
    
    all_tests_str = []
    for step in problem['sub_steps']:
        all_tests_str.extend(step['test_cases'])

    if not all_tests_str:
        return "SKIP: No test cases found for this problem."

    test_runner_path = os.path.join(AGENT_WORKSPACE, TEST_RUNNER_FILENAME)
    with open(test_runner_path, 'w') as f:
        f.write(test_code)

    try:
        result = subprocess.run(
            ['direnv', 'exec', '.', 'python', test_runner_path],
            capture_output=True, text=True, timeout=60
        )
        if result.returncode == 0 and "All tests passed!" in result.stdout:
            return f"PASS\n{result.stdout}"
        else:
            output = f"FAIL\n"
            if result.stdout:
                output += f"--- STDOUT ---\n{result.stdout}\n"
            if result.stderr:
                output += f"--- STDERR ---\n{result.stderr}\n"
            return output
    except subprocess.TimeoutExpired:
        return "FAIL: Test execution timed out after 60 seconds."
    except Exception as e:
        return f"FAIL: An unexpected error occurred during test execution: {e}"

def run_tests_for_solution(problem: dict) -> bool:
    solution_path = os.path.join(AGENT_WORKSPACE, SOLUTION_FILENAME)
    if not os.path.exists(solution_path):
        print("  - Result: FAIL (No solution file found)")
        return False

    solution_code = open(solution_path, 'r').read()
    
    test_code = problem.get('required_dependencies', '') + "\n\n"
    test_code += solution_code
    test_code += "\n\n# --- Running Tests ---\n"
    
    all_tests_str = []
    for step in problem['sub_steps']:
        all_tests_str.extend(step['test_cases'])

    if not all_tests_str:
        print("  - Result: SKIP (No test cases found)")
        return True

    test_runner_path = os.path.join(AGENT_WORKSPACE, TEST_RUNNER_FILENAME)
    with open(test_runner_path, 'w') as f:
        f.write(test_code)

    try:
        result = subprocess.run(
            ['direnv', 'exec', '.', 'python', test_runner_path],
            capture_output=True, text=True, timeout=60
        )
        if result.returncode == 0 and "All tests passed!" in result.stdout:
            print("  - Result: PASS")
            return True
        else:
            print(f"  - Result: FAIL (Tests failed or script crashed)")
            return False
    except subprocess.TimeoutExpired:
        print("  - Result: FAIL (Test execution timed out)")
        return False
    except Exception as e:
        print(f"  - Result: FAIL (Error during test execution: {e})")
        return False

# --- Custom Agent Execution Loop ---
def format_steps_to_scratchpad(intermediate_steps: List[Tuple[AgentAction, str]]) -> str:
    """Formats intermediate steps into a string for the ReAct prompt."""
    log = ""
    for action, observation in intermediate_steps:
        log += action.log
        log += f"\nObservation: {observation}\n"
    return log

def run_agent_custom_loop(
    agent_runnable,
    tools: List[Tool],
    problem_prompt: str,
    optimizer_func=None
) -> dict:
    """
    Runs the agent using a custom loop to allow for online optimization.
    """
    intermediate_steps: List[Tuple[AgentAction, Any]] = []
    
    for i in range(MAX_ITERATIONS):
        print(f"\n--- Iteration {i+1}/{MAX_ITERATIONS} ---")

        if optimizer_func and intermediate_steps:
            if optimizer_func.__name__ == "optimize_steps_with_llm_judge":
                last_action_log = intermediate_steps[-1][0].log
                optimized_steps = optimizer_func(
                    intermediate_steps,
                    query=problem_prompt
                )
            else:
                optimized_steps = optimizer_func(intermediate_steps)
        else:
            optimized_steps = intermediate_steps

        try:
            agent_scratchpad = format_steps_to_scratchpad(optimized_steps)
            agent_input = {
                "input": problem_prompt,
                "agent_scratchpad": agent_scratchpad,
                "intermediate_steps": optimized_steps  # Add the raw steps as well
            }
            output = agent_runnable.invoke(agent_input)
        except Exception as e:
            print(f"Error during agent planning: {e}")
            return {"output": "Agent planning failed."}

        if isinstance(output, AgentFinish):
            print(f"Agent finished with output:\n{output.return_values.get('output')}")
            return output.return_values

        if output.tool == "mark_as_done":
            print("Agent marked task as done. Exiting loop to run tests.")
            break

        print(f"Action: {output.tool}, Input: {output.tool_input}")
        try:
            tool = next(t for t in tools if t.name == output.tool)
            tool_input = output.tool_input
            if isinstance(tool_input, str):
                try:
                    tool_input = json.loads(tool_input)
                except json.JSONDecodeError:
                    # Not a JSON string, pass it as is.
                    pass
            
            if isinstance(tool_input, dict):
                tool_output = tool.func(**tool_input)
            else:
                tool_output = tool.func(tool_input)

            print(f"Observation: {tool_output}")
        except Exception as e:
            print(f"Error executing tool {output.tool}: {e}")
            tool_output = f"Error: {e}"

        intermediate_steps.append((output, tool_output))

    return {"output": "Agent reached maximum iterations."}


def main():
    parser = argparse.ArgumentParser(description="Evaluate a code-generating agent on the SciCode dataset with different optimizers.")
    parser.add_argument("--limit", type=int, default=3, help="The number of problems to evaluate.")
    args = parser.parse_args()

    print_section_header("Loading SciCode Dataset")
    try:
        dataset = load_dataset("SciCode1/SciCode", split='test', streaming=True)
        eval_dataset = list(dataset.take(args.limit))
        print(f"Loaded {len(eval_dataset)} problems for evaluation.")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    llm = ChatOpenAI(model="gpt-4o", temperature=0).bind(stop=None)
    prompt_template = hub.pull("hwchase17/react")
    base_tools = [Tool(name=t.__name__, func=t, description=t.__doc__) for t in coding_tools]

    optimizer_strategies = {
        "No Optimization": None,
        "LLM Judge Optimizer": optimize_steps_with_llm_judge
    }
    
    final_results = {}

    for name, optimizer_func in optimizer_strategies.items():
        print_section_header(f"Running Evaluation with: {name}")
        
        results = []
        with get_openai_callback() as cb:
            for i, problem in enumerate(eval_dataset):
                problem_name = problem['problem_name']
                print(f"\n--- Solving Problem {i+1}/{args.limit}: {problem_name} ---")
                
                setup_workspace()

                # Create a dynamic tool for running tests for the current problem
                llm_test_tool = Tool(
                    name="run_tests",
                    func=lambda: run_llm_test(problem),
                    description="Runs the test suite against the code in agent_workspace/solution.py. Returns PASS or FAIL with details."
                )
                current_tools = base_tools + [llm_test_tool]

                # Re-create agent with the dynamic tool for this problem
                agent_runnable = create_react_agent(llm, current_tools, prompt_template)
                
                agent_prompt = create_agent_prompt(problem)
                
                run_agent_custom_loop(agent_runnable, current_tools, agent_prompt, optimizer_func)
                
                success = run_tests_for_solution(problem)
                if success:
                    print("Problem solved successfully.")
                else:
                    print("Problem failed.")

                results.append(success)

            pass_rate = sum(results) / len(results) if results else 0.0
            final_results[name] = {
                "pass_rate": pass_rate,
                "total_tokens": cb.total_tokens,
                "prompt_tokens": cb.prompt_tokens,
                "completion_tokens": cb.completion_tokens,
                "total_cost": cb.total_cost,
            }

    print_section_header("Final Comparative Results")
    print(f"Evaluated on {len(eval_dataset)} problems.")
    print("-" * 80)
    for name, stats in final_results.items():
        print(f"{name:<25} | Pass Rate: {stats['pass_rate']:.2%}")
        print(f"  - Total Tokens: {stats['total_tokens']}")
        print(f"  - Prompt Tokens: {stats['prompt_tokens']}")
        print(f"  - Completion Tokens: {stats['completion_tokens']}")
        print(f"  - Total Cost (USD): ${stats['total_cost']:.6f}")
    print("-" * 80)


if __name__ == "__main__":
    main()

