# Agentic Code Solver for SciCode

This project implements an AI agent that autonomously solves programming problems from the Hugging Face `SciCode1/SciCode` dataset. The agent operates in an "agentic" way, meaning it uses a set of tools to write, read, and execute code in a dedicated workspace, mimicking a human developer's workflow.

## Core Logic

The agent is built using the LangChain framework with an OpenAI model (`gpt-5`) as its reasoning engine. It follows a ReAct (Reasoning and Acting) prompting strategy to decide which tool to use at each step to solve the problem.

The agent's primary goal is to break down a complex coding problem from the SciCode dataset into its constituent sub-steps, solve each sub-step sequentially, and validate its code using tests.

### Key Features:
- **Agentic Workflow**: The agent doesn't just generate code in one go. It thinks step-by-step, writes code to files, creates tests, and runs them to verify correctness.
- **Tool-Based Operation**: The agent has access to a virtual toolbox, allowing it to interact with a file system. The core tools include:
    - `write_file`: To create or overwrite files (e.g., `solution.py`, `test_solution.py`).
    - `read_file`: To review its own code or other relevant files.
    - `run_python_script`: To execute its generated Python scripts and check for errors or correct outputs.
    - `list_directory`: To see the contents of its workspace.
- **SciCode Dataset Integration**: The main script directly loads a problem from the `SciCode1/SciCode` dataset on Hugging Face to provide the agent with its task.

## How to Use

1.  **Set up Environment**:
    - Make sure you have Python and `uv` installed.
    - Install the required dependencies:
      ```bash
      uv pip install -r requirements.txt 
      # Or use `uv add` for each dependency in pyproject.toml
      ```
    - Create a `.env` file in the root directory and add your OpenAI API key:
      ```
      OPENAI_API_KEY="sk-..."
      ```

2.  **Run the Agent**:
    Execute the main script from the root directory. You can specify which problem from the dataset the agent should solve using the `--problem_id` argument.

    ```bash
    # Run the agent on the first problem (default)
    python src/main.py

    # Run the agent on the 10th problem in the dataset
    python src/main.py --problem_id 10
    ```
    The agent will start its process, and you will see its thoughts, actions, and tool outputs printed to the console in real-time.

3.  **View the Output**:
    The agent will perform all its file operations inside the `agent_workspace/` directory. You can inspect the files it creates (both the solution and test scripts) to see its work. The final answer and a summary of the workspace will be printed at the end of the run.

## File Structure

- `src/main.py`: The main entry point. It loads the dataset, initializes the agent, and starts the problem-solving process.
- `src/tools.py`: Defines the file system and code execution tools available to the agent.
- `agent_workspace/`: A dedicated directory where the agent writes and executes its code. This acts as the agent's sandboxed development environment.
- `pyproject.toml`: Project dependencies and configuration.
- `.env.example`: An example file for setting up the environment variables.
