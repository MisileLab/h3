import os
import subprocess
from typing import List

def write_file(file_path: str, content: str) -> str:
    """
    Writes content to a file. Creates the directory if it doesn't exist.
    Args:
        file_path (str): The path to the file.
        content (str): The content to write to the file.
    Returns:
        str: A confirmation message.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"Successfully wrote to {file_path}"
    except Exception as e:
        return f"Error writing to file: {e}"

def read_file(file_path: str) -> str:
    """
    Reads the content of a file.
    Args:
        file_path (str): The path to the file.
    Returns:
        str: The content of the file or an error message.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {e}"

def run_python_script(script_path: str) -> str:
    """
    Runs a Python script and captures its output.
    Args:
        script_path (str): The path to the Python script.
    Returns:
        str: The stdout and stderr from the script execution.
    """
    try:
        result = subprocess.run(
            ['python', script_path],
            capture_output=True,
            text=True,
            timeout=30  # 30-second timeout to prevent long-running scripts
        )
        output = f"--- STDOUT ---\n{result.stdout}\n"
        if result.stderr:
            output += f"--- STDERR ---\n{result.stderr}\n"
        return output
    except FileNotFoundError:
        return f"Error: Script not found at {script_path}"
    except subprocess.TimeoutExpired:
        return "Error: Script execution timed out after 30 seconds."
    except Exception as e:
        return f"An error occurred while running the script: {e}"

def list_directory(path: str = '.') -> str:
    """
    Lists the files and directories in a given path.
    Args:
        path (str): The path to the directory. Defaults to the current directory.
    Returns:
        str: A string listing the contents of the directory or an error message.
    """
    try:
        entries = os.listdir(path)
        if not entries:
            return "The directory is empty."
        return "\n".join(entries)
    except FileNotFoundError:
        return f"Error: Directory not found at {path}"
    except Exception as e:
        return f"An error occurred: {e}"

def mark_as_done() -> str:
    """
    Signals that the agent has finished its work and is ready for the solution to be tested.
    Returns:
        str: A confirmation message.
    """
    return "Agent has marked the task as done. Proceeding to test."

# It's good practice to group tools into a list for the agent
coding_tools = [write_file, read_file, run_python_script, list_directory, mark_as_done]
