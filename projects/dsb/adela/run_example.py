"""Script to run examples using uv."""

import os
import sys
import subprocess
from pathlib import Path


def main():
    """Run an example script using uv."""
    if len(sys.argv) < 2:
        print("Usage: uv run python run_example.py [example_name]")
        print("Available examples:")
        print("  simple_game - Run a simple game example")
        print("  train_model - Run the training example")
        return
    
    example_name = sys.argv[1]
    
    # Create examples directory if it doesn't exist
    Path("examples").mkdir(exist_ok=True)
    
    if example_name == "simple_game":
        print("Running simple game example...")
        subprocess.run(["uv", "run", "python", "examples/simple_game.py"])
    elif example_name == "train_model":
        print("Running training example...")
        subprocess.run(["uv", "run", "python", "examples/train_model.py"])
    else:
        print(f"Unknown example: {example_name}")
        print("Available examples:")
        print("  simple_game - Run a simple game example")
        print("  train_model - Run the training example")


if __name__ == "__main__":
    main()
