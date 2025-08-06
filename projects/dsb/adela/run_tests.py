"""Script to run tests using uv."""

import os
import sys
import subprocess
from pathlib import Path


def main():
    """Run tests using uv."""
    # Create tests directory if it doesn't exist
    Path("tests").mkdir(exist_ok=True)
    
    # Run tests
    print("Running tests...")
    result = subprocess.run(["uv", "run", "pytest", "-xvs", "tests"])
    
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
