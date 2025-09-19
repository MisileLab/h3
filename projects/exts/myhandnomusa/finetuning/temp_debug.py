import sys
import os

# Add the parent directory to the Python path to find deploy.py
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))

import deploy

if __name__ == "__main__":
    # Simulate command-line arguments for deploy.py
    sys.argv = ["deploy.py", "-y"]
    deploy.main()