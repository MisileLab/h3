"""
Pytest configuration for quantumdb tests.

This module ensures that the correct Python path is set up for testing
in the Nix environment where the virtual environment might not work
as expected.
"""

import sys
import os

# Add the virtual environment site-packages to the Python path
venv_site_packages = (
    "/home/misile/repos/h3/projects/quantumdb/.venv/lib/python3.13/site-packages"
)
if venv_site_packages not in sys.path:
    sys.path.insert(0, venv_site_packages)

# Add the project root to the Python path
project_root = "/home/misile/repos/h3/projects/quantumdb"
if project_root not in sys.path:
    sys.path.insert(0, project_root)
