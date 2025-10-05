#!/usr/bin/env python3
"""
Simple test runner that bypasses pytest's collection mechanism
to work around the import issues in the Nix environment.
"""

import sys
import os
import unittest

# Set up Python path
venv_site_packages = (
    "/home/misile/repos/h3/projects/quantumdb/.venv/lib/python3.13/site-packages"
)
project_root = "/home/misile/repos/h3/projects/quantumdb"

sys.path.insert(0, venv_site_packages)
sys.path.insert(0, project_root)

print("=== Running tests with custom test runner ===")
print(f"Python path: {sys.path[:3]}")

# Import and run tests
try:
    # Import test modules directly
    from tests.test_api import TestQuantumDB
    from tests.test_training import TestLearnablePQ

    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestQuantumDB))
    suite.addTests(loader.loadTestsFromTestCase(TestLearnablePQ))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)

except Exception as e:
    print(f"Error running tests: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)
