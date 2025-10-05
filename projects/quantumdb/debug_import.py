#!/usr/bin/env python3
"""Debug script to test torch imports in isolation"""

import sys
import os

print("=== Debugging torch import issue ===")
print(f"Python executable: {sys.executable}")
print(f"Current working directory: {os.getcwd()}")
print(f"Python path: {sys.path}")

print("\n=== Testing torch import directly ===")
try:
    import torch

    print(f"✓ torch imported successfully from {torch.__file__}")
    print(f"  torch version: {torch.__version__}")
except Exception as e:
    print(f"✗ Failed to import torch: {e}")

print("\n=== Testing torch.nn import directly ===")
try:
    import torch.nn as nn

    print(f"✓ torch.nn imported successfully from {nn.__file__}")
except Exception as e:
    print(f"✗ Failed to import torch.nn: {e}")

print("\n=== Testing quantumdb import ===")
try:
    import quantumdb

    print(f"✓ quantumdb imported successfully from {quantumdb.__file__}")
except Exception as e:
    print(f"✗ Failed to import quantumdb: {e}")
    import traceback

    traceback.print_exc()

print("\n=== Testing LearnablePQ import ===")
try:
    from quantumdb.training.model import LearnablePQ

    print("✓ LearnablePQ imported successfully")
except Exception as e:
    print(f"✗ Failed to import LearnablePQ: {e}")
    import traceback

    traceback.print_exc()
