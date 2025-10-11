#!/usr/bin/env python3
"""
Server script for ELECTRA jailbreak detection API.
"""

import argparse
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.api.server import JailbreakGuardAPI


def main():
    parser = argparse.ArgumentParser(description="Start ELECTRA jailbreak detection API server")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained model directory"
    )
    parser.add_argument(
        "--use-lora",
        action="store_true",
        help="Use LoRA model"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to"
    )
    
    args = parser.parse_args()
    
    # Check if model path exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model path {args.model_path} does not exist")
        sys.exit(1)
    
    # Initialize and run API
    api = JailbreakGuardAPI(args.model_path, args.use_lora)
    api.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()