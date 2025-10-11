#!/usr/bin/env python3
"""
Training script for ELECTRA jailbreak detection model.
"""

import argparse
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.training.trainer import JailbreakTrainer


def main():
    parser = argparse.ArgumentParser(description="Train ELECTRA jailbreak detection model")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/model_config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./checkpoints",
        help="Output directory for model checkpoints"
    )
    parser.add_argument(
        "--use-lora",
        action="store_true",
        help="Use LoRA fine-tuning instead of full fine-tuning"
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize trainer
    trainer = JailbreakTrainer(args.config)
    
    # Run training
    results = trainer.run_full_training(
        output_dir=args.output_dir,
        use_lora=args.use_lora
    )
    
    print("Training completed successfully!")
    print(f"Results: {results}")


if __name__ == "__main__":
    main()