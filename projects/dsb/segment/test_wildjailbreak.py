#!/usr/bin/env python3
"""
Test script to verify WildJailbreak dataset loading
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from datasets import load_dataset
import pandas as pd

def test_wildjailbreak_loading():
    """Test loading WildJailbreak dataset"""
    print("Testing WildJailbreak dataset loading...")
    
    try:
        # Load the WildJailbreak training set
        print("Loading training set...")
        train_dataset = load_dataset("allenai/wildjailbreak", "train", delimiter="\t", keep_default_na=False)
        print(f"Train dataset type: {type(train_dataset)}")
        
        if hasattr(train_dataset, 'column_names'):
            print(f"Train dataset columns: {train_dataset.column_names}")
        elif hasattr(train_dataset, 'features'):
            print(f"Train dataset features: {train_dataset.features}")
            
        # Load the WildJailbreak evaluation set
        print("\nLoading evaluation set...")
        eval_dataset = load_dataset("allenai/wildjailbreak", "eval", delimiter="\t", keep_default_na=False)
        print(f"Eval dataset type: {type(eval_dataset)}")
        
        if hasattr(eval_dataset, 'column_names'):
            print(f"Eval dataset columns: {eval_dataset.column_names}")
        elif hasattr(eval_dataset, 'features'):
            print(f"Eval dataset features: {eval_dataset.features}")
        
        # Convert to pandas to inspect structure
        print("\nConverting to pandas for inspection...")
        train_df = train_dataset.to_pandas()
        eval_df = eval_dataset.to_pandas()
        
        print(f"Train DataFrame shape: {train_df.shape}")
        print(f"Eval DataFrame shape: {eval_df.shape}")
        print(f"Train DataFrame columns: {list(train_df.columns)}")
        print(f"Eval DataFrame columns: {list(eval_df.columns)}")
        
        # Show sample data
        print("\nSample training data:")
        print(train_df.head(2))
        
        print("\nSample evaluation data:")
        print(eval_df.head(2))
        
        # Check label distribution
        if 'type' in train_df.columns:
            print(f"\nTrain label distribution:\n{train_df['type'].value_counts()}")
        
        if 'type' in eval_df.columns:
            print(f"\nEval label distribution:\n{eval_df['type'].value_counts()}")
        
        print("\n✅ WildJailbreak dataset loading test successful!")
        return True
        
    except Exception as e:
        print(f"❌ Error loading WildJailbreak dataset: {e}")
        return False

if __name__ == "__main__":
    test_wildjailbreak_loading()