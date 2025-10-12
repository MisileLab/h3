#!/usr/bin/env python3
"""
Test script to download and examine the jailbreak reasoning dataset
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

try:
    from datasets import load_dataset
    import pandas as pd
except ImportError as e:
    print(f"Error: Required packages not installed. Please run: pip install datasets pandas")
    print(f"Missing package: {e}")
    sys.exit(1)

def test_jailbreak_reasoning_dataset():
    """Test downloading and examining the jailbreak reasoning dataset"""
    
    print("Loading dvilasuero/jailbreak-classification-reasoning dataset...")
    
    try:
        # Load the dataset
        dataset = load_dataset("dvilasuero/jailbreak-classification-reasoning")
        
        print(f"Dataset loaded successfully!")
        print(f"Available splits: {list(dataset.keys())}")
        
        # Examine the train split
        if 'train' in dataset:
            train_data = dataset['train']
            print(f"\nTrain split size: {len(train_data)}")
            print(f"Columns: {train_data.column_names}")
            
            # Show some examples
            print("\nFirst 5 examples:")
            for i in range(min(5, len(train_data))):
                example = train_data[i]
                print(f"\nExample {i+1}:")
                print(f"Type: {example['type']}")
                print(f"Prompt: {example['prompt'][:200]}...")
                if 'qwq32' in example:
                    print(f"Reasoning: {example['qwq32'][:200]}...")
            
            # Show label distribution
            labels = train_data['type']
            label_counts = pd.Series(labels).value_counts()
            print(f"\nLabel distribution:")
            for label, count in label_counts.items():
                print(f"  {label}: {count}")
        
        return dataset
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise

if __name__ == "__main__":
    test_jailbreak_reasoning_dataset()