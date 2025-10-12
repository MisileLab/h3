#!/usr/bin/env python3
"""
Analyze the combined dataset structure
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from datasets import load_from_disk
import pandas as pd

def analyze_combined_dataset():
    """Analyze the combined dataset"""
    
    print("Loading combined dataset...")
    
    try:
        # Load the combined dataset
        dataset = load_from_disk("data/processed/combined_dataset")
        
        print(f"Dataset loaded successfully!")
        print(f"Available splits: {list(dataset.keys())}")
        
        # Examine each split
        for split_name, split_data in dataset.items():
            print(f"\n{split_name.upper()} SPLIT:")
            print(f"  Size: {len(split_data)} examples")
            print(f"  Columns: {split_data.column_names}")
            
            if 'type' in split_data.column_names:
                labels = split_data['type']
                label_counts = pd.Series(labels).value_counts()
                print(f"  Label distribution:")
                for label, count in label_counts.items():
                    print(f"    {label}: {count} ({count/len(split_data)*100:.1f}%)")
            
            # Show some examples
            print(f"  Sample examples:")
            for i in range(min(3, len(split_data))):
                example = split_data[i]
                print(f"    Example {i+1}:")
                print(f"      Type: {example['type']}")
                print(f"      Prompt: {example['prompt'][:100]}...")
                if 'label' in example:
                    print(f"      Label: {example['label']}")
        
        # Check if we have reasoning data
        train_data = dataset['train']
        if 'qwq32' in train_data.column_names:
            print(f"\nREASONING ANALYSIS:")
            reasoning_examples = [ex for ex in train_data if 'qwq32' in ex and ex['qwq32']]
            print(f"  Examples with reasoning: {len(reasoning_examples)}")
            
            if reasoning_examples:
                print(f"  Sample reasoning:")
                example = reasoning_examples[0]
                print(f"    Type: {example['type']}")
                print(f"    Reasoning: {example['qwq32'][:200]}...")
        
    except Exception as e:
        print(f"Error analyzing dataset: {e}")
        raise

if __name__ == "__main__":
    analyze_combined_dataset()