#!/usr/bin/env python3
"""
Test loading the dataset with the JailbreakDataset class
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from data.datasets import JailbreakDataset
from transformers import AutoTokenizer

def test_dataset_loading():
    """Test loading the dataset with the JailbreakDataset class"""
    
    print("Testing JailbreakDataset class...")
    
    try:
        # Create a simple config
        config = {
            "data": {
                "processed_path": "data/processed",
                "max_length": 512
            }
        }
        
        # Initialize dataset
        jailbreak_dataset = JailbreakDataset(config)
        
        # Get dataset statistics
        stats = jailbreak_dataset.get_dataset_statistics()
        
        print("Dataset Statistics:")
        for split_name, split_stats in stats.items():
            print(f"\n{split_name.upper()}:")
            for key, value in split_stats.items():
                print(f"  {key}: {value}")
        
        # Test with tokenizer
        print("\nTesting with tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        train_dataset, eval_dataset = jailbreak_dataset.prepare_datasets(tokenizer)
        
        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Eval dataset size: {len(eval_dataset)}")
        
        # Show a tokenized example
        print("\nSample tokenized example:")
        example = train_dataset[0]
        print(f"  Input IDs shape: {len(example['input_ids'])}")
        print(f"  Attention mask shape: {len(example['attention_mask'])}")
        print(f"  Label: {example['label']}")
        
        print("\nDataset loading test completed successfully!")
        
    except Exception as e:
        print(f"Error testing dataset loading: {e}")
        raise

if __name__ == "__main__":
    test_dataset_loading()