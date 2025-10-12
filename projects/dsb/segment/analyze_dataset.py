#!/usr/bin/env python3
"""
Script to analyze the dvilasuero/jailbreak-classification-reasoning dataset
from Hugging Face and examine its structure.
"""

from datasets import load_dataset
import pandas as pd
from pprint import pprint

def analyze_dataset():
    """Load and analyze the jailbreak classification reasoning dataset."""
    
    print("=" * 60)
    print("Loading dvilasuero/jailbreak-classification-reasoning dataset...")
    print("=" * 60)
    
    try:
        # Load the dataset
        dataset = load_dataset("dvilasuero/jailbreak-classification-reasoning")
        
        # 1. Check available splits
        print("\n1. AVAILABLE SPLITS:")
        print("-" * 30)
        print(f"Splits available: {list(dataset.keys())}")
        
        # 2. Analyze each split
        for split_name, split_data in dataset.items():
            print(f"\n2. {split_name.upper()} SPLIT ANALYSIS:")
            print("-" * 40)
            
            # Size of split
            print(f"Size: {len(split_data)} examples")
            
            # Features/columns
            print(f"\nFeatures: {list(split_data.features.keys())}")
            
            # Feature details
            print("\nFeature Details:")
            for feature_name, feature_info in split_data.features.items():
                print(f"  - {feature_name}: {feature_info}")
            
            # Show sample data
            print(f"\nSample entries from {split_name} split:")
            print("-" * 40)
            
            # Convert to pandas for better display
            df = split_data.to_pandas()
            
            # Show first few rows
            print("First 3 entries:")
            print(df.head(3).to_string())
            
            # Show data types
            print(f"\nData types:")
            print(df.dtypes)
            
            # Show basic statistics for text columns
            text_columns = df.select_dtypes(include=['object']).columns
            if len(text_columns) > 0:
                print(f"\nText column statistics:")
                for col in text_columns:
                    print(f"  - {col}:")
                    print(f"    * Non-null count: {df[col].count()}")
                    print(f"    * Unique values: {df[col].nunique()}")
                    if df[col].count() > 0:
                        avg_length = df[col].astype(str).str.len().mean()
                        print(f"    * Average length: {avg_length:.1f} characters")
            
            # Show value counts for categorical columns if any
            categorical_columns = df.select_dtypes(include=['category', 'bool']).columns
            if len(categorical_columns) > 0:
                print(f"\nCategorical column distributions:")
                for col in categorical_columns:
                    print(f"  - {col}:")
                    print(df[col].value_counts().to_string())
            
            print("\n" + "=" * 60)
        
        # 3. Overall dataset summary
        print("\n3. OVERALL DATASET SUMMARY:")
        print("-" * 30)
        total_examples = sum(len(split) for split in dataset.values())
        print(f"Total examples across all splits: {total_examples}")
        
        # Check if there are any duplicate columns across splits
        all_features = set()
        for split_data in dataset.values():
            all_features.update(split_data.features.keys())
        
        print(f"All unique features across splits: {sorted(all_features)}")
        
        return dataset
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

if __name__ == "__main__":
    dataset = analyze_dataset()