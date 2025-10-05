#!/usr/bin/env python3
"""
Check what files are available in the dataset.
"""

from huggingface_hub import HfFileSystem
import json


def check_dataset_files():
    """Check available files in the dataset."""
    fs = HfFileSystem()

    repo_id = "chaehoyu/wikipedia-22-12-ko-embeddings-100k"

    try:
        # List all files in the repository
        files = fs.ls(f"datasets/{repo_id}")
        print("Available files:")
        for file in files:
            print(f"  {file}")

        # Check if there's a parquet directory
        try:
            parquet_files = fs.ls(f"datasets/{repo_id}/refs/convert/parquet")
            print("\nParquet files:")
            for file in parquet_files:
                print(f"  {file}")
        except:
            print("\nNo parquet directory found")

        # Try to get dataset info
        try:
            from datasets import load_dataset

            dataset = load_dataset(repo_id)
            print(f"\nDataset info: {dataset}")
            print(f"Split: {dataset['train']}")
            print(f"Features: {dataset['train'].features}")
        except Exception as e:
            print(f"\nError loading dataset: {e}")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    check_dataset_files()
