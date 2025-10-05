#!/usr/bin/env python3
"""
Simple script to download the Wikipedia Korean embeddings dataset.
"""

import os
import json
import requests
from pathlib import Path


def download_dataset_simple():
    """Download dataset using huggingface_hub."""
    try:
        from huggingface_hub import hf_hub_download
        import polars as pl

        print("Downloading Wikipedia Korean embeddings dataset...")
        print("Importing libraries successful...")

        # Download the parquet file directly
        file_path = hf_hub_download(
            repo_id="chaehoyu/wikipedia-22-12-ko-embeddings-100k",
            filename="wikipedia-22-12-ko-embeddings-100k.parquet",
            repo_type="dataset",
        )

        print(f"Downloaded file: {file_path}")

        # Read the parquet file
        df = pl.read_parquet(file_path)

        # Create data directory
        data_dir = Path("data/wikipedia_ko_embeddings")
        data_dir.mkdir(parents=True, exist_ok=True)

        # Save as parquet
        parquet_path = data_dir / "wikipedia-22-12-ko-embeddings-100k.parquet"
        df.write_parquet(parquet_path)

        print(f"Dataset saved to: {parquet_path}")
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {df.columns}")

        # Show sample data
        print("\nSample data:")
        for i in range(min(3, len(df))):
            row = df.row(i, named=True)
            print(f"\nRow {i}:")
            print(f"  Title: {row['title']}")
            print(f"  Text: {row['text'][:100]}...")
            print(f"  Embedding length: {len(row['embedding_json'])}")

        return str(parquet_path)

    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Please install required packages:")
        print("uv add huggingface_hub pandas pyarrow")
        return None
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return None


if __name__ == "__main__":
    download_dataset_simple()
