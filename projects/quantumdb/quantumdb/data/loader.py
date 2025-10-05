"""
Data loading utilities for various datasets.

This module provides loaders for common datasets used in vector search
including MS MARCO, BEIR benchmark, and custom vector datasets.
"""

import os
import json
import pickle
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

try:
    from beir import util
    from beir.datasets.data_loader import GenericDataLoader

    BEIR_AVAILABLE = True
except ImportError:
    BEIR_AVAILABLE = False

try:
    import polars as pl

    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False


class VectorDataset(Dataset):
    """PyTorch dataset for vector data."""

    def __init__(self, vectors: np.ndarray, ids: Optional[np.ndarray] = None):
        self.vectors = torch.FloatTensor(vectors)
        self.ids = ids if ids is not None else np.arange(len(vectors))

    def __len__(self):
        return len(self.vectors)

    def __getitem__(self, idx):
        return self.vectors[idx], self.ids[idx]


class MSLoader:
    """
    Loader for MS MARCO dataset.

    Supports loading passages, queries, and relevance judgments
    from the MS MARCO passage ranking dataset.
    """

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.logger = logging.getLogger(__name__)

        # Expected file structure
        self.passages_file = self.data_dir / "collection.tsv"
        self.queries_file = self.data_dir / "queries.dev.tsv"
        self.qrels_file = self.data_dir / "qrels.dev.tsv"

    def load_passages(self) -> Dict[str, str]:
        """Load passage documents."""
        if not self.passages_file.exists():
            raise FileNotFoundError(f"Passages file not found: {self.passages_file}")

        passages = {}
        with open(self.passages_file, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc="Loading passages"):
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    doc_id, doc_text = parts[0], parts[1]
                    passages[doc_id] = doc_text

        self.logger.info(f"Loaded {len(passages)} passages")
        return passages

    def load_queries(self) -> Dict[str, str]:
        """Load queries."""
        if not self.queries_file.exists():
            raise FileNotFoundError(f"Queries file not found: {self.queries_file}")

        queries = {}
        with open(self.queries_file, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc="Loading queries"):
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    query_id, query_text = parts[0], parts[1]
                    queries[query_id] = query_text

        self.logger.info(f"Loaded {len(queries)} queries")
        return queries

    def load_qrels(self) -> Dict[str, List[str]]:
        """Load relevance judgments."""
        if not self.qrels_file.exists():
            raise FileNotFoundError(f"Qrels file not found: {self.qrels_file}")

        qrels = {}
        with open(self.qrels_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 4:
                    query_id, _, doc_id, relevance = parts[:4]
                    if relevance == "1":  # Only include relevant documents
                        if query_id not in qrels:
                            qrels[query_id] = []
                        qrels[query_id].append(doc_id)

        self.logger.info(f"Loaded qrels for {len(qrels)} queries")
        return qrels

    def load_all(self) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, List[str]]]:
        """Load all MS MARCO data."""
        passages = self.load_passages()
        queries = self.load_queries()
        qrels = self.load_qrels()

        return passages, queries, qrels


class BEIRLoader:
    """
    Loader for BEIR benchmark datasets.

    Supports loading any dataset from the BEIR benchmark collection.
    """

    def __init__(self, dataset_name: str, data_dir: str = "beir_data"):
        if not BEIR_AVAILABLE:
            raise ImportError("BEIR is not installed. Install with: pip install beir")

        self.dataset_name = dataset_name
        self.data_dir = Path(data_dir)
        self.logger = logging.getLogger(__name__)

        # Download and load dataset
        self.data_path = self.data_dir / dataset_name
        if not self.data_path.exists():
            self.logger.info(f"Downloading {dataset_name} dataset...")
            url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
            util.download_and_unzip(url, self.data_dir)

        self.corpus, self.queries, self.qrels = GenericDataLoader(
            data_folder=self.data_path
        ).load(split="test")

    def get_corpus(self) -> Dict[str, Dict[str, str]]:
        """Get corpus documents."""
        return self.corpus

    def get_queries(self) -> Dict[str, str]:
        """Get queries."""
        return self.queries

    def get_qrels(self) -> Dict[str, Dict[str, int]]:
        """Get relevance judgments."""
        return self.qrels

    def get_documents(self) -> List[str]:
        """Extract document texts."""
        return [doc["text"] for doc in self.corpus.values()]

    def get_document_ids(self) -> List[str]:
        """Extract document IDs."""
        return list(self.corpus.keys())

    def get_query_texts(self) -> List[str]:
        """Extract query texts."""
        return list(self.queries.values())

    def get_query_ids(self) -> List[str]:
        """Extract query IDs."""
        return list(self.queries.keys())


class WikipediaParquetLoader:
    """
    Loader for Wikipedia Korean embeddings dataset.

    Loads embeddings from the parquet file downloaded by simple_download.py.
    """

    def __init__(
        self,
        parquet_path: str = "data/wikipedia_ko_embeddings/wikipedia-22-12-ko-embeddings-100k.parquet",
    ):
        if not POLARS_AVAILABLE:
            raise ImportError(
                "polars is required for loading parquet files. Install with: pip install polars"
            )

        self.parquet_path = Path(parquet_path)
        self.logger = logging.getLogger(__name__)

        if not self.parquet_path.exists():
            raise FileNotFoundError(f"Parquet file not found: {self.parquet_path}")

    def load_data(
        self, max_samples: Optional[int] = None
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Load embeddings and metadata from parquet file."""
        self.logger.info(f"Loading data from {self.parquet_path}")

        # Read parquet file
        df = pl.read_parquet(self.parquet_path)

        if max_samples:
            df = df.head(max_samples)

        # Extract embeddings from JSON column
        embeddings_list = []
        for embedding_json in df["embedding_json"]:
            # Parse JSON string to list and convert to numpy array
            embedding = np.array(eval(embedding_json), dtype=np.float32)
            embeddings_list.append(embedding)

        embeddings = np.stack(embeddings_list)

        # Create metadata
        metadata = []
        for i, row in enumerate(df.iter_rows(named=True)):
            metadata.append(
                {
                    "id": f"wiki_{i}",
                    "title": row["title"],
                    "text": row["text"],
                    "url": row.get("url", ""),
                    "length": len(row["text"]),
                }
            )

        self.logger.info(
            f"Loaded {len(embeddings)} embeddings with shape {embeddings.shape}"
        )
        return embeddings, metadata

    def get_embeddings(self, max_samples: Optional[int] = None) -> np.ndarray:
        """Get only embeddings."""
        embeddings, _ = self.load_data(max_samples)
        return embeddings

    def get_metadata(self, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get only metadata."""
        _, metadata = self.load_data(max_samples)
        return metadata


class EmbeddingCache:
    """Cache for storing and loading embeddings."""

    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)

    def get_cache_path(self, name: str, suffix: str = ".npy") -> Path:
        """Get cache file path."""
        return self.cache_dir / f"{name}{suffix}"

    def save_embeddings(self, embeddings: np.ndarray, name: str) -> Path:
        """Save embeddings to cache."""
        cache_path = self.get_cache_path(name)
        np.save(cache_path, embeddings)
        self.logger.info(f"Saved embeddings to {cache_path}")
        return cache_path

    def load_embeddings(self, name: str) -> Optional[np.ndarray]:
        """Load embeddings from cache."""
        cache_path = self.get_cache_path(name)
        if cache_path.exists():
            embeddings = np.load(cache_path)
            self.logger.info(f"Loaded embeddings from {cache_path}")
            return embeddings
        return None

    def save_metadata(self, metadata: Dict[str, Any], name: str) -> Path:
        """Save metadata to cache."""
        cache_path = self.get_cache_path(name, ".json")
        with open(cache_path, "w") as f:
            json.dump(metadata, f, indent=2)
        return cache_path

    def load_metadata(self, name: str) -> Optional[Dict[str, Any]]:
        """Load metadata from cache."""
        cache_path = self.get_cache_path(name, ".json")
        if cache_path.exists():
            with open(cache_path, "r") as f:
                return json.load(f)
        return None


class SyntheticDataGenerator:
    """Generate synthetic vector data for testing."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        np.random.seed(random_state)

    def generate_random_vectors(
        self, n_samples: int, dimension: int, distribution: str = "normal"
    ) -> np.ndarray:
        """Generate random vectors."""
        if distribution == "normal":
            return np.random.randn(n_samples, dimension).astype(np.float32)
        elif distribution == "uniform":
            return np.random.uniform(-1, 1, (n_samples, dimension)).astype(np.float32)
        else:
            raise ValueError(f"Unsupported distribution: {distribution}")

    def generate_clustered_vectors(
        self,
        n_samples: int,
        dimension: int,
        n_clusters: int = 10,
        cluster_std: float = 0.1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate clustered vectors with cluster labels."""
        # Generate cluster centers
        centers = np.random.randn(n_clusters, dimension)

        # Generate samples
        vectors = []
        labels = []
        samples_per_cluster = n_samples // n_clusters

        for i in range(n_clusters):
            cluster_vectors = (
                np.random.randn(samples_per_cluster, dimension) * cluster_std
                + centers[i]
            )
            vectors.append(cluster_vectors)
            labels.extend([i] * samples_per_cluster)

        # Handle remaining samples
        remaining = n_samples - len(vectors) * samples_per_cluster
        if remaining > 0:
            last_cluster_vectors = (
                np.random.randn(remaining, dimension) * cluster_std + centers[-1]
            )
            vectors.append(last_cluster_vectors)
            labels.extend([n_clusters - 1] * remaining)

        vectors = np.vstack(vectors).astype(np.float32)
        labels = np.array(labels)

        return vectors, labels

    def generate_text_like_vectors(
        self, n_samples: int, dimension: int = 768, sparsity: float = 0.1
    ) -> np.ndarray:
        """Generate vectors that mimic text embedding properties."""
        # Create sparse-like vectors
        vectors = np.random.randn(n_samples, dimension) * sparsity

        # Add some dense components (like common words)
        n_dense = int(dimension * 0.3)
        dense_indices = np.random.choice(dimension, n_dense, replace=False)
        vectors[:, dense_indices] += np.random.randn(n_samples, n_dense) * 0.5

        # Normalize to unit length (like sentence embeddings)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / (norms + 1e-8)

        return vectors.astype(np.float32)


def load_embeddings_from_file(file_path: str) -> np.ndarray:
    """Load embeddings from various file formats."""
    file_path = Path(file_path)

    if file_path.suffix == ".npy":
        return np.load(file_path)
    elif file_path.suffix == ".npz":
        data = np.load(file_path)
        # Assume embeddings are stored under 'embeddings' key
        return data["embeddings"]
    elif file_path.suffix == ".pkl":
        with open(file_path, "rb") as f:
            data = pickle.load(f)
            # Handle different pickle formats
            if isinstance(data, dict):
                return data.get(
                    "embeddings", data.get("vectors", list(data.values())[0])
                )
            else:
                return np.array(data)
    elif file_path.suffix == ".parquet":
        if not POLARS_AVAILABLE:
            raise ImportError(
                "polars is required for loading parquet files. Install with: pip install polars"
            )

        df = pl.read_parquet(file_path)
        # Extract embeddings from JSON column
        embeddings_list = []
        for embedding_json in df["embedding_json"]:
            # Parse JSON string to list and convert to numpy array
            embedding = np.array(eval(embedding_json), dtype=np.float32)
            embeddings_list.append(embedding)

        return np.stack(embeddings_list)
    elif file_path.suffix in [".csv", ".tsv"]:
        delimiter = "," if file_path.suffix == ".csv" else "\t"
        df = pd.read_csv(file_path, delimiter=delimiter)
        return df.values.astype(np.float32)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")


def save_embeddings_to_file(
    embeddings: np.ndarray, file_path: str, metadata: Optional[Dict[str, Any]] = None
):
    """Save embeddings to various file formats."""
    file_path = Path(file_path)

    if file_path.suffix == ".npy":
        np.save(file_path, embeddings)
    elif file_path.suffix == ".npz":
        if metadata:
            np.savez_compressed(file_path, embeddings=embeddings, **metadata)
        else:
            np.savez_compressed(file_path, embeddings=embeddings)
    elif file_path.suffix == ".pkl":
        data = {"embeddings": embeddings}
        if metadata:
            data.update(metadata)
        with open(file_path, "wb") as f:
            pickle.dump(data, f)
    elif file_path.suffix in [".csv", ".tsv"]:
        delimiter = "," if file_path.suffix == ".csv" else "\t"
        df = pd.DataFrame(embeddings)
        df.to_csv(file_path, index=False, sep=delimiter)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")
