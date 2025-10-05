#!/usr/bin/env python3
"""
Comparison script for Wikipedia Korean embeddings dataset.
This script provides various comparison and analysis functions.
"""

import os
import json
import numpy as np
import polars as pl
from typing import List, Dict, Any, Tuple
import argparse
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns


class EmbeddingComparator:
    """Class for comparing and analyzing embeddings."""

    def __init__(self, dataset_path: str):
        """
        Initialize the comparator with dataset path.

        Args:
            dataset_path: Path to the parquet file
        """
        self.dataset_path = dataset_path
        self.df = None
        self.embeddings = None
        self.texts = None
        self.titles = None
        self.load_dataset()

    def load_dataset(self):
        """Load the dataset from parquet file."""
        print(f"Loading dataset from {self.dataset_path}...")

        # Load parquet file
        self.df = pl.read_parquet(self.dataset_path)

        # Parse embeddings from JSON strings
        self.embeddings = []
        self.texts = []
        self.titles = []

        for row in self.df.iter_rows(named=True):
            # Parse embedding JSON
            embedding = json.loads(row["embedding_json"])
            self.embeddings.append(np.array(embedding))
            self.texts.append(row["text"])
            self.titles.append(row["title"])

        self.embeddings = np.array(self.embeddings)
        print(f"Loaded {len(self.embeddings)} embeddings")
        print(f"Embedding dimension: {self.embeddings.shape[1]}")

    def find_similar_items(
        self, query_idx: int, top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Find most similar items to a given query.

        Args:
            query_idx: Index of the query item
            top_k: Number of similar items to return

        Returns:
            List of similar items with scores
        """
        query_embedding = self.embeddings[query_idx : query_idx + 1]

        # Calculate cosine similarity
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]

        # Get top-k similar items (excluding the query itself)
        top_indices = np.argsort(similarities)[::-1][1 : top_k + 1]

        results = []
        for idx in top_indices:
            results.append(
                {
                    "title": self.titles[idx],
                    "text": self.texts[idx][:200] + "..."
                    if len(self.texts[idx]) > 200
                    else self.texts[idx],
                    "similarity": similarities[idx],
                    "index": idx,
                }
            )

        return results

    def search_by_text(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search items by text content (simple keyword matching).

        Args:
            query_text: Query text to search for
            top_k: Number of results to return

        Returns:
            List of matching items
        """
        query_lower = query_text.lower()
        matches = []

        for i, text in enumerate(self.texts):
            if query_lower in text.lower():
                matches.append(
                    {
                        "title": self.titles[i],
                        "text": text[:200] + "..." if len(text) > 200 else text,
                        "index": i,
                    }
                )

        return matches[:top_k]

    def compare_two_items(self, idx1: int, idx2: int) -> Dict[str, Any]:
        """
        Compare two items and return their similarity.

        Args:
            idx1: Index of first item
            idx2: Index of second item

        Returns:
            Comparison result
        """
        emb1 = self.embeddings[idx1 : idx1 + 1]
        emb2 = self.embeddings[idx2 : idx2 + 1]

        similarity = cosine_similarity(emb1, emb2)[0][0]

        return {
            "item1": {
                "title": self.titles[idx1],
                "text": self.texts[idx1][:100] + "..."
                if len(self.texts[idx1]) > 100
                else self.texts[idx1],
                "index": idx1,
            },
            "item2": {
                "title": self.titles[idx2],
                "text": self.texts[idx2][:100] + "..."
                if len(self.texts[idx2]) > 100
                else self.texts[idx2],
                "index": idx2,
            },
            "similarity": similarity,
        }

    def get_embedding_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the embeddings.

        Returns:
            Dictionary with embedding statistics
        """
        # Sample for similarity calculations to avoid memory issues
        sample_size = min(1000, len(self.embeddings))
        sample_indices = np.random.choice(
            len(self.embeddings), sample_size, replace=False
        )
        sample_embeddings = self.embeddings[sample_indices]

        # Calculate pairwise similarities on sample
        sample_similarities = cosine_similarity(sample_embeddings)

        return {
            "total_items": len(self.embeddings),
            "embedding_dimension": self.embeddings.shape[1],
            "mean_norm": np.mean(np.linalg.norm(self.embeddings, axis=1)),
            "std_norm": np.std(np.linalg.norm(self.embeddings, axis=1)),
            "min_similarity": np.min(
                sample_similarities[np.triu_indices_from(sample_similarities, k=1)]
            ),
            "max_similarity": np.max(
                sample_similarities[np.triu_indices_from(sample_similarities, k=1)]
            ),
            "mean_similarity": np.mean(
                sample_similarities[np.triu_indices_from(sample_similarities, k=1)]
            ),
        }

    def visualize_embeddings_pca(
        self, sample_size: int = 1000, save_path: str = "embeddings_pca.png"
    ):
        """
        Create a PCA visualization of embeddings.

        Args:
            sample_size: Number of embeddings to sample for visualization
            save_path: Path to save the visualization
        """
        print(f"Creating PCA visualization with {sample_size} samples...")

        # Sample embeddings
        indices = np.random.choice(
            len(self.embeddings), min(sample_size, len(self.embeddings)), replace=False
        )
        sample_embeddings = self.embeddings[indices]
        sample_titles = [self.titles[i] for i in indices]

        # Apply PCA
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(sample_embeddings)

        # Create plot
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.6, s=20)
        plt.title("PCA Visualization of Korean Wikipedia Embeddings")
        plt.xlabel("First Principal Component")
        plt.ylabel("Second Principal Component")

        # Add some labels for interesting points
        label_interval = max(1, len(indices) // 20)  # Show at most 20 labels
        for i in range(0, len(indices), label_interval):
            plt.annotate(
                sample_titles[i][:20],
                (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                fontsize=8,
                alpha=0.7,
            )

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"PCA visualization saved to: {save_path}")
        print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
        print(f"Total explained variance: {sum(pca.explained_variance_ratio_):.3f}")

    def find_most_similar_pairs(self, n_pairs: int = 10) -> List[Dict[str, Any]]:
        """
        Find the most similar pairs of items in the dataset.

        Args:
            n_pairs: Number of pairs to return

        Returns:
            List of most similar pairs
        """
        # Sample to avoid O(n^2) complexity
        sample_size = min(1000, len(self.embeddings))
        sample_indices = np.random.choice(
            len(self.embeddings), sample_size, replace=False
        )
        sample_embeddings = self.embeddings[sample_indices]

        # Calculate pairwise similarities
        similarities = cosine_similarity(sample_embeddings)

        # Find top similar pairs (excluding diagonal)
        pairs = []
        for i in range(len(sample_indices)):
            for j in range(i + 1, len(sample_indices)):
                pairs.append(
                    {
                        "similarity": similarities[i, j],
                        "idx1": sample_indices[i],
                        "idx2": sample_indices[j],
                        "title1": self.titles[sample_indices[i]],
                        "title2": self.titles[sample_indices[j]],
                    }
                )

        # Sort by similarity and return top pairs
        pairs.sort(key=lambda x: x["similarity"], reverse=True)
        return pairs[:n_pairs]

    def analyze_embedding_clusters(self, n_clusters: int = 5, sample_size: int = 1000):
        """
        Analyze clusters in the embeddings using K-means.

        Args:
            n_clusters: Number of clusters to find
            sample_size: Number of embeddings to sample
        """
        from sklearn.cluster import KMeans

        print(f"Analyzing {n_clusters} clusters with {sample_size} samples...")

        # Sample embeddings
        indices = np.random.choice(
            len(self.embeddings), min(sample_size, len(self.embeddings)), replace=False
        )
        sample_embeddings = self.embeddings[indices]

        # Apply K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(sample_embeddings)

        # Analyze clusters
        for cluster_id in range(n_clusters):
            cluster_indices = indices[cluster_labels == cluster_id]
            print(f"\nCluster {cluster_id} ({len(cluster_indices)} items):")

            # Show some sample titles from this cluster
            sample_titles = [self.titles[i] for i in cluster_indices[:5]]
            for title in sample_titles:
                print(f"  - {title}")

        return cluster_labels, kmeans


def main():
    """Main function to run comparison operations."""
    parser = argparse.ArgumentParser(description="Compare Wikipedia Korean embeddings")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="data/wikipedia_ko_embeddings/wikipedia-22-12-ko-embeddings-100k.parquet",
        help="Path to the parquet dataset file",
    )
    parser.add_argument(
        "--operation",
        type=str,
        choices=[
            "similar",
            "search",
            "compare",
            "stats",
            "visualize",
            "pairs",
            "clusters",
        ],
        default="stats",
        help="Operation to perform",
    )
    parser.add_argument(
        "--index", type=int, help="Index for similar/compare operations"
    )
    parser.add_argument("--index2", type=int, help="Second index for compare operation")
    parser.add_argument("--query", type=str, help="Query text for search operation")
    parser.add_argument(
        "--top_k", type=int, default=5, help="Number of results to return"
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=1000,
        help="Sample size for visualization/clustering",
    )
    parser.add_argument(
        "--n_clusters",
        type=int,
        default=5,
        help="Number of clusters for cluster analysis",
    )

    args = parser.parse_args()

    # Initialize comparator
    comparator = EmbeddingComparator(args.dataset_path)

    if args.operation == "similar":
        if args.index is None:
            print("Error: --index is required for similar operation")
            return

        results = comparator.find_similar_items(args.index, args.top_k)
        print(
            f"\nItems similar to index {args.index} ({comparator.titles[args.index]}):"
        )
        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result['title']} (similarity: {result['similarity']:.4f})")
            print(f"   {result['text']}")

    elif args.operation == "search":
        if args.query is None:
            print("Error: --query is required for search operation")
            return

        results = comparator.search_by_text(args.query, args.top_k)
        print(f"\nSearch results for '{args.query}':")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result['title']}")
            print(f"   {result['text']}")

    elif args.operation == "compare":
        if args.index is None or args.index2 is None:
            print("Error: --index and --index2 are required for compare operation")
            return

        result = comparator.compare_two_items(args.index, args.index2)
        print(f"\nComparison between items {args.index} and {args.index2}:")
        print(f"\nItem 1: {result['item1']['title']}")
        print(f"   {result['item1']['text']}")
        print(f"\nItem 2: {result['item2']['title']}")
        print(f"   {result['item2']['text']}")
        print(f"\nSimilarity: {result['similarity']:.4f}")

    elif args.operation == "stats":
        stats = comparator.get_embedding_stats()
        print("\nEmbedding Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

    elif args.operation == "visualize":
        comparator.visualize_embeddings_pca(args.sample_size)

    elif args.operation == "pairs":
        pairs = comparator.find_most_similar_pairs(args.top_k)
        print(f"\nTop {args.top_k} most similar pairs:")
        for i, pair in enumerate(pairs, 1):
            print(f"\n{i}. Similarity: {pair['similarity']:.4f}")
            print(f"   {pair['title1']}")
            print(f"   {pair['title2']}")

    elif args.operation == "clusters":
        comparator.analyze_embedding_clusters(args.n_clusters, args.sample_size)


if __name__ == "__main__":
    main()
