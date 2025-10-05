#!/usr/bin/env python3
"""
Script to download the Wikipedia Korean embeddings dataset and create a comparison script.
"""

import os
import json
import pandas as pd
from datasets import load_dataset
import numpy as np
from typing import List, Dict, Any
import argparse


def download_dataset(save_path: str = "data/wikipedia_ko_embeddings") -> str:
    """
    Download the Wikipedia Korean embeddings dataset from Hugging Face.

    Args:
        save_path: Path to save the dataset

    Returns:
        Path to the downloaded dataset
    """
    print("Downloading Wikipedia Korean embeddings dataset...")

    # Create data directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)

    # Load dataset from Hugging Face
    dataset = load_dataset("chaehoyu/wikipedia-22-12-ko-embeddings-100k")

    # Save to disk
    dataset.save_to_disk(save_path)

    print(f"Dataset downloaded and saved to: {save_path}")
    print(f"Dataset info: {dataset}")

    return save_path


def create_comparison_script(
    dataset_path: str = "data/wikipedia_ko_embeddings",
) -> None:
    """
    Create a comparison script for the embeddings dataset.

    Args:
        dataset_path: Path to the downloaded dataset
    """
    script_content = '''#!/usr/bin/env python3
"""
Comparison script for Wikipedia Korean embeddings dataset.
This script provides various comparison and analysis functions.
"""

import os
import json
import numpy as np
import pandas as pd
from datasets import load_from_disk
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
            dataset_path: Path to the dataset
        """
        self.dataset_path = dataset_path
        self.dataset = None
        self.embeddings = None
        self.texts = None
        self.titles = None
        self.load_dataset()
    
    def load_dataset(self):
        """Load the dataset from disk."""
        print(f"Loading dataset from {self.dataset_path}...")
        self.dataset = load_from_disk(self.dataset_path)
        
        # Extract data
        train_data = self.dataset['train']
        
        # Parse embeddings from JSON strings
        self.embeddings = []
        self.texts = []
        self.titles = []
        
        for item in train_data:
            # Parse embedding JSON
            embedding = json.loads(item['embedding_json'])
            self.embeddings.append(np.array(embedding))
            self.texts.append(item['text'])
            self.titles.append(item['title'])
        
        self.embeddings = np.array(self.embeddings)
        print(f"Loaded {len(self.embeddings)} embeddings")
        print(f"Embedding dimension: {self.embeddings.shape[1]}")
    
    def find_similar_items(self, query_idx: int, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Find most similar items to a given query.
        
        Args:
            query_idx: Index of the query item
            top_k: Number of similar items to return
            
        Returns:
            List of similar items with scores
        """
        query_embedding = self.embeddings[query_idx:query_idx+1]
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Get top-k similar items (excluding the query itself)
        top_indices = np.argsort(similarities)[::-1][1:top_k+1]
        
        results = []
        for idx in top_indices:
            results.append({
                'title': self.titles[idx],
                'text': self.texts[idx][:200] + "..." if len(self.texts[idx]) > 200 else self.texts[idx],
                'similarity': similarities[idx],
                'index': idx
            })
        
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
                matches.append({
                    'title': self.titles[i],
                    'text': text[:200] + "..." if len(text) > 200 else text,
                    'index': i
                })
        
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
        emb1 = self.embeddings[idx1:idx1+1]
        emb2 = self.embeddings[idx2:idx2+1]
        
        similarity = cosine_similarity(emb1, emb2)[0][0]
        
        return {
            'item1': {
                'title': self.titles[idx1],
                'text': self.texts[idx1][:100] + "..." if len(self.texts[idx1]) > 100 else self.texts[idx1],
                'index': idx1
            },
            'item2': {
                'title': self.titles[idx2],
                'text': self.texts[idx2][:100] + "..." if len(self.texts[idx2]) > 100 else self.texts[idx2],
                'index': idx2
            },
            'similarity': similarity
        }
    
    def get_embedding_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the embeddings.
        
        Returns:
            Dictionary with embedding statistics
        """
        return {
            'total_items': len(self.embeddings),
            'embedding_dimension': self.embeddings.shape[1],
            'mean_norm': np.mean(np.linalg.norm(self.embeddings, axis=1)),
            'std_norm': np.std(np.linalg.norm(self.embeddings, axis=1)),
            'min_similarity': np.min(cosine_similarity(self.embeddings[:1000], self.embeddings[:1000])),
            'max_similarity': np.max(cosine_similarity(self.embeddings[:1000], self.embeddings[:1000]))
        }
    
    def visualize_embeddings_pca(self, sample_size: int = 1000, save_path: str = "embeddings_pca.png"):
        """
        Create a PCA visualization of embeddings.
        
        Args:
            sample_size: Number of embeddings to sample for visualization
            save_path: Path to save the visualization
        """
        print(f"Creating PCA visualization with {sample_size} samples...")
        
        # Sample embeddings
        indices = np.random.choice(len(self.embeddings), min(sample_size, len(self.embeddings)), replace=False)
        sample_embeddings = self.embeddings[indices]
        sample_titles = [self.titles[i] for i in indices]
        
        # Apply PCA
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(sample_embeddings)
        
        # Create plot
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.6, s=20)
        plt.title('PCA Visualization of Korean Wikipedia Embeddings')
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        
        # Add some labels for interesting points
        for i in range(0, len(indices), min(100, len(indices)//10)):
            plt.annotate(sample_titles[i][:20], (embeddings_2d[i, 0], embeddings_2d[i, 1]), 
                        fontsize=8, alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"PCA visualization saved to: {save_path}")
        print(f"Explained variance ratio: {pca.explained_variance_ratio_}")


def main():
    """Main function to run comparison operations."""
    parser = argparse.ArgumentParser(description='Compare Wikipedia Korean embeddings')
    parser.add_argument('--dataset_path', type=str, default='data/wikipedia_ko_embeddings',
                        help='Path to the dataset')
    parser.add_argument('--operation', type=str, choices=['similar', 'search', 'compare', 'stats', 'visualize'],
                        default='stats', help='Operation to perform')
    parser.add_argument('--index', type=int, help='Index for similar/compare operations')
    parser.add_argument('--index2', type=int, help='Second index for compare operation')
    parser.add_argument('--query', type=str, help='Query text for search operation')
    parser.add_argument('--top_k', type=int, default=5, help='Number of results to return')
    
    args = parser.parse_args()
    
    # Initialize comparator
    comparator = EmbeddingComparator(args.dataset_path)
    
    if args.operation == 'similar':
        if args.index is None:
            print("Error: --index is required for similar operation")
            return
        
        results = comparator.find_similar_items(args.index, args.top_k)
        print(f"\\nItems similar to index {args.index} ({comparator.titles[args.index]}):")
        for i, result in enumerate(results, 1):
            print(f"\\n{i}. {result['title']} (similarity: {result['similarity']:.4f})")
            print(f"   {result['text']}")
    
    elif args.operation == 'search':
        if args.query is None:
            print("Error: --query is required for search operation")
            return
        
        results = comparator.search_by_text(args.query, args.top_k)
        print(f"\\nSearch results for '{args.query}':")
        for i, result in enumerate(results, 1):
            print(f"\\n{i}. {result['title']}")
            print(f"   {result['text']}")
    
    elif args.operation == 'compare':
        if args.index is None or args.index2 is None:
            print("Error: --index and --index2 are required for compare operation")
            return
        
        result = comparator.compare_two_items(args.index, args.index2)
        print(f"\\nComparison between items {args.index} and {args.index2}:")
        print(f"\\nItem 1: {result['item1']['title']}")
        print(f"   {result['item1']['text']}")
        print(f"\\nItem 2: {result['item2']['title']}")
        print(f"   {result['item2']['text']}")
        print(f"\\nSimilarity: {result['similarity']:.4f}")
    
    elif args.operation == 'stats':
        stats = comparator.get_embedding_stats()
        print("\\nEmbedding Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    
    elif args.operation == 'visualize':
        comparator.visualize_embeddings_pca()


if __name__ == "__main__":
    main()
'''

    script_path = "compare_embeddings.py"
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(script_content)

    print(f"Comparison script created: {script_path}")

    # Create a simple example usage script
    example_script = '''#!/usr/bin/env python3
"""
Example usage of the embedding comparison script.
"""

from compare_embeddings import EmbeddingComparator

def main():
    # Initialize the comparator
    comparator = EmbeddingComparator("data/wikipedia_ko_embeddings")
    
    # Get statistics
    print("=== Embedding Statistics ===")
    stats = comparator.get_embedding_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Find similar items to the first item
    print("\\n=== Similar Items to First Entry ===")
    similar_items = comparator.find_similar_items(0, top_k=3)
    for i, item in enumerate(similar_items, 1):
        print(f"{i}. {item['title']} (similarity: {item['similarity']:.4f})")
        print(f"   {item['text'][:100]}...")
    
    # Search for items containing "문화방송"
    print("\\n=== Search Results for '문화방송' ===")
    search_results = comparator.search_by_text("문화방송", top_k=3)
    for i, item in enumerate(search_results, 1):
        print(f"{i}. {item['title']}")
        print(f"   {item['text'][:100]}...")
    
    # Compare two items
    print("\\n=== Compare Two Items ===")
    comparison = comparator.compare_two_items(0, 1)
    print(f"Similarity between '{comparison['item1']['title']}' and '{comparison['item2']['title']}': {comparison['similarity']:.4f}")
    
    # Create visualization
    print("\\n=== Creating PCA Visualization ===")
    comparator.visualize_embeddings_pca(sample_size=500)

if __name__ == "__main__":
    main()
'''

    example_path = "example_usage.py"
    with open(example_path, "w", encoding="utf-8") as f:
        f.write(example_script)

    print(f"Example usage script created: {example_path}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Download Wikipedia Korean embeddings dataset"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/wikipedia_ko_embeddings",
        help="Output directory for the dataset",
    )

    args = parser.parse_args()

    # Download dataset
    dataset_path = download_dataset(args.output_dir)

    # Create comparison script
    create_comparison_script(dataset_path)

    print("\\n=== Setup Complete ===")
    print(f"Dataset downloaded to: {dataset_path}")
    print("Comparison script: compare_embeddings.py")
    print("Example usage script: example_usage.py")
    print("\\nTo run the example:")
    print("python example_usage.py")
    print("\\nTo use the comparison script directly:")
    print("python compare_embeddings.py --operation stats")
    print("python compare_embeddings.py --operation similar --index 0 --top_k 5")
    print("python compare_embeddings.py --operation search --query '문화방송'")
    print("python compare_embeddings.py --operation compare --index 0 --index2 1")
    print("python compare_embeddings.py --operation visualize")


if __name__ == "__main__":
    main()
