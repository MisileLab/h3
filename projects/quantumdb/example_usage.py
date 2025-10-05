#!/usr/bin/env python3
"""
Example usage of the embedding comparison script.
"""

from compare_embeddings import EmbeddingComparator


def main():
    # Initialize the comparator
    # Update this path to where your parquet file is located
    dataset_path = (
        "data/wikipedia_ko_embeddings/wikipedia-22-12-ko-embeddings-100k.parquet"
    )

    try:
        comparator = EmbeddingComparator(dataset_path)

        # Get statistics
        print("=== Embedding Statistics ===")
        stats = comparator.get_embedding_stats()
        for key, value in stats.items():
            print(f"{key}: {value}")

        # Find similar items to the first item
        print("\n=== Similar Items to First Entry ===")
        similar_items = comparator.find_similar_items(0, top_k=3)
        for i, item in enumerate(similar_items, 1):
            print(f"{i}. {item['title']} (similarity: {item['similarity']:.4f})")
            print(f"   {item['text'][:100]}...")

        # Search for items containing specific Korean terms
        print("\n=== Search Results for '문화방송' ===")
        search_results = comparator.search_by_text("문화방송", top_k=3)
        for i, item in enumerate(search_results, 1):
            print(f"{i}. {item['title']}")
            print(f"   {item['text'][:100]}...")

        # Compare two items
        print("\n=== Compare Two Items ===")
        comparison = comparator.compare_two_items(0, 1)
        print(
            f"Similarity between '{comparison['item1']['title']}' and '{comparison['item2']['title']}': {comparison['similarity']:.4f}"
        )

        # Find most similar pairs
        print("\n=== Most Similar Pairs ===")
        pairs = comparator.find_most_similar_pairs(n_pairs=3)
        for i, pair in enumerate(pairs, 1):
            print(f"{i}. Similarity: {pair['similarity']:.4f}")
            print(f"   {pair['title1']}")
            print(f"   {pair['title2']}")

        # Create visualization
        print("\n=== Creating PCA Visualization ===")
        comparator.visualize_embeddings_pca(sample_size=500)

        # Analyze clusters
        print("\n=== Cluster Analysis ===")
        comparator.analyze_embedding_clusters(n_clusters=3, sample_size=500)

    except FileNotFoundError:
        print(f"Dataset file not found at: {dataset_path}")
        print("Please make sure you have downloaded the dataset first.")
        print("You can download it using:")
        print("uv run python simple_download.py")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
