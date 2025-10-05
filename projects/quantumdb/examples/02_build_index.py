#!/usr/bin/env python3
"""
Example 2: Building a vector index with QuantumDB.

This script demonstrates how to use QuantumDB to build a vector index
with a trained compression model and add vectors to it.
"""

import numpy as np
import time
from pathlib import Path

from quantumdb import QuantumDB
from quantumdb.data import SyntheticDataGenerator, create_embedder


def main():
    """Main index building script."""
    print("🏗️  Building Vector Index with QuantumDB")
    print("=" * 50)

    # Configuration
    config = {
        "model_path": "models/learnablepq_final.safetensors",
        "collection_name": "demo_collection",
        "n_documents": 10000,
        "dimension": 768,
        "batch_size": 1000,
    }

    print(f"Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()

    # Check if model exists
    model_path = Path(config["model_path"])
    if not model_path.exists():
        print(f"❌ Model file not found: {model_path}")
        print("Please run example 01 first to train a model.")
        return

    # Generate synthetic document data
    print("📄 Generating synthetic document data...")
    data_generator = SyntheticDataGenerator(random_state=42)

    # Generate document vectors
    doc_vectors = data_generator.generate_text_like_vectors(
        n_samples=config["n_documents"], dimension=config["dimension"], sparsity=0.1
    )

    # Generate document metadata
    doc_ids = [f"doc_{i:06d}" for i in range(config["n_documents"])]
    doc_metadata = [
        {
            "title": f"Document {i}",
            "category": f"category_{i % 10}",
            "length": np.random.randint(100, 5000),
            "timestamp": time.time() - np.random.randint(0, 86400 * 30),  # Last 30 days
        }
        for i in range(config["n_documents"])
    ]

    print(f"Generated {len(doc_vectors)} documents")
    print(f"Vector dimension: {doc_vectors.shape[1]}")
    print()

    # Initialize QuantumDB
    print("🚀 Initializing QuantumDB...")
    db = QuantumDB(
        model_path=config["model_path"],
        collection_name=config["collection_name"],
        vector_size=256,  # Compressed dimension
        distance="cosine",
    )

    # Get collection info
    collection_info = db.get_collection_info()
    print(f"Collection info:")
    print(f"  Name: {collection_info.get('name')}")
    print(f"  Vector size: {collection_info.get('vector_size')}")
    print(f"  Distance: {collection_info.get('distance')}")
    print()

    # Add vectors to the database
    print("📚 Adding vectors to database...")
    start_time = time.time()

    result = db.add(
        vectors=doc_vectors,
        ids=doc_ids,
        payloads=doc_metadata,
        batch_size=config["batch_size"],
    )

    end_time = time.time()
    indexing_time = end_time - start_time

    print(f"✅ Indexing completed!")
    print(f"  Vectors added: {result['vectors_added']}")
    print(f"  Indexing time: {indexing_time:.2f} seconds")
    print(f"  Indexing speed: {len(doc_vectors) / indexing_time:.0f} vectors/second")
    print()

    # Get final collection info
    final_info = db.get_collection_info()
    print(f"Final collection info:")
    print(f"  Total vectors: {final_info.get('vectors_count')}")
    print()

    # Test search functionality
    print("🔍 Testing search functionality...")

    # Generate a test query
    query_vector = data_generator.generate_text_like_vectors(
        n_samples=1, dimension=config["dimension"]
    )[0]

    # Perform search
    search_start = time.time()
    search_results = db.search(
        query_vector=query_vector,
        limit=10,
        with_payload=True,
    )
    search_time = time.time() - search_start

    print(f"Search completed in {search_time * 1000:.2f}ms")
    print(f"Found {len(search_results)} results:")
    print()

    # Display search results
    for i, (doc_id, score, payload) in enumerate(search_results[:5]):
        print(f"  {i + 1}. ID: {doc_id}")
        print(f"     Score: {score:.4f}")
        print(f"     Title: {payload.get('title', 'N/A')}")
        print(f"     Category: {payload.get('category', 'N/A')}")
        print()

    # Test batch search
    print("🔍 Testing batch search...")
    batch_queries = data_generator.generate_text_like_vectors(
        n_samples=5, dimension=config["dimension"]
    )

    batch_start = time.time()
    batch_results = []
    for query in batch_queries:
        results = db.search(query, limit=5)
        batch_results.append(results)
    batch_time = time.time() - batch_start

    print(f"Batch search completed:")
    print(f"  Queries: {len(batch_queries)}")
    print(f"  Total time: {batch_time * 1000:.2f}ms")
    print(f"  Average per query: {batch_time * 1000 / len(batch_queries):.2f}ms")
    print(f"  QPS: {len(batch_queries) / batch_time:.1f}")
    print()

    # Test filtering
    print("🎯 Testing filtered search...")
    filter_results = db.search(
        query_vector=query_vector,
        limit=10,
        filter_params={"must": [{"key": "category", "match": {"value": "category_1"}}]},
        with_payload=True,
    )

    print(f"Filtered search found {len(filter_results)} results in category_1:")
    for i, (doc_id, score, payload) in enumerate(filter_results[:3]):
        print(f"  {i + 1}. ID: {doc_id}, Score: {score:.4f}")
    print()

    # Performance summary
    print("📊 Performance Summary:")
    print(f"  Dataset size: {config['n_documents']:,} vectors")
    print(f"  Original dimension: {config['dimension']}")
    print(f"  Compressed dimension: 256")
    print(
        f"  Compression ratio: {config['dimension'] * 4 / 16:.1f}x"
    )  # float32 -> 16 bytes
    print(f"  Indexing speed: {len(doc_vectors) / indexing_time:.0f} vectors/sec")
    print(f"  Search latency: {search_time * 1000:.2f}ms")
    print(f"  QPS: {1 / search_time:.1f}")
    print()

    # Close connection
    db.close()

    print("🎉 Index building completed successfully!")
    print("Next steps:")
    print("1. Run example 03 to test advanced search features")
    print("2. Use the QuantumDB API in your own applications")
    print("3. Try different embedding models for real-world data")


if __name__ == "__main__":
    main()
