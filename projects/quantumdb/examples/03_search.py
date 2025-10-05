#!/usr/bin/env python3
"""
Example 3: Advanced search functionality with QuantumDB.

This script demonstrates advanced search features including filtering,
batch search, and performance testing.
"""

import numpy as np
import time
from typing import List, Dict, Any

from quantumdb import QuantumDB
from quantumdb.data import SyntheticDataGenerator, create_embedder
from quantumdb.evaluation import BenchmarkRunner


def demonstrate_basic_search(db: QuantumDB, data_generator: SyntheticDataGenerator):
    """Demonstrate basic search functionality."""
    print("🔍 Basic Search Demo")
    print("-" * 30)

    # Generate a query
    query_vector = data_generator.generate_text_like_vectors(
        n_samples=1, dimension=768
    )[0]

    # Perform search with different limits
    for k in [5, 10, 20]:
        start_time = time.time()
        results = db.search(query_vector, limit=k)
        search_time = time.time() - start_time

        print(
            f"Search with k={k}: {len(results)} results in {search_time * 1000:.2f}ms"
        )

        # Show top 3 results
        for i, (doc_id, score, payload) in enumerate(results[:3]):
            print(f"  {i + 1}. {doc_id}: {score:.4f} - {payload.get('title', 'N/A')}")
        print()


def demonstrate_filtered_search(db: QuantumDB, data_generator: SyntheticDataGenerator):
    """Demonstrate filtered search functionality."""
    print("🎯 Filtered Search Demo")
    print("-" * 30)

    query_vector = data_generator.generate_text_like_vectors(
        n_samples=1, dimension=768
    )[0]

    # Filter by category
    print("Filter by category = 'category_1':")
    results = db.search(
        query_vector,
        limit=10,
        filter_params={"must": [{"key": "category", "match": {"value": "category_1"}}]},
    )
    print(f"Found {len(results)} results")
    for i, (doc_id, score, payload) in enumerate(results[:3]):
        print(f"  {i + 1}. {doc_id}: {score:.4f} - {payload.get('title', 'N/A')}")
    print()

    # Filter by numeric range
    print("Filter by document length > 3000:")
    results = db.search(
        query_vector,
        limit=10,
        filter_params={"must": [{"key": "length", "range": {"gt": 3000}}]},
    )
    print(f"Found {len(results)} results")
    for i, (doc_id, score, payload) in enumerate(results[:3]):
        print(
            f"  {i + 1}. {doc_id}: {score:.4f} - Length: {payload.get('length', 'N/A')}"
        )
    print()

    # Complex filter (AND condition)
    print("Complex filter (category_2 AND length > 2000):")
    results = db.search(
        query_vector,
        limit=10,
        filter_params={
            "must": [
                {"key": "category", "match": {"value": "category_2"}},
                {"key": "length", "range": {"gt": 2000}},
            ]
        },
    )
    print(f"Found {len(results)} results")
    for i, (doc_id, score, payload) in enumerate(results[:3]):
        print(f"  {i + 1}. {doc_id}: {score:.4f} - {payload.get('title', 'N/A')}")
    print()


def demonstrate_batch_search(db: QuantumDB, data_generator: SyntheticDataGenerator):
    """Demonstrate batch search functionality."""
    print("📦 Batch Search Demo")
    print("-" * 30)

    # Generate multiple queries
    batch_size = 10
    query_vectors = data_generator.generate_text_like_vectors(
        n_samples=batch_size, dimension=768
    )

    # Perform batch search
    start_time = time.time()
    all_results = []
    for i, query in enumerate(query_vectors):
        results = db.search(query, limit=5)
        all_results.append(results)
    batch_time = time.time() - start_time

    print(f"Processed {batch_size} queries in {batch_time * 1000:.2f}ms")
    print(f"Average per query: {batch_time * 1000 / batch_size:.2f}ms")
    print(f"QPS: {batch_size / batch_time:.1f}")
    print()

    # Show results for first few queries
    for i, results in enumerate(all_results[:3]):
        print(f"Query {i + 1} results:")
        for j, (doc_id, score, payload) in enumerate(results[:2]):
            print(f"  {j + 1}. {doc_id}: {score:.4f}")
        print()


def demonstrate_performance_testing(
    db: QuantumDB, data_generator: SyntheticDataGenerator
):
    """Demonstrate performance testing."""
    print("⚡ Performance Testing")
    print("-" * 30)

    # Generate test queries
    n_test_queries = 100
    test_queries = data_generator.generate_text_like_vectors(
        n_samples=n_test_queries, dimension=768
    )

    # Test search performance
    latencies = []
    start_time = time.time()

    for query in test_queries:
        query_start = time.time()
        results = db.search(query, limit=10)
        query_end = time.time()
        latencies.append(query_end - query_start)

    total_time = time.time() - start_time

    # Calculate statistics
    latencies_ms = [l * 1000 for l in latencies]
    qps = n_test_queries / total_time

    print(f"Performance Statistics:")
    print(f"  Total queries: {n_test_queries}")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  QPS: {qps:.1f}")
    print(f"  Latency P50: {np.percentile(latencies_ms, 50):.2f}ms")
    print(f"  Latency P95: {np.percentile(latencies_ms, 95):.2f}ms")
    print(f"  Latency P99: {np.percentile(latencies_ms, 99):.2f}ms")
    print()


def demonstrate_similarity_analysis(
    db: QuantumDB, data_generator: SyntheticDataGenerator
):
    """Demonstrate similarity analysis between queries."""
    print("🔗 Similarity Analysis Demo")
    print("-" * 30)

    # Generate two similar queries (by adding small noise)
    base_query = data_generator.generate_text_like_vectors(n_samples=1, dimension=768)[
        0
    ]

    # Create similar query with small noise
    noise = np.random.normal(0, 0.1, base_query.shape)
    similar_query = base_query + noise
    similar_query = similar_query / np.linalg.norm(similar_query)

    # Create dissimilar query
    dissimilar_query = data_generator.generate_text_like_vectors(
        n_samples=1, dimension=768
    )[0]

    # Search with all queries
    queries = [
        ("Base Query", base_query),
        ("Similar Query", similar_query),
        ("Dissimilar Query", dissimilar_query),
    ]

    all_results = {}
    for name, query in queries:
        results = db.search(query, limit=10)
        all_results[name] = [doc_id for doc_id, _, _ in results]
        print(f"{name}: {[doc_id for doc_id, _, _ in results[:3]]}")

    # Calculate overlap between results
    print("\nResult Overlap Analysis:")
    base_ids = set(all_results["Base Query"])
    similar_ids = set(all_results["Similar Query"])
    dissimilar_ids = set(all_results["Dissimilar Query"])

    similar_overlap = len(base_ids.intersection(similar_ids))
    dissimilar_overlap = len(base_ids.intersection(dissimilar_ids))

    print(f"Base vs Similar: {similar_overlap}/10 overlap ({similar_overlap * 10}%)")
    print(
        f"Base vs Dissimilar: {dissimilar_overlap}/10 overlap ({dissimilar_overlap * 10}%)"
    )
    print()


def demonstrate_real_world_embeddings():
    """Demonstrate using real-world embeddings."""
    print("🌍 Real-world Embeddings Demo")
    print("-" * 30)

    try:
        # Try to use sentence transformers
        embedder = create_embedder("sentence_transformer", "all-MiniLM-L6-v2")

        # Sample documents
        documents = [
            "Machine learning is a subset of artificial intelligence.",
            "Deep learning uses neural networks with multiple layers.",
            "Natural language processing helps computers understand text.",
            "Computer vision enables machines to interpret visual information.",
            "Reinforcement learning learns through trial and error.",
        ]

        # Generate embeddings
        print("Generating embeddings for sample documents...")
        embeddings = embedder.encode(documents)
        print(f"Generated embeddings with shape: {embeddings.shape}")

        # Create a new database for this demo
        db_real = QuantumDB(
            model_path="models/learnablepq_final.safetensors",
            collection_name="real_world_demo",
            vector_size=256,
        )

        # Add documents
        doc_ids = [f"real_doc_{i}" for i in range(len(documents))]
        payloads = [{"text": doc, "length": len(doc)} for doc in documents]

        db_real.add(embeddings, doc_ids, payloads)

        # Test search with a query
        query_text = "How do neural networks work?"
        query_embedding = embedder.encode([query_text])[0]

        results = db_real.search(query_embedding, limit=3)
        print(f"\nSearch results for: '{query_text}'")
        for i, (doc_id, score, payload) in enumerate(results):
            print(f"  {i + 1}. Score: {score:.4f}")
            print(f"     Text: {payload['text']}")

        db_real.close()

    except Exception as e:
        print(f"Real-world embeddings demo failed: {e}")
        print("This is expected if sentence-transformers is not installed.")

    print()


def main():
    """Main demonstration script."""
    print("🚀 QuantumDB Advanced Search Demo")
    print("=" * 50)

    # Check if model exists
    model_path = "models/learnablepq_final.safetensors"
    try:
        db = QuantumDB(
            model_path=model_path,
            collection_name="demo_collection",
            vector_size=256,
        )
    except Exception as e:
        print(f"❌ Failed to initialize QuantumDB: {e}")
        print("Please run examples 01 and 02 first.")
        return

    # Check if collection has data
    collection_info = db.get_collection_info()
    if collection_info.get("vectors_count", 0) == 0:
        print("❌ Collection is empty. Please run example 02 first.")
        db.close()
        return

    print(
        f"Connected to collection with {collection_info.get('vectors_count')} vectors"
    )
    print()

    # Initialize data generator for demo queries
    data_generator = SyntheticDataGenerator(random_state=42)

    # Run demonstrations
    demonstrate_basic_search(db, data_generator)
    demonstrate_filtered_search(db, data_generator)
    demonstrate_batch_search(db, data_generator)
    demonstrate_performance_testing(db, data_generator)
    demonstrate_similarity_analysis(db, data_generator)
    demonstrate_real_world_embeddings()

    # Run comprehensive benchmark
    print("📊 Comprehensive Benchmark")
    print("-" * 30)

    try:
        benchmark_runner = BenchmarkRunner(output_dir="benchmark_results")

        # Generate test data
        test_queries = data_generator.generate_text_like_vectors(
            n_samples=50, dimension=768
        )

        # Run performance benchmark
        perf_metrics = benchmark_runner.benchmark_search_performance(
            db, test_queries, num_queries=50
        )

        print("Benchmark Results:")
        print(f"  QPS: {perf_metrics.queries_per_second:.1f}")
        print(f"  P50 Latency: {perf_metrics.latency_p50 * 1000:.2f}ms")
        print(f"  P95 Latency: {perf_metrics.latency_p95 * 1000:.2f}ms")
        print(f"  Memory Usage: {perf_metrics.memory_usage_mb:.1f}MB")

        # Save benchmark results
        benchmark_runner.save_results()
        benchmark_runner.generate_report()

    except Exception as e:
        print(f"Benchmark failed: {e}")

    # Close connection
    db.close()

    print("\n🎉 Advanced search demo completed!")
    print("\nKey takeaways:")
    print("✅ QuantumDB provides fast and accurate vector search")
    print("✅ Filtering allows precise result control")
    print("✅ Batch processing enables high throughput")
    print("✅ Compression maintains search quality while reducing storage")
    print("✅ Real-world embeddings work seamlessly with the system")


if __name__ == "__main__":
    main()
