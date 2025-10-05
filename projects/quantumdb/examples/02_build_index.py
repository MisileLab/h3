#!/usr/bin/env python3
"""
Example 2 (patched): Building a vector index with QuantumDB.

This full script includes a fix for the "invalid point ID" error by normalizing
IDs to unsigned integers (and preserving original IDs in payload under "orig_id").
It also corrects the compression-ratio calculation and adds a few safety checks.

Drop-in replacement for your original example.
"""

import time
from pathlib import Path
import uuid
import numpy as np

from quantumdb import QuantumDB
from quantumdb.data import (
    SyntheticDataGenerator,
    WikipediaParquetLoader,
    create_embedder,
)


def normalize_ids_for_db(ids, payloads):
    """
    Convert ids to DB-compatible IDs (unsigned integers).
    If an id is already a non-negative integer, use it.
    Otherwise assign a sequential integer mapping (0..N-1) and
    preserve the original id inside the payload as payload['orig_id'].

    Returns (new_ids, new_payloads).
    """
    if ids is None:
        raise ValueError("ids must be provided to normalize_ids_for_db")

    # Ensure payloads list matches ids length (create empty dicts if None)
    if payloads is None:
        payloads = [{} for _ in ids]
    else:
        # Pad or trim payloads to match ids length
        if len(payloads) < len(ids):
            payloads = list(payloads) + [{} for _ in range(len(ids) - len(payloads))]
        elif len(payloads) > len(ids):
            payloads = list(payloads)[: len(ids)]

    new_ids = []
    new_payloads = []

    for i, (pid, pl) in enumerate(zip(ids, payloads)):
        pl = dict(pl) if pl is not None else {}

        # If pid is int-like and non-negative, use it
        if isinstance(pid, int) and pid >= 0:
            new_id = pid
        else:
            # Try to convert numeric strings like "123" -> int(123)
            try:
                if isinstance(pid, str) and pid.isdigit():
                    new_id = int(pid)
                else:
                    # Fallback: assign sequential int index
                    new_id = i
            except Exception:
                new_id = i

        # Preserve original id for lookups later
        pl["orig_id"] = pid
        new_ids.append(new_id)
        new_payloads.append(pl)

    return new_ids, new_payloads


def safe_cast_wiki_ids(wiki_metadata):
    """
    Attempt to extract numeric IDs from wikipedia metadata.
    If ID cannot be cast to int, leave as-is (normalize step will map it).
    """
    ids = []
    for item in wiki_metadata:
        raw = item.get("id", None)
        try:
            if raw is None:
                ids.append(None)
            elif isinstance(raw, int):
                ids.append(raw)
            elif isinstance(raw, str) and raw.isdigit():
                ids.append(int(raw))
            else:
                ids.append(raw)
        except Exception:
            ids.append(raw)
    return ids


def main():
    """Main index building script (full)."""
    print("🏗️  Building Vector Index with QuantumDB (patched)")
    print("=" * 60)

    # Configuration
    config = {
        "model_path": "models/learnablepq_final.safetensors",
        "collection_name": "demo_collection",
        "n_documents": 10000,
        "dimension": 768,
        "batch_size": 1000,
        "use_real_data": True,
        # If you prefer UUIDs instead of integer mapping, set to True.
        # (This script defaults to integer mapping for simplicity.)
        "use_uuids_for_points": False,
        # Compressed vector settings used for reporting compression ratio:
        # set bytes_per_compressed_element to the bytes used to store each
        # component of the compressed vector in your DB (adjust if needed).
        "compressed_dimension": 256,
        "bytes_per_original_element": 4,  # float32
        "bytes_per_compressed_element": 1,  # e.g., 1 byte per codebook index (adjust to your system)
    }

    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()

    # Check if model exists
    model_path = Path(config["model_path"])
    if not model_path.exists():
        print(f"❌ Model file not found: {model_path}")
        print("Please run example 01 first to train a model.")
        return

    # Load document data
    doc_vectors = None
    doc_metadata = None
    doc_ids = None
    data_generator = SyntheticDataGenerator(random_state=42)  # Initialize for fallback

    if config["use_real_data"]:
        print("📄 Loading real Wikipedia document data...")
        try:
            wiki_loader = WikipediaParquetLoader()
            doc_vectors, wiki_metadata = wiki_loader.load_data(
                max_samples=config["n_documents"]
            )

            # If loader returned metadata, attempt to extract numeric IDs where possible
            doc_ids = safe_cast_wiki_ids(wiki_metadata)

            # Build payloads with truncated text and other fields
            doc_metadata = []
            for i, item in enumerate(wiki_metadata):
                text = item.get("text", "") or ""
                title = item.get("title", "") or f"wiki_{i}"
                length = item.get("length", len(text))
                doc_metadata.append(
                    {
                        "title": title,
                        "text": (text[:500] + "...") if len(text) > 500 else text,
                        "category": f"wikipedia_{i % 20}",
                        "length": length,
                        "timestamp": time.time() - np.random.randint(0, 86400 * 30),
                    }
                )

            print(f"Loaded {len(doc_vectors)} real Wikipedia documents")
            print(f"Vector dimension: {doc_vectors.shape[1]}")
        except FileNotFoundError:
            print("❌ Wikipedia parquet file not found!")
            print("Please run 'python simple_download.py' first to download the dataset.")
            print("Falling back to synthetic data...")
            config["use_real_data"] = False
        except Exception as e:
            print(f"❌ Error loading Wikipedia data: {e}")
            print("Falling back to synthetic data...")
            config["use_real_data"] = False

    if doc_vectors is None:
        print("📄 Generating synthetic document data...")

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
                "length": int(np.random.randint(100, 5000)),
                "timestamp": time.time() - np.random.randint(0, 86400 * 30),
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
        vector_size=config["compressed_dimension"],  # Compressed dimension
        distance="cosine",
    )

    # Get collection info
    collection_info = db.get_collection_info()
    print("Collection info:")
    print(f"  Name: {collection_info.get('name')}")
    print(f"  Vector size: {collection_info.get('vector_size')}")
    print(f"  Distance: {collection_info.get('distance')}")
    print()

    # Normalize IDs to DB-compatible values (and preserve original IDs)
    if config["use_uuids_for_points"]:
        # Use UUIDs as point IDs (string form). Store original id in payload['orig_id'].
        print("🔁 Converting point IDs to UUIDs (preserving original IDs in payload)...")
        new_ids = []
        new_payloads = []
        for pid, pl in zip(doc_ids, doc_metadata or [{}] * len(doc_ids)):
            pl = dict(pl) if pl is not None else {}
            new_id = str(uuid.uuid4())
            pl["orig_id"] = pid
            new_ids.append(new_id)
            new_payloads.append(pl)
        doc_ids = new_ids
        doc_metadata = new_payloads
    else:
        print("🔁 Normalizing point IDs to unsigned integers (preserving original IDs in payload)...")
        doc_ids, doc_metadata = normalize_ids_for_db(doc_ids, doc_metadata)

    # Sanity checks
    assert len(doc_vectors) == len(doc_ids) == len(doc_metadata), (
        "Vectors, ids, and metadata must have the same length. "
        f"len(vectors)={len(doc_vectors)}, len(ids)={len(doc_ids)}, len(metadata)={len(doc_metadata)}"
    )

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
    vectors_added = result.get("vectors_added", len(doc_vectors))

    print("✅ Indexing completed!")
    print(f"  Vectors added: {vectors_added}")
    print(f"  Indexing time: {indexing_time:.2f} seconds")
    if indexing_time > 0:
        print(f"  Indexing speed: {int(len(doc_vectors) / indexing_time)} vectors/second")
    print()

    # Get final collection info
    final_info = db.get_collection_info()
    print("Final collection info:")
    print(f"  Total vectors: {final_info.get('vectors_count')}")
    print()

    # Test search functionality
    print("🔍 Testing search functionality...")

    # Generate a test query (same dimensionality as original embedder input)
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

    # Display top-5 search results (handles payload possibly being None)
    for i, item in enumerate(search_results[:5]):
        # The original example looped over tuples (doc_id, score, payload)
        # But different DB wrappers may return different structures. We try to handle common forms.
        try:
            doc_id, score, payload = item
        except Exception:
            # fallback: assume dict with keys
            if isinstance(item, dict):
                doc_id = item.get("id") or item.get("point_id") or item.get("pointId")
                score = item.get("score")
                payload = item.get("payload", {})
            else:
                # best-effort extraction
                doc_id = getattr(item, "id", "<unknown>")
                score = getattr(item, "score", 0.0)
                payload = getattr(item, "payload", {}) or {}

        print(f"  {i + 1}. ID: {doc_id}")
        print(f"     Score: {score:.4f}")
        print(f"     Title: {payload.get('title', payload.get('orig_id', 'N/A'))}")
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

    print("Batch search completed:")
    print(f"  Queries: {len(batch_queries)}")
    print(f"  Total time: {batch_time * 1000:.2f}ms")
    if len(batch_queries) > 0:
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
    for i, item in enumerate(filter_results[:3]):
        try:
            doc_id, score, payload = item
        except Exception:
            if isinstance(item, dict):
                doc_id = item.get("id")
                score = item.get("score")
                payload = item.get("payload", {})
            else:
                doc_id = getattr(item, "id", "<unknown>")
                score = getattr(item, "score", 0.0)
                payload = getattr(item, "payload", {}) or {}
        print(f"  {i + 1}. ID: {doc_id}, Score: {score:.4f}")
    print()

    # Performance summary
    print("📊 Performance Summary:")
    print(f"  Dataset size: {len(doc_vectors):,} vectors")
    print(f"  Original dimension: {config['dimension']}")
    print(f"  Compressed dimension: {config['compressed_dimension']}")

    # Compute compression ratio using configured bytes-per-element values.
    orig_bytes = config["dimension"] * config["bytes_per_original_element"]
    comp_bytes = config["compressed_dimension"] * config["bytes_per_compressed_element"]
    if comp_bytes > 0:
        compression_ratio = orig_bytes / comp_bytes
        print(f"  Compression ratio: {compression_ratio:.1f}x  (orig bytes: {orig_bytes}, compressed bytes: {comp_bytes})")
    else:
        print("  Compression ratio: N/A (check bytes_per_compressed_element)")

    if indexing_time > 0:
        print(f"  Indexing speed: {int(len(doc_vectors) / indexing_time)} vectors/sec")
    print(f"  Search latency: {search_time * 1000:.2f}ms")
    if search_time > 0:
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

