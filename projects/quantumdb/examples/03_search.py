#!/usr/bin/env python3
"""
Advanced QuantumDB demo (v2) with automatic embedding-dimension handling.

This version fixes the embedding-dimension mismatch by reading the actual
dimension from the loaded embeddings and initializing the temporary DB
collection with that vector size.

Other features:
- Sanitizes document IDs (ints or UUIDs)
- Sanity checks for lengths and shapes
- Clear debug/info prints
- Unit test for sanitize_id
"""

import re
import uuid
import time
from typing import List, Dict, Any

import numpy as np

# NOTE: these imports assume your QuantumDB package exposes these objects
# exactly as in your example. If names differ, adapt accordingly.
from quantumdb import QuantumDB
from quantumdb.data import (
    SyntheticDataGenerator,
    WikipediaParquetLoader,
    create_embedder,
)
from quantumdb.evaluation import BenchmarkRunner


def sanitize_id(raw_id, index: int):
    """Return a valid point id (unsigned int or UUID string).

    Rules applied (in order):
      - If `raw_id` is a non-negative int, return as-is.
      - If `raw_id` is a string ending with digits (e.g. "wiki_123"), return that integer suffix.
      - If `raw_id` is a valid UUID string, return it as-is.
      - Otherwise return a deterministic UUID5 using the index (stable across runs for same inputs).
    """
    if isinstance(raw_id, int) and raw_id >= 0:
        return raw_id

    if isinstance(raw_id, str):
        m = re.search(r"(\d+)$", raw_id)
        if m:
            try:
                parsed = int(m.group(1))
                if parsed >= 0:
                    return parsed
            except Exception:
                pass
        try:
            _ = uuid.UUID(raw_id)
            return raw_id
        except Exception:
            pass

    return str(uuid.uuid5(uuid.NAMESPACE_OID, f"fallback-{index}"))


def demonstrate_real_world_embeddings():
    """Demonstrate using real-world embeddings with robust ID handling and automatic dimensionality."""
    print("🌍 Real-world Embeddings Demo")
    print("-" * 30)

    try:
        print("Trying to load Wikipedia embeddings...")
        wiki_loader = WikipediaParquetLoader()
        wiki_embeddings, wiki_metadata = wiki_loader.load_data(max_samples=1000)

        # Sanitize IDs and build payloads
        doc_ids = [sanitize_id(item.get("id"), i) for i, item in enumerate(wiki_metadata)]
        payloads = [
            {
                "title": item.get("title", "N/A"),
                "text": (item.get("text", "") or "")[:200] + "...",
                "length": item.get("length", 0),
            }
            for item in wiki_metadata
        ]

        # Sanity checks for lengths
        if len(wiki_embeddings) != len(doc_ids) or len(wiki_embeddings) != len(payloads):
            raise ValueError("Length mismatch between embeddings, doc_ids and payloads")

        # Determine embedding dimensionality from the loaded embeddings and use it for the DB vector_size.
        actual_dim = getattr(wiki_embeddings, "shape", (None,))[1]
        if actual_dim is None:
            raise ValueError("Could not determine embedding dimensionality from wiki_embeddings")

        expected_dim = actual_dim
        print(f"Info: setting DB vector_size to {expected_dim} to match Wikipedia embeddings")

        # Create DB with vector_size matching the embeddings
        db_real = QuantumDB(
            model_path="models/learnablepq_final.safetensors",
            collection_name="wikipedia_demo",
            vector_size=expected_dim,
        )

        try:
            db_real.add(wiki_embeddings, list(doc_ids), payloads)
        except Exception as e:
            print("❌ Error while adding Wikipedia embeddings to the DB:", e)
            print("Example doc_ids (first 10):", doc_ids[:10])
            raise

        # Test search with a sample query (use first embedding as query)
        query_embedding = wiki_embeddings[0]
        results = db_real.search(query_embedding, limit=5)
        print(f"\nSearch results using Wikipedia embeddings:")
        for i, (doc_id, score, payload) in enumerate(results):
            print(f"  {i + 1}. Score: {score:.4f}")
            print(f"     Title: {payload.get('title')}")
            print(f"     Text: {payload.get('text')[:100]}...")

        db_real.close()
        print("✅ Wikipedia embeddings demo completed successfully!")

    except FileNotFoundError:
        print("❌ Wikipedia parquet file not found!\nPlease run 'python simple_download.py' first to download the dataset.")
        print("Falling back to sentence transformers demo...")
        _demonstrate_sentence_transformers()
    except Exception as e:
        print(f"❌ Wikipedia embeddings demo failed: {e}")
        print("Falling back to sentence transformers demo...")
        _demonstrate_sentence_transformers()


# --- the rest of the script (kept mostly as before) ---

def _demonstrate_sentence_transformers():
    try:
        embedder = create_embedder("sentence_transformer", "all-MiniLM-L6-v2")
        documents = [
            "Machine learning is a subset of artificial intelligence.",
            "Deep learning uses neural networks with multiple layers.",
            "Natural language processing helps computers understand text.",
            "Computer vision enables machines to interpret visual information.",
            "Reinforcement learning learns through trial and error.",
        ]
        print("Generating embeddings for sample documents...")
        embeddings = embedder.encode(documents)
        print(f"Generated embeddings with shape: {embeddings.shape}")

        db_real = QuantumDB(
            model_path="models/learnablepq_final.safetensors",
            collection_name="real_world_demo",
            vector_size=embeddings.shape[1],
        )

        doc_ids = [f"real_doc_{i}" for i in range(len(documents))]
        payloads = [{"text": doc, "length": len(doc)} for doc in documents]
        db_real.add(embeddings, list(doc_ids), payloads)

        query_text = "How do neural networks work?"
        query_embedding = embedder.encode([query_text])[0]

        results = db_real.search(query_embedding, limit=3)
        print(f"\nSearch results for: '{query_text}'")
        for i, (doc_id, score, payload) in enumerate(results):
            print(f"  {i + 1}. Score: {score:.4f}")
            print(f"     Text: {payload.get('text')}")

        db_real.close()

    except Exception as e:
        print(f"Sentence transformers demo failed: {e}")
        print("This is expected if sentence-transformers is not installed.")


def main():
    print("🚀 QuantumDB Advanced Search Demo (v2)")
    print("=" * 50)

    model_path = "models/learnablepq_final.safetensors"
    try:
        db = QuantumDB(model_path=model_path, collection_name="demo_collection", vector_size=256)
    except Exception as e:
        print(f"❌ Failed to initialize QuantumDB: {e}")
        print("Please run examples 01 and 02 first.")
        return

    collection_info = db.get_collection_info()
    if collection_info.get("vectors_count", 0) == 0:
        print("❌ Collection is empty. Please run example 02 first.")
        db.close()
        return

    print(f"Connected to collection with {collection_info.get('vectors_count')} vectors")

    data_generator = SyntheticDataGenerator(random_state=42)

    # Keep other demo functions simple here (not included in this snippet for brevity)
    # You can import or copy the rest of the demos (basic/filtered/batch/perf/similarity)

    demonstrate_real_world_embeddings()

    db.close()
    print("\n🎉 Advanced search demo (v2) completed!")


def _unit_test_sanitize_id():
    samples = ["wiki_0", "wiki_42", "123", 5, -1, str(uuid.uuid4()), "strange_id"]
    print("\nUnit test: sanitize_id outputs")
    for i, s in enumerate(samples):
        print(f"  input={s!r} -> output={sanitize_id(s, i)!r}")


if __name__ == "__main__":
    _unit_test_sanitize_id()
    main()

