"""
Tests for QuantumDB API.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil
import torch
from typing import List, Union

# Import modules to test
from quantumdb.api import QuantumDB
from quantumdb.training.model import LearnablePQ


class TestQuantumDB:
    """Test cases for QuantumDB API."""

    def setup_method(self):
        """Setup test fixtures."""
        # Create temporary directory for test data
        self.temp_dir = tempfile.mkdtemp()

        # Create a real compression model for testing
        self.model = LearnablePQ(
            input_dim=768,
            target_dim=256,
            n_subvectors=16,
            codebook_size=256,
        )

        # Save model to temporary file
        model_path = Path(self.temp_dir) / "test_model.safetensors"
        from safetensors.torch import save_file

        save_file(self.model.state_dict(), model_path)

        # Create metadata file
        metadata_path = model_path.with_suffix(".json")
        import json

        with open(metadata_path, "w") as f:
            json.dump(
                {
                    "model_config": {
                        "input_dim": 768,
                        "target_dim": 256,
                        "n_subvectors": 16,
                        "codebook_size": 256,
                    }
                },
                f,
            )

        self.model_path = str(model_path)

        # Initialize QuantumDB with in-memory Qdrant
        self.db = QuantumDB(
            model_path=None,  # Start without model for basic tests
            collection_name="test_collection",
            vector_size=256,
            qdrant_host="localhost",
            qdrant_port=6333,
        )

    def teardown_method(self):
        """Cleanup test fixtures."""
        # Clean up temporary directory
        shutil.rmtree(self.temp_dir, ignore_errors=True)

        # Close database connection
        if hasattr(self, "db"):
            self.db.close()

    def test_initialization_without_model(self):
        """Test initialization without model."""
        db = QuantumDB(
            model_path=None,
            collection_name="test_collection_init",
            vector_size=256,
        )

        assert db.collection_name == "test_collection_init"
        assert db.vector_size == 256
        assert db.model is None

        db.close()

    def test_initialization_with_model(self):
        """Test initialization with model."""
        db = QuantumDB(
            model_path=self.model_path,
            collection_name="test_collection_model",
            vector_size=256,
        )

        assert db.collection_name == "test_collection_model"
        assert db.model is not None
        assert db.model.get_compression_ratio() > 0

        db.close()

    def test_add_vectors_without_model(self):
        """Test adding vectors without compression model."""
        vectors = np.random.randn(100, 256).astype(np.float32)
        ids: List[Union[str, int]] = list(range(100))

        result = self.db.add(vectors, ids)

        assert result["status"] == "success"
        assert result["vectors_added"] == 100
        assert result["collection"] == "test_collection"

    def test_add_vectors_with_model(self):
        """Test adding vectors with compression model."""
        # Load model
        self.db.load_model(self.model_path)

        vectors = np.random.randn(100, 768).astype(np.float32)
        ids = list(range(100))  # type: ignore

        result = self.db.add(vectors, ids)

        assert result["status"] == "success"
        assert result["vectors_added"] == 100

    def test_search_without_model(self):
        """Test search without compression model."""
        # First add some vectors
        vectors = np.random.randn(10, 256).astype(np.float32)
        ids = list(range(10))
        payloads = [{"title": f"Document {i}"} for i in range(10)]

        self.db.add(vectors, ids, payloads)

        # Search
        query_vector = vectors[0]  # Use first vector as query
        results = self.db.search(query_vector, limit=5)

        assert len(results) > 0
        doc_id, score, payload = results[0]
        assert doc_id in ids
        assert isinstance(score, float)
        assert "title" in payload

    def test_search_with_model(self):
        """Test search with compression model."""
        # Load model
        self.db.load_model(self.model_path)

        # Add some vectors
        vectors = np.random.randn(10, 768).astype(np.float32)
        ids = list(range(10))
        payloads = [{"title": f"Document {i}"} for i in range(10)]

        self.db.add(vectors, ids, payloads)

        # Search
        query_vector = vectors[0]  # Use first vector as query
        results = self.db.search(query_vector, limit=5)

        assert len(results) > 0
        doc_id, score, payload = results[0]
        assert doc_id in ids
        assert isinstance(score, float)

    def test_delete_by_ids(self):
        """Test deleting vectors by IDs."""
        # Add some vectors first
        vectors = np.random.randn(10, 256).astype(np.float32)
        ids = list(range(10))

        self.db.add(vectors, ids)

        # Delete some vectors
        ids_to_delete = [0, 1, 2]
        result = self.db.delete(ids=ids_to_delete)

        assert result["status"] == "success"
        assert result["deleted_count"] == 3
        assert result["collection"] == "test_collection"

    def test_delete_by_filter(self):
        """Test deleting vectors by filter."""
        # Add some vectors with payloads
        vectors = np.random.randn(10, 256).astype(np.float32)
        ids = list(range(10))
        payloads = [{"category": "test" if i < 5 else "other"} for i in range(10)]

        self.db.add(vectors, ids, payloads)

        # Delete by filter
        filter_params = {"must": [{"key": "category", "match": {"value": "test"}}]}
        result = self.db.delete(filter_params=filter_params)

        assert result["status"] == "success"
        assert result["deleted_count"] == "unknown"

    def test_get_collection_info(self):
        """Test getting collection info."""
        # Add some vectors
        vectors = np.random.randn(10, 256).astype(np.float32)
        ids = list(range(10))
        self.db.add(vectors, ids)

        info = self.db.get_collection_info()

        assert info["name"] == "test_collection"
        assert info["vectors_count"] >= 10
        assert info["vector_size"] == 256
        assert info["distance"] in [
            "COSINE",
            "EUCLID",
            "DOT",
            "Cosine",
            "Euclid",
            "Dot",
        ]

    def test_count_vectors(self):
        """Test counting vectors."""
        # Add some vectors
        vectors = np.random.randn(10, 256).astype(np.float32)
        ids = list(range(10))
        self.db.add(vectors, ids)

        count = self.db.count_vectors()
        assert count >= 10

    def test_recreate_collection(self):
        """Test recreating collection."""
        # Add some vectors
        vectors = np.random.randn(10, 256).astype(np.float32)
        ids = list(range(10))
        self.db.add(vectors, ids)

        # Recreate collection
        self.db.recreate_collection()

        # Count should be 0 after recreation
        count = self.db.count_vectors()
        assert count == 0

    def test_context_manager(self):
        """Test using QuantumDB as context manager."""
        with QuantumDB(collection_name="test_context") as db:
            assert db.collection_name == "test_context"

            # Add some vectors
            vectors = np.random.randn(5, 256).astype(np.float32)
            ids = list(range(5))
            db.add(vectors, ids)

            count = db.count_vectors()
            assert count == 5

    def test_encode_vectors(self):
        """Test encoding vectors with model."""
        # Load model
        self.db.load_model(self.model_path)

        vectors = np.random.randn(10, 768).astype(np.float32)
        encoded = self.db.encode_vectors(vectors)

        assert encoded.shape == (10, 256)
        assert np.isfinite(encoded).all()

    def test_error_handling_add_vectors_dimension_mismatch(self):
        """Test error handling when adding vectors with wrong dimensions."""
        # Wrong dimension vectors
        vectors = np.random.randn(100, 128).astype(np.float32)  # Should be 256
        ids = list(range(100))

        with pytest.raises(ValueError):
            self.db.add(vectors, ids)

    def test_error_handling_search_dimension_mismatch(self):
        """Test error handling during search with wrong dimensions."""
        # Wrong dimension query
        query_vector = np.random.randn(128).astype(np.float32)  # Should be 256

        with pytest.raises(ValueError):
            self.db.search(query_vector)

    def test_error_handling_encode_without_model(self):
        """Test error encoding vectors without model."""
        vectors = np.random.randn(10, 768).astype(np.float32)

        with pytest.raises(ValueError, match="No model loaded"):
            self.db.encode_vectors(vectors)

    def test_model_loading_nonexistent_file(self):
        """Test loading model from nonexistent file."""
        with pytest.raises(FileNotFoundError):
            self.db.load_model("nonexistent_model.safetensors")


if __name__ == "__main__":
    pytest.main([__file__])
