"""
Tests for QuantumDB API.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from pathlib import Path

# Mock the dependencies that might not be available
import sys
from unittest.mock import MagicMock

# Mock torch and qdrant_client
sys.modules["torch"] = MagicMock()
sys.modules["safetensors"] = MagicMock()
sys.modules["safetensors.torch"] = MagicMock()
sys.modules["qdrant_client"] = MagicMock()
sys.modules["qdrant_client.http"] = MagicMock()
sys.modules["qdrant_client.http.models"] = MagicMock()

# Now import the modules to test
from quantumdb.api import QuantumDB


class TestQuantumDB:
    """Test cases for QuantumDB API."""

    def setup_method(self):
        """Setup test fixtures."""
        # Mock QdrantClient
        self.mock_client = Mock()

        # Mock model loading
        self.mock_model = Mock()
        self.mock_model.get_compression_ratio.return_value = 24.0

        with patch("quantumdb.api.QdrantClient", return_value=self.mock_client):
            with patch("quantumdb.api.load_file", return_value={}):
                with patch("quantumdb.api.LearnablePQ", return_value=self.mock_model):
                    self.db = QuantumDB(
                        model_path=None,  # Skip model loading for basic tests
                        collection_name="test_collection",
                        vector_size=256,
                    )

    def test_initialization_without_model(self):
        """Test initialization without model."""
        with patch("quantumdb.api.QdrantClient", return_value=self.mock_client):
            db = QuantumDB(
                model_path=None,
                collection_name="test_collection",
                vector_size=256,
            )

            assert db.collection_name == "test_collection"
            assert db.vector_size == 256
            assert db.model is None

    def test_initialization_with_model(self):
        """Test initialization with model."""
        mock_model = Mock()
        mock_model.get_compression_ratio.return_value = 24.0

        with patch("quantumdb.api.QdrantClient", return_value=self.mock_client):
            with patch("quantumdb.api.load_file", return_value={}):
                with patch("quantumdb.api.LearnablePQ", return_value=mock_model):
                    with patch("pathlib.Path.exists", return_value=True):
                        db = QuantumDB(
                            model_path="test_model.safetensors",
                            collection_name="test_collection",
                            vector_size=256,
                        )

                        assert db.model == mock_model

    def test_add_vectors_without_model(self):
        """Test adding vectors without compression model."""
        vectors = np.random.randn(100, 256).astype(np.float32)
        ids = list(range(100))

        # Mock the upsert response
        self.mock_client.upsert.return_value = None

        result = self.db.add(vectors, ids)

        assert result["status"] == "success"
        assert result["vectors_added"] == 100
        assert result["collection"] == "test_collection"

        # Check that upsert was called
        self.mock_client.upsert.assert_called_once()

    def test_add_vectors_with_model(self):
        """Test adding vectors with compression model."""
        # Setup model mock
        self.db.model = self.mock_model
        self.mock_model.encode.return_value = np.random.randn(100, 256).astype(
            np.float32
        )

        vectors = np.random.randn(100, 768).astype(np.float32)
        ids = list(range(100))

        # Mock the upsert response
        self.mock_client.upsert.return_value = None

        result = self.db.add(vectors, ids)

        assert result["status"] == "success"
        assert result["vectors_added"] == 100

        # Check that model.encode was called
        self.mock_model.encode.assert_called_once()

    def test_search_without_model(self):
        """Test search without compression model."""
        query_vector = np.random.randn(256).astype(np.float32)

        # Mock search response
        mock_hit = Mock()
        mock_hit.id = "doc_1"
        mock_hit.score = 0.95
        mock_hit.payload = {"title": "Test Document"}

        self.mock_client.search.return_value = [mock_hit]

        results = self.db.search(query_vector, limit=10)

        assert len(results) == 1
        doc_id, score, payload = results[0]
        assert doc_id == "doc_1"
        assert score == 0.95
        assert payload["title"] == "Test Document"

        # Check that search was called
        self.mock_client.search.assert_called_once()

    def test_search_with_model(self):
        """Test search with compression model."""
        # Setup model mock
        self.db.model = self.mock_model
        self.mock_model.encode.return_value = np.random.randn(256).astype(np.float32)

        query_vector = np.random.randn(768).astype(np.float32)

        # Mock search response
        mock_hit = Mock()
        mock_hit.id = "doc_1"
        mock_hit.score = 0.95
        mock_hit.payload = {"title": "Test Document"}

        self.mock_client.search.return_value = [mock_hit]

        results = self.db.search(query_vector, limit=10)

        assert len(results) == 1

        # Check that model.encode was called
        self.mock_model.encode.assert_called_once()

    def test_delete_by_ids(self):
        """Test deleting vectors by IDs."""
        ids = ["doc_1", "doc_2", "doc_3"]

        # Mock delete response
        self.mock_client.delete.return_value = None

        result = self.db.delete(ids=ids)

        assert result["status"] == "success"
        assert result["deleted_count"] == 3
        assert result["collection"] == "test_collection"

        # Check that delete was called
        self.mock_client.delete.assert_called_once()

    def test_delete_by_filter(self):
        """Test deleting vectors by filter."""
        filter_params = {"must": [{"key": "category", "match": {"value": "test"}}]}

        # Mock delete response
        self.mock_client.delete.return_value = None

        result = self.db.delete(filter_params=filter_params)

        assert result["status"] == "success"
        assert result["deleted_count"] == "unknown"

        # Check that delete was called
        self.mock_client.delete.assert_called_once()

    def test_get_collection_info(self):
        """Test getting collection info."""
        # Mock collection info
        mock_info = Mock()
        mock_info.name = "test_collection"
        mock_info.points_count = 1000
        mock_info.config.params.vectors.size = 256
        mock_info.config.params.vectors.distance = "COSINE"
        mock_info.config.hnsw_config = None
        mock_info.status = "green"

        self.mock_client.get_collection.return_value = mock_info

        info = self.db.get_collection_info()

        assert info["name"] == "test_collection"
        assert info["vectors_count"] == 1000
        assert info["vector_size"] == 256
        assert info["distance"] == "COSINE"

    def test_count_vectors(self):
        """Test counting vectors."""
        # Mock count response
        mock_count = Mock()
        mock_count.count = 1000

        self.mock_client.count.return_value = mock_count

        count = self.db.count_vectors()

        assert count == 1000
        self.mock_client.count.assert_called_once_with("test_collection")

    def test_recreate_collection(self):
        """Test recreating collection."""
        # Mock recreate response
        self.mock_client.recreate_collection.return_value = None

        self.db.recreate_collection()

        self.mock_client.recreate_collection.assert_called_once()

    def test_context_manager(self):
        """Test using QuantumDB as context manager."""
        with patch("quantumdb.api.QdrantClient", return_value=self.mock_client):
            with patch("quantumdb.api.load_file", return_value={}):
                with patch("quantumdb.api.LearnablePQ", return_value=self.mock_model):
                    with QuantumDB(collection_name="test") as db:
                        assert db.collection_name == "test"

                    # Check that close was called (if method exists)
                    if hasattr(self.mock_client, "close"):
                        self.mock_client.close.assert_called_once()

    def test_error_handling_add_vectors(self):
        """Test error handling when adding vectors."""
        # Mock upsert to raise exception
        self.mock_client.upsert.side_effect = Exception("Connection error")

        vectors = np.random.randn(100, 256).astype(np.float32)
        ids = list(range(100))

        with pytest.raises(Exception):
            self.db.add(vectors, ids)

    def test_error_handling_search(self):
        """Test error handling during search."""
        # Mock search to raise exception
        self.mock_client.search.side_effect = Exception("Connection error")

        query_vector = np.random.randn(256).astype(np.float32)

        with pytest.raises(Exception):
            self.db.search(query_vector)

    def test_dimension_mismatch_add(self):
        """Test dimension mismatch when adding vectors."""
        # Wrong dimension vectors
        vectors = np.random.randn(100, 128).astype(np.float32)  # Should be 256
        ids = list(range(100))

        with pytest.raises(ValueError):
            self.db.add(vectors, ids)

    def test_dimension_mismatch_search(self):
        """Test dimension mismatch during search."""
        # Wrong dimension query
        query_vector = np.random.randn(128).astype(np.float32)  # Should be 256

        with pytest.raises(ValueError):
            self.db.search(query_vector)


if __name__ == "__main__":
    pytest.main([__file__])
