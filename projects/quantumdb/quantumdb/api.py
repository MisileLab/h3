"""
QuantumDB API - Main interface for vector database operations.

This module provides the main QuantumDB class that integrates learnable
compression models with Qdrant vector database for production deployments.
"""

import os
import json
import logging
from typing import List, Tuple, Optional, Dict, Any, Union
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams, HnswConfigDiff

try:
    import torch
    from safetensors.torch import load_file

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .training.model import LearnablePQ


class QuantumDB:
    """
    QuantumDB: High-performance vector database with learnable compression.

    This class provides a Python interface to train learnable vector compression
    models and integrate with Qdrant vector database for production deployments.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        collection_name: str = "quantumdb_collection",
        vector_size: int = 256,
        distance: str = "cosine",
        hnsw_config: Optional[Dict[str, Any]] = None,
        api_key: Optional[str] = None,
        timeout: int = 30,
        device: Optional[str] = None,
    ):
        """
        Initialize QuantumDB.

        Args:
            model_path: Path to trained model (.safetensors file)
            qdrant_host: Qdrant server host
            qdrant_port: Qdrant server port
            collection_name: Name of the Qdrant collection
            vector_size: Size of compressed vectors
            distance: Distance metric ('cosine', 'euclidean', 'dot')
            hnsw_config: HNSW index configuration
            api_key: Qdrant API key (for cloud deployments)
            timeout: Request timeout in seconds
            device: Device for model inference ('cpu', 'cuda', etc.)
        """
        self.logger = logging.getLogger(__name__)

        # Setup Qdrant client
        self.client = QdrantClient(
            host=qdrant_host,
            port=qdrant_port,
            api_key=api_key,
            timeout=timeout,
        )

        self.collection_name = collection_name
        self.vector_size = vector_size

        # Load compression model
        self.model = None
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if model_path:
            self.load_model(model_path)

        # Setup collection
        self.distance = self._get_distance_metric(distance)
        self.hnsw_config = hnsw_config or {
            "m": 16,
            "ef_construct": 200,
            "full_scan_threshold": 10000,
        }

        self._ensure_collection()

        self.logger.info(f"Initialized QuantumDB with collection: {collection_name}")

    def _get_distance_metric(self, distance: str) -> Distance:
        """Convert distance string to Distance enum."""
        distance_map = {
            "cosine": Distance.COSINE,
            "euclidean": Distance.EUCLID,
            "dot": Distance.DOT,
        }

        if distance not in distance_map:
            raise ValueError(f"Unsupported distance metric: {distance}")

        return distance_map[distance]

    def _ensure_collection(self):
        """Ensure Qdrant collection exists."""
        try:
            collections = self.client.get_collections().collections
            collection_exists = any(c.name == self.collection_name for c in collections)

            if not collection_exists:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=self.distance,
                    ),
                    hnsw_config=HnswConfigDiff(**self.hnsw_config),
                )
                self.logger.info(f"Created collection: {self.collection_name}")
            else:
                self.logger.info(f"Collection already exists: {self.collection_name}")

        except Exception as e:
            self.logger.error(f"Error ensuring collection: {e}")
            raise

    def load_model(self, model_path: str):
        """Load trained compression model."""
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is not installed. Install with: pip install torch"
            )

        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Load metadata if available
        metadata_path = Path(str(model_path).replace(".safetensors", "_metadata.json"))
        if not metadata_path.exists():
            # Fallback for older metadata naming
            metadata_path = model_path.with_suffix(".json")

        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            model_config = metadata.get("model_config", {})
            input_dim = model_config.get("input_dim", 768)
            target_dim = model_config.get("target_dim", self.vector_size)
            n_subvectors = model_config.get("n_subvectors", 16)
            codebook_size = model_config.get("codebook_size", 256)
        else:
            # Default configuration
            input_dim = 768
            target_dim = self.vector_size
            n_subvectors = 16
            codebook_size = 256

        # Create and load model
        self.model = LearnablePQ(
            input_dim=input_dim,
            target_dim=target_dim,
            n_subvectors=n_subvectors,
            codebook_size=codebook_size,
        )

        # Load state dict
        state_dict = load_file(str(model_path))
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        self.logger.info(f"Loaded model from {model_path}")
        self.logger.info(
            f"Model compression ratio: {self.model.get_compression_ratio():.2f}x"
        )

    def encode_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """Encode vectors using the compression model."""
        if self.model is None:
            raise ValueError("No model loaded. Please load a model first.")

        with torch.no_grad():
            # Convert to tensor
            vectors_tensor = torch.FloatTensor(vectors).to(self.device)

            # Encode and quantize
            encoded = self.model.encode(vectors_tensor)
            quantized, _ = self.model.quantize(encoded)

            # Convert back to numpy
            return quantized.cpu().numpy()

    def add(
        self,
        vectors: np.ndarray,
        ids: Optional[List[Union[str, int]]] = None,
        payloads: Optional[List[Dict[str, Any]]] = None,
        batch_size: int = 1000,
    ) -> Dict[str, Any]:
        """
        Add vectors to the database.

        Args:
            vectors: Input vectors to add [n_vectors, input_dim]
            ids: Optional IDs for vectors (auto-generated if None)
            payloads: Optional metadata payloads
            batch_size: Batch size for upsert operations

        Returns:
            result: Operation result information
        """
        if len(vectors) == 0:
            return {"status": "error", "message": "No vectors provided"}

        # Generate IDs if not provided
        if ids is None:
            # Get current max ID
            try:
                info = self.client.get_collection(self.collection_name)
                max_id = info.points_count if info.points_count else 0
                ids = list(range(max_id, max_id + len(vectors)))
            except:
                ids = list(range(len(vectors)))

        if len(ids) != len(vectors):
            raise ValueError("Number of IDs must match number of vectors")

        # Encode vectors
        if self.model:
            encoded_vectors = self.encode_vectors(vectors)
        else:
            # Use vectors as-is if no model
            encoded_vectors = vectors

        if encoded_vectors.shape[1] != self.vector_size:
            raise ValueError(
                f"Vector dimension mismatch: expected {self.vector_size}, "
                f"got {encoded_vectors.shape[1]}"
            )

        # Prepare payloads
        if payloads is None:
            payloads = [{} for _ in range(len(vectors))]
        elif len(payloads) != len(vectors):
            raise ValueError("Number of payloads must match number of vectors")

        # Batch upsert
        total_upserted = 0
        for i in range(0, len(encoded_vectors), batch_size):
            batch_end = min(i + batch_size, len(encoded_vectors))

            batch_vectors = encoded_vectors[i:batch_end].tolist()
            batch_ids = ids[i:batch_end]
            batch_payloads = payloads[i:batch_end]

            points = [
                models.PointStruct(
                    id=idx,
                    vector=vec,
                    payload=payload,
                )
                for idx, vec, payload in zip(batch_ids, batch_vectors, batch_payloads)
            ]

            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )

            total_upserted += len(points)

        result = {
            "status": "success",
            "vectors_added": total_upserted,
            "collection": self.collection_name,
        }

        self.logger.info(f"Added {total_upserted} vectors to collection")
        return result

    def search(
        self,
        query_vector: np.ndarray,
        limit: int = 10,
        score_threshold: Optional[float] = None,
        filter_params: Optional[Dict[str, Any]] = None,
        with_payload: bool = True,
        with_vectors: bool = False,
    ) -> List[Tuple[Union[str, int], float, Dict[str, Any]]]:
        """
        Search for similar vectors.

        Args:
            query_vector: Query vector [input_dim]
            limit: Maximum number of results
            score_threshold: Minimum similarity score threshold
            filter_params: Optional filter parameters
            with_payload: Whether to include payloads in results
            with_vectors: Whether to include vectors in results

        Returns:
            results: List of (id, score, payload) tuples
        """
        # Encode query vector
        if self.model:
            if query_vector.ndim == 1:
                query_vector = query_vector.reshape(1, -1)
            encoded_query = self.encode_vectors(query_vector)[0]
        else:
            encoded_query = query_vector

        if len(encoded_query) != self.vector_size:
            raise ValueError(
                f"Query vector dimension mismatch: expected {self.vector_size}, "
                f"got {len(encoded_query)}"
            )

        # Build search filter
        search_filter = None
        if filter_params:
            search_filter = models.Filter(**filter_params)

        # Perform search
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=encoded_query.tolist(),
            limit=limit,
            query_filter=search_filter,
            with_payload=with_payload,
            with_vectors=with_vectors,
            score_threshold=score_threshold,
        )

        # Format results
        results = []
        for hit in search_result:
            payload = hit.payload if with_payload else {}
            results.append((hit.id, hit.score, payload))

        return results

    def delete(
        self,
        ids: Optional[List[Union[str, int]]] = None,
        filter_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Delete vectors from the database.

        Args:
            ids: List of IDs to delete
            filter_params: Filter to select vectors for deletion

        Returns:
            result: Operation result information
        """
        if ids is None and filter_params is None:
            raise ValueError("Either ids or filter_params must be provided")

        if ids is not None:
            # Delete by IDs
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(points=ids),
            )
            deleted_count = len(ids)
        else:
            # Delete by filter
            search_filter = models.Filter(**filter_params)
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=search_filter,
            )
            deleted_count = "unknown"  # Qdrant doesn't return count for filter deletes

        result = {
            "status": "success",
            "deleted_count": deleted_count,
            "collection": self.collection_name,
        }

        self.logger.info(f"Deleted {deleted_count} vectors from collection")
        return result

    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection."""
        try:
            info = self.client.get_collection(self.collection_name)

            return {
                "name": self.collection_name,
                "vectors_count": info.points_count,
                "vector_size": info.config.params.vectors.size,
                "distance": str(info.config.params.vectors.distance),
                "hnsw_config": info.config.hnsw_config.__dict__
                if info.config.hnsw_config
                else None,
                "status": info.status,
            }
        except Exception as e:
            self.logger.error(f"Error getting collection info: {e}")
            return {"error": str(e)}

    def count_vectors(self) -> int:
        """Count vectors in the collection."""
        try:
            result = self.client.count(collection_name=self.collection_name)
            return result.count
        except Exception as e:
            self.logger.error(f"Error counting vectors: {e}")
            return 0

    def recreate_collection(self):
        """Recreate the collection (clear all data)."""
        try:
            self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=self.distance,
                ),
                hnsw_config=HnswConfigDiff(**self.hnsw_config),
            )
            self.logger.info(f"Recreated collection: {self.collection_name}")
        except Exception as e:
            self.logger.error(f"Error recreating collection: {e}")
            raise

    def close(self):
        """Close the database connection."""
        if hasattr(self.client, "close"):
            self.client.close()
        self.logger.info("Closed QuantumDB connection")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
