"""
Embedding generation utilities.

This module provides various embedding generators including sentence transformers,
OpenAI embeddings, and custom embedding models.
"""

import os
import logging
from typing import List, Optional, Union, Dict, Any
from abc import ABC, abstractmethod

import numpy as np
from tqdm import tqdm

try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import openai

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import torch
    from transformers import AutoTokenizer, AutoModel

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class BaseEmbedder(ABC):
    """Abstract base class for embedders."""

    @abstractmethod
    def encode(self, texts: List[str], batch_size: int = 32, **kwargs) -> np.ndarray:
        """Encode texts to embeddings."""
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        """Get embedding dimension."""
        pass


class SentenceTransformerEmbedder(BaseEmbedder):
    """Embedder using sentence-transformers models."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
        cache_folder: Optional[str] = None,
    ):
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers is not installed. "
                "Install with: pip install sentence-transformers"
            )

        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = SentenceTransformer(
            model_name,
            device=self.device,
            cache_folder=cache_folder,
        )

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Loaded sentence transformer: {model_name}")

    def encode(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True,
        normalize_embeddings: bool = True,
        **kwargs,
    ) -> np.ndarray:
        """Encode texts using sentence transformer."""
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=normalize_embeddings,
            **kwargs,
        )

        return embeddings.astype(np.float32)

    def get_dimension(self) -> int:
        """Get embedding dimension."""
        return self.model.get_sentence_embedding_dimension()


class OpenAIEmbedder(BaseEmbedder):
    """Embedder using OpenAI's embedding API."""

    def __init__(
        self,
        model_name: str = "text-embedding-ada-002",
        api_key: Optional[str] = None,
        organization: Optional[str] = None,
        max_retries: int = 3,
        request_timeout: float = 30.0,
    ):
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "openai is not installed. Install with: pip install openai"
            )

        self.model_name = model_name
        self.max_retries = max_retries
        self.request_timeout = request_timeout

        # Setup OpenAI client
        client_config = {}
        if api_key:
            client_config["api_key"] = api_key
        if organization:
            client_config["organization"] = organization

        self.client = openai.OpenAI(**client_config)

        # Model dimensions
        self.dimensions = {
            "text-embedding-ada-002": 1536,
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
        }

        if model_name not in self.dimensions:
            raise ValueError(f"Unknown model: {model_name}")

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized OpenAI embedder with model: {model_name}")

    def encode(self, texts: List[str], batch_size: int = 100, **kwargs) -> np.ndarray:
        """Encode texts using OpenAI API."""
        embeddings = []

        # Process in batches
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
            batch_texts = texts[i : i + batch_size]

            try:
                response = self.client.embeddings.create(
                    model=self.model_name, input=batch_texts, **kwargs
                )

                batch_embeddings = [data.embedding for data in response.data]
                embeddings.extend(batch_embeddings)

            except Exception as e:
                self.logger.error(f"Error encoding batch {i}: {e}")
                # Add zero embeddings as fallback
                batch_embeddings = [np.zeros(self.get_dimension()) for _ in batch_texts]
                embeddings.extend(batch_embeddings)

        return np.array(embeddings, dtype=np.float32)

    def get_dimension(self) -> int:
        """Get embedding dimension."""
        return self.dimensions[self.model_name]


class HuggingFaceEmbedder(BaseEmbedder):
    """Embedder using Hugging Face transformers."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers is not installed. Install with: pip install transformers"
            )

        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        tokenizer_kwargs = tokenizer_kwargs or {}
        model_kwargs = model_kwargs or {}

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
        self.model = AutoModel.from_pretrained(model_name, **model_kwargs)
        self.model.to(self.device)
        self.model.eval()

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Loaded Hugging Face model: {model_name}")

    def encode(
        self,
        texts: List[str],
        batch_size: int = 32,
        max_length: int = 512,
        normalize: bool = True,
        **kwargs,
    ) -> np.ndarray:
        """Encode texts using Hugging Face model."""
        embeddings = []

        for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
            batch_texts = texts[i : i + batch_size]

            # Tokenize
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )

            # Move to device
            encoded = {k: v.to(self.device) for k, v in encoded.items()}

            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**encoded)

                # Mean pooling
                attention_mask = encoded["attention_mask"]
                token_embeddings = outputs.last_hidden_state

                input_mask_expanded = (
                    attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                )
                batch_embeddings = torch.sum(
                    token_embeddings * input_mask_expanded, 1
                ) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

                if normalize:
                    batch_embeddings = torch.nn.functional.normalize(
                        batch_embeddings, p=2, dim=1
                    )

                embeddings.append(batch_embeddings.cpu().numpy())

        return np.vstack(embeddings).astype(np.float32)

    def get_dimension(self) -> int:
        """Get embedding dimension."""
        return self.model.config.hidden_size


class RandomEmbedder(BaseEmbedder):
    """Random embedder for testing purposes."""

    def __init__(self, dimension: int = 768, random_state: int = 42):
        self.dimension = dimension
        self.random_state = random_state
        np.random.seed(random_state)

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized random embedder with dimension: {dimension}")

    def encode(
        self, texts: List[str], batch_size: int = 32, normalize: bool = True, **kwargs
    ) -> np.ndarray:
        """Generate random embeddings."""
        n_texts = len(texts)
        embeddings = np.random.randn(n_texts, self.dimension).astype(np.float32)

        if normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / (norms + 1e-8)

        return embeddings

    def get_dimension(self) -> int:
        """Get embedding dimension."""
        return self.dimension


class CachedEmbedder(BaseEmbedder):
    """Wrapper that caches embeddings to avoid recomputation."""

    def __init__(
        self,
        embedder: BaseEmbedder,
        cache_dir: str = "embeddings_cache",
        cache_key_func: Optional[callable] = None,
    ):
        self.embedder = embedder
        self.cache_dir = cache_dir
        self.cache_key_func = cache_key_func or (lambda x: x)

        import os

        os.makedirs(cache_dir, exist_ok=True)

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized cached embedder with cache dir: {cache_dir}")

    def _get_cache_path(self, text_hash: str) -> str:
        """Get cache file path for a text hash."""
        return os.path.join(self.cache_dir, f"{text_hash}.npy")

    def _get_text_hash(self, text: str) -> str:
        """Get hash for text."""
        import hashlib

        return hashlib.md5(self.cache_key_func(text).encode()).hexdigest()

    def encode(
        self, texts: List[str], batch_size: int = 32, use_cache: bool = True, **kwargs
    ) -> np.ndarray:
        """Encode texts with caching."""
        if not use_cache:
            return self.embedder.encode(texts, batch_size, **kwargs)

        embeddings = []
        uncached_texts = []
        uncached_indices = []

        # Check cache first
        for i, text in enumerate(texts):
            text_hash = self._get_text_hash(text)
            cache_path = self._get_cache_path(text_hash)

            if os.path.exists(cache_path):
                embedding = np.load(cache_path)
                embeddings.append((i, embedding))
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)

        # Generate embeddings for uncached texts
        if uncached_texts:
            uncached_embeddings = self.embedder.encode(
                uncached_texts, batch_size, **kwargs
            )

            # Cache new embeddings
            for text, embedding, original_idx in zip(
                uncached_texts, uncached_embeddings, uncached_indices
            ):
                text_hash = self._get_text_hash(text)
                cache_path = self._get_cache_path(text_hash)
                np.save(cache_path, embedding)
                embeddings.append((original_idx, embedding))

        # Sort by original index and return
        embeddings.sort(key=lambda x: x[0])
        return np.array([emb for _, emb in embeddings])

    def get_dimension(self) -> int:
        """Get embedding dimension."""
        return self.embedder.get_dimension()


def create_embedder(
    embedder_type: str, model_name: Optional[str] = None, **kwargs
) -> BaseEmbedder:
    """
    Factory function to create embedders.

    Args:
        embedder_type: Type of embedder ('sentence_transformer', 'openai', 'huggingface', 'random')
        model_name: Model name (if applicable)
        **kwargs: Additional arguments for embedder

    Returns:
        embedder: Configured embedder instance
    """
    if embedder_type == "sentence_transformer":
        model_name = model_name or "all-MiniLM-L6-v2"
        return SentenceTransformerEmbedder(model_name, **kwargs)

    elif embedder_type == "openai":
        model_name = model_name or "text-embedding-ada-002"
        return OpenAIEmbedder(model_name, **kwargs)

    elif embedder_type == "huggingface":
        model_name = model_name or "sentence-transformers/all-MiniLM-L6-v2"
        return HuggingFaceEmbedder(model_name, **kwargs)

    elif embedder_type == "random":
        return RandomEmbedder(**kwargs)

    else:
        raise ValueError(f"Unknown embedder type: {embedder_type}")


def batch_encode_texts(
    texts: List[str],
    embedder: BaseEmbedder,
    batch_size: int = 32,
    save_path: Optional[str] = None,
    **kwargs,
) -> np.ndarray:
    """
    Encode texts in batches with optional saving.

    Args:
        texts: List of texts to encode
        embedder: Embedder instance
        batch_size: Batch size for encoding
        save_path: Path to save embeddings (optional)
        **kwargs: Additional arguments for embedder

    Returns:
        embeddings: Generated embeddings
    """
    embeddings = embedder.encode(texts, batch_size, **kwargs)

    if save_path:
        np.save(save_path, embeddings)
        logging.info(f"Saved embeddings to {save_path}")

    return embeddings
