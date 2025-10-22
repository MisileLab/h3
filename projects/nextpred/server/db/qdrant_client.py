"""
Qdrant Vector Database Client for Next Action Predictor
Handles storage and retrieval of embeddings for URLs and search queries
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import asyncio
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from qdrant_client.async_client import AsyncQdrantClient
import hashlib

logger = logging.getLogger(__name__)


class QdrantManager:
    """Manager for Qdrant vector database operations"""

    def __init__(self, host: str = "localhost", port: int = 6333):
        self.host = host
        self.port = port
        self.client: Optional[AsyncQdrantClient] = None
        self.collection_names = {
            'urls': 'url_embeddings',
            'searches': 'search_embeddings'
        }
        self.vector_size = 768  # Standard embedding size

    async def initialize(self):
        """Initialize Qdrant client and create collections"""
        try:
            self.client = AsyncQdrantClient(host=self.host, port=self.port)
            
            # Create collections if they don't exist
            await self._create_collection_if_not_exists(
                self.collection_names['urls'],
                self.vector_size
            )
            
            await self._create_collection_if_not_exists(
                self.collection_names['searches'],
                self.vector_size
            )
            
            logger.info("Qdrant client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant client: {e}")
            raise

    async def close(self):
        """Close Qdrant client connection"""
        if self.client:
            await self.client.close()
            logger.info("Qdrant client connection closed")

    async def health_check(self) -> bool:
        """Check Qdrant health"""
        try:
            if not self.client:
                return False
            
            # Try to get collection info
            collections = await self.client.get_collections()
            return True
            
        except Exception as e:
            logger.error(f"Qdrant health check failed: {e}")
            return False

    async def get_status(self) -> Dict[str, Any]:
        """Get Qdrant status information"""
        try:
            if not self.client:
                return {"status": "uninitialized"}
            
            collections = await self.client.get_collections()
            collection_info = {}
            
            for collection in collections.collections:
                info = await self.client.get_collection(collection.name)
                collection_info[collection.name] = {
                    "vectors_count": info.vectors_count,
                    "segments_count": info.segments_count,
                    "status": info.status
                }
            
            return {
                "status": "healthy",
                "collections": collection_info,
                "host": self.host,
                "port": self.port
            }
            
        except Exception as e:
            logger.error(f"Failed to get Qdrant status: {e}")
            return {"status": "unhealthy", "error": str(e)}

    async def _create_collection_if_not_exists(self, name: str, vector_size: int):
        """Create collection if it doesn't exist"""
        try:
            collections = await self.client.get_collections()
            existing_names = [c.name for c in collections.collections]
            
            if name not in existing_names:
                await self.client.create_collection(
                    collection_name=name,
                    vectors_config=VectorParams(
                        size=vector_size,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Created collection: {name}")
            else:
                logger.info(f"Collection already exists: {name}")
                
        except Exception as e:
            logger.error(f"Failed to create collection {name}: {e}")
            raise

    async def store_url_embedding(
        self, 
        url: str, 
        embedding: List[float], 
        metadata: Dict[str, Any]
    ):
        """Store URL embedding in Qdrant"""
        try:
            # Generate point ID from URL
            point_id = self._generate_point_id(url)
            
            point = PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    'url': url,
                    'type': 'url',
                    'domain': metadata.get('domain', ''),
                    'timestamp': metadata.get('timestamp', 0),
                    'event_id': metadata.get('event_id', ''),
                    'event_type': metadata.get('event_type', '')
                }
            )
            
            await self.client.upsert(
                collection_name=self.collection_names['urls'],
                points=[point]
            )
            
        except Exception as e:
            logger.error(f"Failed to store URL embedding for {url}: {e}")
            raise

    async def store_search_embedding(
        self, 
        query: str, 
        embedding: List[float], 
        metadata: Dict[str, Any]
    ):
        """Store search query embedding in Qdrant"""
        try:
            # Generate point ID from query
            point_id = self._generate_point_id(query, prefix='search_')
            
            point = PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    'query': query,
                    'type': 'search',
                    'timestamp': metadata.get('timestamp', 0),
                    'event_id': metadata.get('event_id', ''),
                    'url': metadata.get('url', '')
                }
            )
            
            await self.client.upsert(
                collection_name=self.collection_names['searches'],
                points=[point]
            )
            
        except Exception as e:
            logger.error(f"Failed to store search embedding for '{query}': {e}")
            raise

    async def find_similar_urls(
        self, 
        url: str, 
        limit: int = 10,
        score_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Find similar URLs based on embedding similarity"""
        try:
            # Generate embedding for the query URL
            query_embedding = await self._generate_url_embedding(url)
            
            # Search for similar URLs
            search_result = await self.client.search(
                collection_name=self.collection_names['urls'],
                query_vector=query_embedding,
                limit=limit,
                score_threshold=score_threshold
            )
            
            results = []
            for hit in search_result:
                results.append({
                    'url': hit.payload.get('url', ''),
                    'domain': hit.payload.get('domain', ''),
                    'score': hit.score,
                    'event_type': hit.payload.get('event_type', ''),
                    'timestamp': hit.payload.get('timestamp', 0)
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to find similar URLs for {url}: {e}")
            return []

    async def find_similar_searches(
        self, 
        query: str, 
        limit: int = 10,
        score_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Find similar search queries based on embedding similarity"""
        try:
            # Generate embedding for the query
            query_embedding = await self._generate_search_embedding(query)
            
            # Search for similar queries
            search_result = await self.client.search(
                collection_name=self.collection_names['searches'],
                query_vector=query_embedding,
                limit=limit,
                score_threshold=score_threshold
            )
            
            results = []
            for hit in search_result:
                results.append({
                    'query': hit.payload.get('query', ''),
                    'score': hit.score,
                    'url': hit.payload.get('url', ''),
                    'timestamp': hit.payload.get('timestamp', 0)
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to find similar searches for '{query}': {e}")
            return []

    async def get_url_embedding(self, url: str) -> Optional[List[float]]:
        """Get stored embedding for a specific URL"""
        try:
            point_id = self._generate_point_id(url)
            
            points = await self.client.retrieve(
                collection_name=self.collection_names['urls'],
                ids=[point_id]
            )
            
            if points:
                return points[0].vector
            return None
            
        except Exception as e:
            logger.error(f"Failed to get URL embedding for {url}: {e}")
            return None

    async def get_search_embedding(self, query: str) -> Optional[List[float]]:
        """Get stored embedding for a specific search query"""
        try:
            point_id = self._generate_point_id(query, prefix='search_')
            
            points = await self.client.retrieve(
                collection_name=self.collection_names['searches'],
                ids=[point_id]
            )
            
            if points:
                return points[0].vector
            return None
            
        except Exception as e:
            logger.error(f"Failed to get search embedding for '{query}': {e}")
            return None

    async def get_domain_embeddings(
        self, 
        domain: str, 
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get all embeddings for a specific domain"""
        try:
            # Create filter for domain
            filter_condition = Filter(
                must=[
                    FieldCondition(
                        key="domain",
                        match=MatchValue(value=domain)
                    )
                ]
            )
            
            # Search with filter
            search_result = await self.client.scroll(
                collection_name=self.collection_names['urls'],
                scroll_filter=filter_condition,
                limit=limit
            )
            
            results = []
            for point in search_result[0]:  # scroll returns (points, next_page_offset)
                results.append({
                    'url': point.payload.get('url', ''),
                    'vector': point.vector,
                    'timestamp': point.payload.get('timestamp', 0),
                    'event_type': point.payload.get('event_type', '')
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to get domain embeddings for {domain}: {e}")
            return []

    async def _generate_url_embedding(self, url: str) -> List[float]:
        """Generate embedding for URL (simplified implementation)"""
        # In practice, use a proper embedding model
        # For now, create a simple hash-based embedding
        url_hash = hashlib.sha256(url.encode()).hexdigest()
        embedding = []
        for i in range(0, len(url_hash), 2):
            byte_val = int(url_hash[i:i+2], 16)
            embedding.append(byte_val / 255.0)
        
        # Pad or truncate to 768 dimensions
        while len(embedding) < self.vector_size:
            embedding.append(0.0)
        return embedding[:self.vector_size]

    async def _generate_search_embedding(self, query: str) -> List[float]:
        """Generate embedding for search query (simplified implementation)"""
        query_hash = hashlib.sha256(query.encode()).hexdigest()
        embedding = []
        for i in range(0, len(query_hash), 2):
            byte_val = int(query_hash[i:i+2], 16)
            embedding.append(byte_val / 255.0)
        
        while len(embedding) < self.vector_size:
            embedding.append(0.0)
        return embedding[:self.vector_size]

    def _generate_point_id(self, text: str, prefix: str = '') -> str:
        """Generate unique point ID from text"""
        hash_value = hashlib.md5(text.encode()).hexdigest()
        return f"{prefix}{hash_value}"

    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics for all collections"""
        try:
            stats = {}
            
            for collection_name in self.collection_names.values():
                try:
                    info = await self.client.get_collection(collection_name)
                    stats[collection_name] = {
                        'vectors_count': info.vectors_count,
                        'segments_count': info.segments_count,
                        'status': info.status
                    }
                except Exception as e:
                    stats[collection_name] = {'error': str(e)}
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {}