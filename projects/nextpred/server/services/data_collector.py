"""
Data Collector Service for Next Action Predictor
Handles event storage, processing, and user analytics
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import asyncio
import hashlib
import re
from urllib.parse import urlparse

from ..db.postgres import Database
from ..db.qdrant_client import QdrantManager

logger = logging.getLogger(__name__)


class DataCollector:
    """Service for collecting and processing user behavior events"""

    def __init__(self, db: Database, qdrant: QdrantManager):
        self.db = db
        self.qdrant = qdrant
        self.embedding_cache = {}
        self.url_patterns = {
            'sensitive': [
                r'accounts\.',
                r'auth\.',
                r'login',
                r'register',
                r'signup',
                r'token=',
                r'session=',
                r'key=',
                r'password',
                r'bank',
                r'payment',
                r'checkout',
                r'cart'
            ],
            'search_engines': [
                r'google\.com/search',
                r'bing\.com/search',
                r'duckduckgo\.com',
                r'yahoo\.com/search'
            ]
        }

    async def store_events(self, events: List[Dict[str, Any]]) -> int:
        """Store a batch of events in the database"""
        try:
            # Filter and sanitize events
            sanitized_events = []
            for event in events:
                if self._should_store_event(event):
                    sanitized_event = self._sanitize_event(event)
                    sanitized_events.append(sanitized_event)

            if not sanitized_events:
                logger.info("No events to store after filtering")
                return 0

            # Store in database
            stored_count = await self.db.store_events_batch(sanitized_events)
            
            # Process events for embeddings (async)
            asyncio.create_task(self.process_events_batch(sanitized_events))
            
            logger.info(f"Stored {stored_count} events out of {len(events)} received")
            return stored_count

        except Exception as e:
            logger.error(f"Failed to store events: {e}")
            raise

    async def process_events_batch(self, events: List[Dict[str, Any]]):
        """Process events for embeddings and analysis"""
        try:
            for event in events:
                await self._process_single_event(event)
        except Exception as e:
            logger.error(f"Failed to process events batch: {e}")

    async def _process_single_event(self, event: Dict[str, Any]):
        """Process a single event for embeddings"""
        try:
            # Generate embeddings for URLs
            if event.get('data', {}).get('url'):
                url = event['data']['url']
                await self._store_url_embedding(url, event)

            # Generate embeddings for search queries
            if event.get('type') == 'search' and event.get('data', {}).get('query'):
                query = event['data']['query']
                await self._store_search_embedding(query, event)

        except Exception as e:
            logger.error(f"Failed to process event {event.get('id')}: {e}")

    async def _store_url_embedding(self, url: str, event: Dict[str, Any]):
        """Store URL embedding in Qdrant"""
        try:
            # Check cache first
            url_hash = hashlib.md5(url.encode()).hexdigest()
            if url_hash in self.embedding_cache:
                return

            # Generate embedding (simplified - in practice use a proper embedding model)
            embedding = await self._generate_url_embedding(url)
            
            # Store in Qdrant
            await self.qdrant.store_url_embedding(
                url=url,
                embedding=embedding,
                metadata={
                    'event_id': event.get('id'),
                    'timestamp': event.get('timestamp'),
                    'domain': self._extract_domain(url),
                    'event_type': event.get('type')
                }
            )

            # Cache the embedding
            self.embedding_cache[url_hash] = embedding

        except Exception as e:
            logger.error(f"Failed to store URL embedding for {url}: {e}")

    async def _store_search_embedding(self, query: str, event: Dict[str, Any]):
        """Store search query embedding in Qdrant"""
        try:
            # Generate embedding for search query
            embedding = await self._generate_search_embedding(query)
            
            # Store in Qdrant
            await self.qdrant.store_search_embedding(
                query=query,
                embedding=embedding,
                metadata={
                    'event_id': event.get('id'),
                    'timestamp': event.get('timestamp'),
                    'url': event.get('data', {}).get('url'),
                    'event_type': 'search'
                }
            )

        except Exception as e:
            logger.error(f"Failed to store search embedding for '{query}': {e}")

    async def _generate_url_embedding(self, url: str) -> List[float]:
        """Generate embedding for URL (simplified implementation)"""
        # In practice, use a proper embedding model like sentence-transformers
        # For now, create a simple hash-based embedding
        url_hash = hashlib.sha256(url.encode()).hexdigest()
        embedding = []
        for i in range(0, len(url_hash), 2):
            byte_val = int(url_hash[i:i+2], 16)
            embedding.append(byte_val / 255.0)
        
        # Pad or truncate to 768 dimensions (standard for many models)
        while len(embedding) < 768:
            embedding.append(0.0)
        return embedding[:768]

    async def _generate_search_embedding(self, query: str) -> List[float]:
        """Generate embedding for search query (simplified implementation)"""
        # Similar to URL embedding, use hash-based approach for now
        query_hash = hashlib.sha256(query.encode()).hexdigest()
        embedding = []
        for i in range(0, len(query_hash), 2):
            byte_val = int(query_hash[i:i+2], 16)
            embedding.append(byte_val / 255.0)
        
        while len(embedding) < 768:
            embedding.append(0.0)
        return embedding[:768]

    def _should_store_event(self, event: Dict[str, Any]) -> bool:
        """Check if event should be stored based on privacy rules"""
        try:
            url = event.get('data', {}).get('url', '')
            
            # Check for sensitive URLs
            for pattern in self.url_patterns['sensitive']:
                if re.search(pattern, url, re.IGNORECASE):
                    return False
            
            # Check for sensitive search queries
            if event.get('type') == 'search':
                query = event.get('data', {}).get('query', '')
                if self._is_sensitive_query(query):
                    return False
            
            return True

        except Exception as e:
            logger.error(f"Error checking event storage: {e}")
            return False

    def _sanitize_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize event data to remove sensitive information"""
        try:
            sanitized = event.copy()
            data = sanitized.get('data', {}).copy()

            # Sanitize URL
            if 'url' in data:
                data['url'] = self._sanitize_url(data['url'])
            
            # Sanitize search query
            if 'query' in data:
                data['query'] = self._sanitize_query(data['query'])

            sanitized['data'] = data
            return sanitized

        except Exception as e:
            logger.error(f"Error sanitizing event: {e}")
            return event

    def _sanitize_url(self, url: str) -> str:
        """Remove sensitive parameters from URL"""
        try:
            parsed = urlparse(url)
            
            # Remove sensitive query parameters
            sensitive_params = [
                'token', 'session', 'key', 'password', 'auth', 'access_token',
                'refresh_token', 'api_key', 'secret', 'csrf', 'ssid'
            ]
            
            query_params = []
            if parsed.query:
                for param in parsed.query.split('&'):
                    if '=' in param:
                        key, _ = param.split('=', 1)
                        if key not in sensitive_params:
                            query_params.append(param)
            
            # Reconstruct URL
            clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
            if query_params:
                clean_url += f"?{'&'.join(query_params)}"
            if parsed.fragment:
                clean_url += f"#{parsed.fragment}"
            
            return clean_url

        except Exception:
            return url

    def _sanitize_query(self, query: str) -> str:
        """Remove sensitive information from search queries"""
        # Remove potential PII patterns
        sensitive_patterns = [
            r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',  # Credit card numbers
            r'\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b',  # SSN patterns
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'  # Email addresses
        ]

        sanitized = query
        for pattern in sensitive_patterns:
            sanitized = re.sub(pattern, '[REDACTED]', sanitized, flags=re.IGNORECASE)
        
        return sanitized

    def _is_sensitive_query(self, query: str) -> bool:
        """Check if search query contains sensitive information"""
        sensitive_keywords = [
            'password', 'social security', 'credit card', 'bank account',
            'medical record', 'private', 'confidential', 'ssn', 'cvv'
        ]
        
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in sensitive_keywords)

    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        try:
            return urlparse(url).netloc
        except Exception:
            return ''

    async def store_feedback(self, feedback_data: Dict[str, Any]) -> str:
        """Store user feedback"""
        try:
            feedback_id = await self.db.store_feedback(feedback_data)
            logger.info(f"Stored feedback: {feedback_id}")
            return feedback_id
        except Exception as e:
            logger.error(f"Failed to store feedback: {e}")
            raise

    async def get_user_stats(self, user_id: Optional[str] = None, days: int = 7) -> Dict[str, Any]:
        """Get user statistics and activity summary"""
        try:
            # Get recent events
            events = await self.db.get_recent_events(minutes=days * 24 * 60, user_id=user_id)
            
            # Analyze events
            stats = {
                'user_id': user_id,
                'total_events': len(events),
                'events_by_type': {},
                'unique_domains': set(),
                'top_domains': {},
                'avg_session_duration': 0.0,
                'period_days': days,
                'timestamp': datetime.utcnow()
            }

            # Process events
            session_starts = []
            for event in events:
                # Count by type
                event_type = event.type
                stats['events_by_type'][event_type] = stats['events_by_type'].get(event_type, 0) + 1
                
                # Track domains
                if event.data and event.data.get('url'):
                    domain = self._extract_domain(event.data['url'])
                    if domain:
                        stats['unique_domains'].add(domain)
                        stats['top_domains'][domain] = stats['top_domains'].get(domain, 0) + 1
                
                # Track session starts (simplified)
                if event.type == 'tab_switch':
                    session_starts.append(event.timestamp)

            # Convert sets to counts
            stats['unique_domains'] = len(stats['unique_domains'])
            
            # Sort top domains
            stats['top_domains'] = [
                {'domain': domain, 'count': count}
                for domain, count in sorted(stats['top_domains'].items(), key=lambda x: x[1], reverse=True)[:10]
            ]

            # Calculate average session duration (simplified)
            if len(session_starts) > 1:
                session_durations = []
                for i in range(1, len(session_starts)):
                    duration = (session_starts[i] - session_starts[i-1]) / 1000 / 60  # minutes
                    if duration > 0 and duration < 120:  # Filter reasonable session lengths
                        session_durations.append(duration)
                
                if session_durations:
                    stats['avg_session_duration'] = sum(session_durations) / len(session_durations)

            return stats

        except Exception as e:
            logger.error(f"Failed to get user stats: {e}")
            raise

    async def get_feedback_stats(self, days: int = 7) -> Dict[str, Any]:
        """Get feedback statistics and accuracy metrics"""
        try:
            # This would query the feedback table and calculate metrics
            # For now, return placeholder data
            return {
                'total_feedback': 0,
                'accuracy_rate': 0.0,
                'top_prediction_accuracy': 0.0,
                'feedback_by_type': {},
                'period_days': days,
                'timestamp': datetime.utcnow()
            }

        except Exception as e:
            logger.error(f"Failed to get feedback stats: {e}")
            raise