"""
Predictor Service for Next Action Predictor
Handles ML model inference and prediction generation
"""

import logging
from typing import List, Dict, Any, Optional
import asyncio
import numpy as np
from datetime import datetime
import json

from ..db.postgres import Database
from ..db.qdrant_client import QdrantManager

logger = logging.getLogger(__name__)


class Predictor:
    """Service for generating next action predictions"""

    def __init__(self, db: Database, qdrant: QdrantManager):
        self.db = db
        self.qdrant = qdrant
        self.model_version = None
        self.rule_based_weights = {
            'tab_switch': 0.7,
            'search': 0.2,
            'scroll': 0.1
        }

    async def predict(
        self, 
        context: Dict[str, Any], 
        current_tab: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate next action predictions based on context and current tab"""
        try:
            # Extract features from context
            features = await self._extract_features(context, current_tab)
            
            # Get candidate predictions from different sources
            tab_predictions = await self._predict_tab_switches(features, context)
            search_predictions = await self._predict_searches(features, context)
            scroll_predictions = await self._predict_scrolls(features, context)
            
            # Combine and rank predictions
            all_predictions = (
                tab_predictions + 
                search_predictions + 
                scroll_predictions
            )
            
            # Apply router weights and rank
            ranked_predictions = self._rank_predictions(all_predictions, features)
            
            # Return top 3 predictions
            return ranked_predictions[:3]

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            # Fallback to simple rule-based predictions
            return self._fallback_predictions(context, current_tab)

    async def _extract_features(
        self, 
        context: Dict[str, Any], 
        current_tab: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract features from context and current tab"""
        try:
            now = datetime.now()
            events = context.get('events', [])
            summary = context.get('summary', {})
            
            features = {
                # Temporal features
                'hour': now.hour,
                'minute': now.minute,
                'day_of_week': now.weekday(),
                'is_weekend': now.weekday() >= 5,
                
                # Current state
                'current_url': current_tab.get('url', ''),
                'current_domain': self._extract_domain(current_tab.get('url', '')),
                'current_tab_index': current_tab.get('index', 0),
                'total_tabs': len(summary.get('uniqueDomains', [])),
                'time_on_page': self._calculate_time_on_page(events, current_tab),
                
                # Recent activity patterns
                'recent_urls': summary.get('recentUrls', [])[:5],
                'recent_domains': summary.get('uniqueDomains', [])[:5],
                'recent_searches': summary.get('recentSearches', [])[:3],
                'tab_switch_frequency': summary.get('tabSwitches', 0),
                'search_frequency': summary.get('searches', 0),
                'scroll_frequency': summary.get('scrolls', 0),
                
                # Time-based features
                'time_since_last_tab_switch': self._get_time_since_last_activity(
                    events, 'tab_switch'
                ),
                'time_since_last_search': self._get_time_since_last_activity(
                    events, 'search'
                ),
                'time_since_last_navigation': self._get_time_since_last_activity(
                    events, 'navigation'
                )
            }
            
            return features

        except Exception as e:
            logger.error(f"Failed to extract features: {e}")
            return {}

    async def _predict_tab_switches(
        self, 
        features: Dict[str, Any], 
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Predict tab switch actions"""
        try:
            predictions = []
            current_url = features.get('current_url', '')
            current_domain = features.get('current_domain', '')
            
            # Get similar URLs from embedding search
            similar_urls = await self.qdrant.find_similar_urls(
                current_url, 
                limit=5, 
                score_threshold=0.5
            )
            
            # Generate predictions from similar URLs
            for i, similar in enumerate(similar_urls[:3]):
                if similar['url'] != current_url:
                    confidence = max(0.1, similar['score'] * 0.8)
                    predictions.append({
                        'type': 'tab_switch',
                        'url': similar['url'],
                        'title': self._extract_title(similar['url']),
                        'confidence': confidence,
                        'source': 'embedding_similarity',
                        'domain': self._extract_domain(similar['url']),
                        'tabId': None  # Would need to map to actual tab ID
                    })
            
            # Add predictions from recent domains
            recent_domains = features.get('recent_domains', [])
            for i, domain in enumerate(recent_domains[:3]):
                if domain and domain != current_domain:
                    confidence = max(0.1, 0.6 - i * 0.15)
                    predictions.append({
                        'type': 'tab_switch',
                        'url': f'https://{domain}',
                        'title': domain,
                        'confidence': confidence,
                        'source': 'recent_domains',
                        'domain': domain,
                        'tabId': None
                    })
            
            return predictions

        except Exception as e:
            logger.error(f"Failed to predict tab switches: {e}")
            return []

    async def _predict_searches(
        self, 
        features: Dict[str, Any], 
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Predict search actions"""
        try:
            predictions = []
            recent_searches = features.get('recent_searches', [])
            
            # Generate predictions from recent searches
            for i, search_query in enumerate(recent_searches[:2]):
                if search_query and len(search_query.strip()) > 2:
                    # Find similar searches
                    similar_searches = await self.qdrant.find_similar_searches(
                        search_query,
                        limit=2,
                        score_threshold=0.6
                    )
                    
                    # Use original or similar search
                    query = search_query
                    if similar_searches and similar_searches[0]['score'] > 0.8:
                        query = similar_searches[0]['query']
                    
                    confidence = max(0.2, 0.7 - i * 0.2)
                    predictions.append({
                        'type': 'search',
                        'url': f'https://www.google.com/search?q={query}',
                        'title': f'Search: {query}',
                        'confidence': confidence,
                        'source': 'recent_searches',
                        'query': query
                    })
            
            # Predict search based on current context
            current_domain = features.get('current_domain', '')
            if current_domain and self._is_search_heavy_domain(current_domain):
                predictions.append({
                    'type': 'search',
                    'url': f'https://www.google.com/search?q=',
                    'title': 'New Search',
                    'confidence': 0.4,
                    'source': 'context_based',
                    'query': ''
                })
            
            return predictions

        except Exception as e:
            logger.error(f"Failed to predict searches: {e}")
            return []

    async def _predict_scrolls(
        self, 
        features: Dict[str, Any], 
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Predict scroll actions (less common for tab switching)"""
        try:
            predictions = []
            
            # Only predict scroll if user has been scrolling recently
            scroll_frequency = features.get('scroll_frequency', 0)
            if scroll_frequency > 2:
                current_url = features.get('current_url', '')
                
                # Predict scroll to top (common action)
                predictions.append({
                    'type': 'scroll',
                    'url': current_url,
                    'title': 'Scroll to Top',
                    'confidence': 0.3,
                    'source': 'scroll_pattern',
                    'scrollPosition': 0.0
                })
                
                # Predict scroll to bottom for content-heavy pages
                if self._is_content_heavy_page(current_url):
                    predictions.append({
                        'type': 'scroll',
                        'url': current_url,
                        'title': 'Scroll to Bottom',
                        'confidence': 0.25,
                        'source': 'content_heavy',
                        'scrollPosition': 1.0
                    })
            
            return predictions

        except Exception as e:
            logger.error(f"Failed to predict scrolls: {e}")
            return []

    def _rank_predictions(
        self, 
        predictions: List[Dict[str, Any]], 
        features: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Rank and filter predictions"""
        try:
            # Apply router weights based on context
            router_weights = self._calculate_router_weights(features)
            
            # Adjust confidence scores based on router weights
            for prediction in predictions:
                pred_type = prediction['type']
                if pred_type in router_weights:
                    prediction['confidence'] *= router_weights[pred_type]
            
            # Remove duplicates and low-confidence predictions
            seen_urls = set()
            filtered_predictions = []
            
            for pred in predictions:
                url = pred.get('url', '')
                if url not in seen_urls and pred['confidence'] > 0.1:
                    seen_urls.add(url)
                    filtered_predictions.append(pred)
            
            # Sort by confidence
            filtered_predictions.sort(key=lambda x: x['confidence'], reverse=True)
            
            return filtered_predictions

        except Exception as e:
            logger.error(f"Failed to rank predictions: {e}")
            return predictions

    def _calculate_router_weights(self, features: Dict[str, Any]) -> Dict[str, float]:
        """Calculate router weights based on current context"""
        try:
            weights = self.rule_based_weights.copy()
            
            # Adjust weights based on activity patterns
            tab_switch_freq = features.get('tab_switch_frequency', 0)
            search_freq = features.get('search_frequency', 0)
            scroll_freq = features.get('scroll_frequency', 0)
            
            total_activity = tab_switch_freq + search_freq + scroll_freq
            if total_activity > 0:
                weights['tab_switch'] = 0.3 + (tab_switch_freq / total_activity) * 0.4
                weights['search'] = 0.2 + (search_freq / total_activity) * 0.3
                weights['scroll'] = 0.1 + (scroll_freq / total_activity) * 0.2
            
            # Time-based adjustments
            hour = features.get('hour', 12)
            if 9 <= hour <= 17:  # Work hours
                weights['search'] *= 1.2
            elif hour >= 20 or hour <= 6:  # Evening/night
                weights['tab_switch'] *= 1.1
            
            # Normalize weights
            total_weight = sum(weights.values())
            if total_weight > 0:
                weights = {k: v / total_weight for k, v in weights.items()}
            
            return weights

        except Exception as e:
            logger.error(f"Failed to calculate router weights: {e}")
            return self.rule_based_weights

    def _fallback_predictions(
        self, 
        context: Dict[str, Any], 
        current_tab: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Fallback rule-based predictions"""
        try:
            predictions = []
            summary = context.get('summary', {})
            current_url = current_tab.get('url', '')
            current_domain = self._extract_domain(current_url)
            
            # Predict returning to recent domains
            recent_domains = summary.get('uniqueDomains', [])
            for i, domain in enumerate(recent_domains[:3]):
                if domain and domain != current_domain:
                    predictions.append({
                        'type': 'tab_switch',
                        'url': f'https://{domain}',
                        'title': domain,
                        'confidence': max(0.1, 0.5 - i * 0.15),
                        'source': 'fallback_rules',
                        'domain': domain
                    })
            
            # Predict search if recent searches exist
            recent_searches = summary.get('recentSearches', [])
            if recent_searches:
                last_search = recent_searches[0]
                predictions.append({
                    'type': 'search',
                    'url': f'https://www.google.com/search?q={last_search}',
                    'title': f'Search: {last_search}',
                    'confidence': 0.3,
                    'source': 'fallback_rules',
                    'query': last_search
                })
            
            return predictions[:3]

        except Exception as e:
            logger.error(f"Failed to generate fallback predictions: {e}")
            return []

    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        try:
            from urllib.parse import urlparse
            return urlparse(url).netloc
        except Exception:
            return ''

    def _extract_title(self, url: str) -> str:
        """Extract title from URL"""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            return parsed.netloc + parsed.pathname
        except Exception:
            return url

    def _calculate_time_on_page(
        self, 
        events: List[Dict[str, Any]], 
        current_tab: Dict[str, Any]
    ) -> float:
        """Calculate time spent on current page"""
        try:
            current_url = current_tab.get('url', '')
            page_load_events = [
                e for e in events 
                if e.get('type') == 'page_load' and 
                e.get('data', {}).get('url') == current_url
            ]
            
            if not page_load_events:
                return 0.0
            
            last_page_load = max(e.get('timestamp', 0) for e in page_load_events)
            return (datetime.now().timestamp() * 1000 - last_page_load) / 1000  # seconds

        except Exception:
            return 0.0

    def _get_time_since_last_activity(
        self, 
        events: List[Dict[str, Any]], 
        activity_type: str
    ) -> float:
        """Get time since last activity of specific type"""
        try:
            type_events = [
                e for e in events 
                if e.get('type') == activity_type
            ]
            
            if not type_events:
                return 10.0 * 60.0  # 10 minutes
            
            last_activity = max(e.get('timestamp', 0) for e in type_events)
            current_time = datetime.now().timestamp() * 1000
            return (current_time - last_activity) / 1000  # seconds

        except Exception:
            return 10.0 * 60.0

    def _is_search_heavy_domain(self, domain: str) -> bool:
        """Check if domain typically involves a lot of searching"""
        search_domains = [
            'google.com', 'bing.com', 'duckduckgo.com', 'yahoo.com',
            'stackoverflow.com', 'github.com', 'reddit.com', 'wikipedia.org'
        ]
        return any(search_domain in domain for search_domain in search_domains)

    def _is_content_heavy_page(self, url: str) -> bool:
        """Check if URL likely contains long-form content"""
        content_patterns = [
            'docs.', 'documentation', 'wiki', 'blog', 'article',
            'news', 'medium.com', 'reddit.com/r/', 'stackoverflow.com/questions'
        ]
        return any(pattern in url for pattern in content_patterns)

    async def get_model_version(self) -> Optional[str]:
        """Get current model version"""
        try:
            if not self.model_version:
                model_info = await self.db.get_model_version()
                if model_info:
                    self.model_version = model_info.version
            return self.model_version
        except Exception as e:
            logger.error(f"Failed to get model version: {e}")
            return None