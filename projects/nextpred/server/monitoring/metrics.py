"""
Metrics Collection Service for Next Action Predictor
Handles Prometheus metrics and performance monitoring
"""

import logging
from typing import Dict, Any, List, Optional
import time
import asyncio
from datetime import datetime, timedelta
from collections import defaultdict, deque
import json

logger = logging.getLogger(__name__)


class MetricsCollector:
    """Service for collecting and managing application metrics"""

    def __init__(self):
        self.metrics = defaultdict(float)
        self.counters = defaultdict(int)
        self.histograms = defaultdict(lambda: deque(maxlen=1000))
        self.gauges = defaultdict(float)
        
        # Performance tracking
        self.request_times = deque(maxlen=1000)
        self.prediction_times = deque(maxlen=1000)
        self.inference_times = deque(maxlen=1000)
        
        # System metrics
        self.cpu_usage = deque(maxlen=60)  # Last 60 minutes
        self.memory_usage = deque(maxlen=60)
        
        # Business metrics
        self.daily_predictions = 0
        self.daily_events = 0
        self.daily_feedback = 0
        
        # Error tracking
        self.error_counts = defaultdict(int)
        self.error_rates = defaultdict(float)
        
        self._collection_active = False

    async def start_collection(self):
        """Start background metrics collection"""
        if self._collection_active:
            return
        
        self._collection_active = True
        logger.info("Started metrics collection")
        
        # Start background tasks
        asyncio.create_task(self._collect_system_metrics())
        asyncio.create_task(self._reset_daily_counters())
        asyncio.create_task(self._calculate_rates())

    async def stop_collection(self):
        """Stop metrics collection"""
        self._collection_active = False
        logger.info("Stopped metrics collection")

    def record_request(self, endpoint: str, duration: float, status_code: int = 200):
        """Record API request metrics"""
        self.counters[f"requests_total_{endpoint}"] += 1
        self.counters["requests_total_all"] += 1
        
        if status_code >= 400:
            self.error_counts[f"errors_{endpoint}"] += 1
            self.error_counts["errors_total"] += 1
        
        self.histograms[f"request_duration_{endpoint}"].append(duration)
        self.request_times.append(duration)

    def record_prediction_request(self, duration: Optional[float] = None):
        """Record prediction request"""
        self.counters["predictions_total"] += 1
        self.daily_predictions += 1
        
        if duration:
            self.prediction_times.append(duration)
            self.histograms["prediction_duration"].append(duration)

    def record_inference(self, duration: float, model_version: str = "unknown"):
        """Record model inference metrics"""
        self.counters["inferences_total"] += 1
        self.inference_times.append(duration)
        self.histograms["inference_duration"].append(duration)
        
        # Track by model version
        self.counters[f"inferences_model_{model_version}"] += 1

    def record_events_uploaded(self, count: int):
        """Record events uploaded to server"""
        self.counters["events_uploaded_total"] += count
        self.daily_events += count

    def record_feedback(self, feedback_type: str):
        """Record user feedback"""
        self.counters["feedback_total"] += 1
        self.counters[f"feedback_{feedback_type}"] += 1
        self.daily_feedback += 1

    def record_model_download(self, version: str, file_size: int):
        """Record model download"""
        self.counters["model_downloads_total"] += 1
        self.counters[f"model_downloads_version_{version}"] += 1
        self.metrics["model_download_size_bytes"] += file_size

    def record_cache_hit(self, cache_type: str):
        """Record cache hit"""
        self.counters["cache_hits_total"] += 1
        self.counters[f"cache_hits_{cache_type}"] += 1

    def record_cache_miss(self, cache_type: str):
        """Record cache miss"""
        self.counters["cache_misses_total"] += 1
        self.counters[f"cache_misses_{cache_type}"] += 1

    def record_embedding_operation(self, operation: str, duration: float):
        """Record embedding generation/storage operation"""
        self.counters["embedding_operations_total"] += 1
        self.counters[f"embedding_operations_{operation}"] += 1
        self.histograms["embedding_duration"].append(duration)

    def update_gauge(self, metric_name: str, value: float):
        """Update gauge metric"""
        self.gauges[metric_name] = value

    def increment_counter(self, metric_name: str, value: int = 1):
        """Increment counter metric"""
        self.counters[metric_name] += value

    async def get_prometheus_metrics(self) -> str:
        """Get metrics in Prometheus format"""
        try:
            metrics_lines = []
            
            # Counters
            for name, value in self.counters.items():
                metrics_lines.append(f"# TYPE {name} counter")
                metrics_lines.append(f"{name} {value}")
            
            # Gauges
            for name, value in self.gauges.items():
                metrics_lines.append(f"# TYPE {name} gauge")
                metrics_lines.append(f"{name} {value}")
            
            # Histograms (simplified - just provide summary stats)
            for name, values in self.histograms.items():
                if values:
                    metrics_lines.append(f"# TYPE {name} histogram")
                    metrics_lines.append(f"{name}_count {len(values)}")
                    metrics_lines.append(f"{name}_sum {sum(values)}")
                    
                    # Calculate quantiles
                    sorted_values = sorted(values)
                    n = len(sorted_values)
                    if n > 0:
                        metrics_lines.append(f"{name}_quantile_0.5 {sorted_values[int(n * 0.5)]}")
                        metrics_lines.append(f"{name}_quantile_0.95 {sorted_values[int(n * 0.95)]}")
                        metrics_lines.append(f"{name}_quantile_0.99 {sorted_values[int(n * 0.99)]}")
            
            # System metrics
            metrics_lines.append("# TYPE system_cpu_usage gauge")
            if self.cpu_usage:
                metrics_lines.append(f"system_cpu_usage {self.cpu_usage[-1] if self.cpu_usage else 0}")
            
            metrics_lines.append("# TYPE system_memory_usage gauge")
            if self.memory_usage:
                metrics_lines.append(f"system_memory_usage {self.memory_usage[-1] if self.memory_usage else 0}")
            
            # Business metrics
            metrics_lines.append("# TYPE daily_predictions counter")
            metrics_lines.append(f"daily_predictions {self.daily_predictions}")
            
            metrics_lines.append("# TYPE daily_events counter")
            metrics_lines.append(f"daily_events {self.daily_events}")
            
            metrics_lines.append("# TYPE daily_feedback counter")
            metrics_lines.append(f"daily_feedback {self.daily_feedback}")
            
            # Error rates
            for name, rate in self.error_rates.items():
                metrics_lines.append(f"# TYPE {name} gauge")
                metrics_lines.append(f"{name} {rate}")
            
            return "\n".join(metrics_lines)

        except Exception as e:
            logger.error(f"Failed to generate Prometheus metrics: {e}")
            return "# Error generating metrics\n"

    async def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary for dashboard"""
        try:
            summary = {
                "timestamp": datetime.utcnow().isoformat(),
                "requests": {
                    "total": self.counters.get("requests_total_all", 0),
                    "avg_duration": self._calculate_average(self.request_times),
                    "p95_duration": self._calculate_percentile(self.request_times, 0.95),
                    "error_rate": self._calculate_error_rate()
                },
                "predictions": {
                    "total": self.counters.get("predictions_total", 0),
                    "daily": self.daily_predictions,
                    "avg_duration": self._calculate_average(self.prediction_times),
                    "p95_duration": self._calculate_percentile(self.prediction_times, 0.95)
                },
                "inferences": {
                    "total": self.counters.get("inferences_total", 0),
                    "avg_duration": self._calculate_average(self.inference_times),
                    "p95_duration": self._calculate_percentile(self.inference_times, 0.95)
                },
                "events": {
                    "uploaded_total": self.counters.get("events_uploaded_total", 0),
                    "daily": self.daily_events
                },
                "feedback": {
                    "total": self.counters.get("feedback_total", 0),
                    "daily": self.daily_feedback
                },
                "system": {
                    "cpu_usage": self.cpu_usage[-1] if self.cpu_usage else 0,
                    "memory_usage": self.memory_usage[-1] if self.memory_usage else 0
                },
                "cache": {
                    "hits_total": self.counters.get("cache_hits_total", 0),
                    "misses_total": self.counters.get("cache_misses_total", 0),
                    "hit_rate": self._calculate_cache_hit_rate()
                },
                "models": {
                    "downloads_total": self.counters.get("model_downloads_total", 0),
                    "download_size_mb": self.metrics.get("model_download_size_bytes", 0) / (1024 * 1024)
                }
            }
            
            return summary

        except Exception as e:
            logger.error(f"Failed to generate metrics summary: {e}")
            return {"error": str(e), "timestamp": datetime.utcnow().isoformat()}

    async def _collect_system_metrics(self):
        """Collect system metrics periodically"""
        while self._collection_active:
            try:
                # CPU usage (simplified)
                cpu_usage = self._get_cpu_usage()
                self.cpu_usage.append(cpu_usage)
                self.gauges["system_cpu_usage"] = cpu_usage
                
                # Memory usage (simplified)
                memory_usage = self._get_memory_usage()
                self.memory_usage.append(memory_usage)
                self.gauges["system_memory_usage"] = memory_usage
                
            except Exception as e:
                logger.error(f"Failed to collect system metrics: {e}")
            
            await asyncio.sleep(60)  # Collect every minute

    async def _reset_daily_counters(self):
        """Reset daily counters at midnight"""
        while self._collection_active:
            try:
                now = datetime.now()
                
                # Calculate seconds until midnight
                midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
                if midnight <= now:
                    midnight = midnight + timedelta(days=1)
                
                seconds_until_midnight = (midnight - now).total_seconds()
                
                await asyncio.sleep(seconds_until_midnight)
                
                # Reset daily counters
                self.daily_predictions = 0
                self.daily_events = 0
                self.daily_feedback = 0
                
                logger.info("Reset daily counters")
                
            except Exception as e:
                logger.error(f"Failed to reset daily counters: {e}")
                await asyncio.sleep(3600)  # Retry in an hour

    async def _calculate_rates(self):
        """Calculate rates and percentages periodically"""
        while self._collection_active:
            try:
                # Calculate error rates
                total_requests = self.counters.get("requests_total_all", 0)
                total_errors = self.error_counts.get("errors_total", 0)
                
                if total_requests > 0:
                    self.error_rates["error_rate_total"] = total_errors / total_requests
                
                # Calculate cache hit rate
                cache_hits = self.counters.get("cache_hits_total", 0)
                cache_misses = self.counters.get("cache_misses_total", 0)
                total_cache_ops = cache_hits + cache_misses
                
                if total_cache_ops > 0:
                    self.gauges["cache_hit_rate"] = cache_hits / total_cache_ops
                
            except Exception as e:
                logger.error(f"Failed to calculate rates: {e}")
            
            await asyncio.sleep(300)  # Calculate every 5 minutes

    def _get_cpu_usage(self) -> float:
        """Get CPU usage percentage (simplified)"""
        try:
            import psutil
            return psutil.cpu_percent(interval=1)
        except ImportError:
            # Fallback: return a mock value
            return min(50.0 + (time.time() % 100) / 2, 100.0)

    def _get_memory_usage(self) -> float:
        """Get memory usage percentage (simplified)"""
        try:
            import psutil
            return psutil.virtual_memory().percent
        except ImportError:
            # Fallback: return a mock value
            return min(40.0 + (time.time() % 80) / 2, 100.0)

    def _calculate_average(self, values: deque) -> float:
        """Calculate average of values"""
        if not values:
            return 0.0
        return sum(values) / len(values)

    def _calculate_percentile(self, values: deque, percentile: float) -> float:
        """Calculate percentile of values"""
        if not values:
            return 0.0
        
        sorted_values = sorted(values)
        n = len(sorted_values)
        index = int(n * percentile)
        return sorted_values[min(index, n - 1)]

    def _calculate_error_rate(self) -> float:
        """Calculate overall error rate"""
        total_requests = self.counters.get("requests_total_all", 0)
        total_errors = self.error_counts.get("errors_total", 0)
        
        if total_requests == 0:
            return 0.0
        
        return total_errors / total_requests

    def _calculate_cache_hit_rate(self) -> float:
        """Calculate cache hit rate"""
        cache_hits = self.counters.get("cache_hits_total", 0)
        cache_misses = self.counters.get("cache_misses_total", 0)
        total_ops = cache_hits + cache_misses
        
        if total_ops == 0:
            return 0.0
        
        return cache_hits / total_ops

    def get_health_status(self) -> Dict[str, Any]:
        """Get health status based on metrics"""
        try:
            status = {
                "healthy": True,
                "checks": {}
            }
            
            # Check error rate
            error_rate = self._calculate_error_rate()
            status["checks"]["error_rate"] = {
                "status": "healthy" if error_rate < 0.05 else "unhealthy",
                "value": error_rate,
                "threshold": 0.05
            }
            
            # Check response times
            avg_response_time = self._calculate_average(self.request_times)
            status["checks"]["response_time"] = {
                "status": "healthy" if avg_response_time < 1.0 else "unhealthy",
                "value": avg_response_time,
                "threshold": 1.0
            }
            
            # Check system resources
            cpu_usage = self.cpu_usage[-1] if self.cpu_usage else 0
            status["checks"]["cpu_usage"] = {
                "status": "healthy" if cpu_usage < 80 else "unhealthy",
                "value": cpu_usage,
                "threshold": 80.0
            }
            
            memory_usage = self.memory_usage[-1] if self.memory_usage else 0
            status["checks"]["memory_usage"] = {
                "status": "healthy" if memory_usage < 85 else "unhealthy",
                "value": memory_usage,
                "threshold": 85.0
            }
            
            # Overall health
            status["healthy"] = all(
                check["status"] == "healthy" 
                for check in status["checks"].values()
            )
            
            return status

        except Exception as e:
            logger.error(f"Failed to get health status: {e}")
            return {
                "healthy": False,
                "error": str(e)
            }