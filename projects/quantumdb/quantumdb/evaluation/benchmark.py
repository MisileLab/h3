"""
Benchmarking utilities for vector search systems.

This module provides comprehensive benchmarking capabilities including
performance testing, scalability analysis, and comparison with baseline systems.
"""

import time
import psutil
import threading
import logging
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .metrics import MetricCalculator, create_default_metrics


@dataclass
class BenchmarkResult:
    """Result of a benchmark run."""

    metric_name: str
    value: float
    unit: str
    timestamp: float
    metadata: Dict[str, Any]


@dataclass
class PerformanceMetrics:
    """Performance metrics for a system."""

    queries_per_second: float
    latency_p50: float
    latency_p95: float
    latency_p99: float
    memory_usage_mb: float
    cpu_usage_percent: float
    index_build_time: Optional[float] = None
    storage_size_mb: Optional[float] = None


class BenchmarkRunner:
    """
    Comprehensive benchmark runner for vector search systems.

    This class provides utilities to benchmark various aspects of vector
    search systems including search performance, indexing speed, memory usage,
    and retrieval quality.
    """

    def __init__(
        self,
        output_dir: str = "benchmark_results",
        k_values: List[int] = [1, 5, 10, 20, 100],
        num_threads: int = 4,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.k_values = k_values
        self.num_threads = num_threads
        self.logger = logging.getLogger(__name__)

        # Setup metrics calculator
        self.metric_calculator = create_default_metrics(k_values)

        # Results storage
        self.results: List[BenchmarkResult] = []

    def benchmark_search_performance(
        self,
        model,
        query_vectors: np.ndarray,
        ground_truth: Optional[List[set]] = None,
        num_queries: int = 1000,
        limit: int = 10,
        warmup_queries: int = 100,
    ) -> PerformanceMetrics:
        """
        Benchmark search performance.

        Args:
            model: Vector search model
            query_vectors: Query vectors for testing
            ground_truth: Optional ground truth relevance for quality metrics
            num_queries: Number of queries to run
            limit: Search limit (k)
            warmup_queries: Number of warmup queries

        Returns:
            metrics: Performance metrics
        """
        self.logger.info("Starting search performance benchmark...")

        # Sample queries
        if len(query_vectors) > num_queries:
            indices = np.random.choice(len(query_vectors), num_queries, replace=False)
            test_queries = query_vectors[indices]
        else:
            test_queries = query_vectors

        # Warmup
        self.logger.info(f"Warming up with {warmup_queries} queries...")
        for i in range(min(warmup_queries, len(test_queries))):
            try:
                if hasattr(model, "search"):
                    model.search(test_queries[i], limit=limit)
                else:
                    # Custom search logic
                    pass
            except Exception as e:
                self.logger.warning(f"Warmup query {i} failed: {e}")

        # Benchmark queries
        latencies = []
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

        self.logger.info(f"Running {len(test_queries)} benchmark queries...")
        start_time = time.time()

        for query in tqdm(test_queries, desc="Running queries"):
            query_start = time.time()

            try:
                if hasattr(model, "search"):
                    results = model.search(query, limit=limit)
                else:
                    # Fallback for custom models
                    results = []

                query_end = time.time()
                latencies.append(query_end - query_start)

            except Exception as e:
                self.logger.error(f"Query failed: {e}")
                latencies.append(float("inf"))

        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

        # Calculate metrics
        total_time = end_time - start_time
        valid_latencies = [l for l in latencies if l != float("inf")]

        queries_per_second = len(valid_latencies) / total_time if total_time > 0 else 0
        latency_p50 = np.percentile(valid_latencies, 50) if valid_latencies else 0
        latency_p95 = np.percentile(valid_latencies, 95) if valid_latencies else 0
        latency_p99 = np.percentile(valid_latencies, 99) if valid_latencies else 0
        memory_usage = end_memory - start_memory
        cpu_usage = psutil.cpu_percent()

        metrics = PerformanceMetrics(
            queries_per_second=queries_per_second,
            latency_p50=latency_p50,
            latency_p95=latency_p95,
            latency_p99=latency_p99,
            memory_usage_mb=memory_usage,
            cpu_usage_percent=cpu_usage,
        )

        # Log results
        self.logger.info(f"Search Performance Results:")
        self.logger.info(f"  QPS: {queries_per_second:.2f}")
        self.logger.info(f"  Latency P50: {latency_p50 * 1000:.2f}ms")
        self.logger.info(f"  Latency P95: {latency_p95 * 1000:.2f}ms")
        self.logger.info(f"  Latency P99: {latency_p99 * 1000:.2f}ms")
        self.logger.info(f"  Memory Usage: {memory_usage:.2f}MB")
        self.logger.info(f"  CPU Usage: {cpu_usage:.1f}%")

        # Store results
        self._store_performance_metrics("search_performance", metrics)

        return metrics

    def benchmark_indexing_performance(
        self,
        model,
        vectors: np.ndarray,
        batch_sizes: List[int] = [100, 500, 1000, 5000],
    ) -> Dict[int, PerformanceMetrics]:
        """
        Benchmark indexing performance with different batch sizes.

        Args:
            model: Vector search model
            vectors: Vectors to index
            batch_sizes: List of batch sizes to test

        Returns:
            results: Dictionary mapping batch size to performance metrics
        """
        self.logger.info("Starting indexing performance benchmark...")

        results = {}

        for batch_size in batch_sizes:
            self.logger.info(f"Testing batch size: {batch_size}")

            # Clear existing index if possible
            if hasattr(model, "recreate_collection"):
                model.recreate_collection()

            # Time the indexing process
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024

            try:
                # Add vectors in batches
                for i in range(0, len(vectors), batch_size):
                    batch_end = min(i + batch_size, len(vectors))
                    batch_vectors = vectors[i:batch_end]
                    batch_ids = list(range(i, batch_end))

                    if hasattr(model, "add"):
                        model.add(batch_vectors, batch_ids)

                end_time = time.time()
                end_memory = psutil.Process().memory_info().rss / 1024 / 1024

                # Calculate metrics
                index_time = end_time - start_time
                memory_usage = end_memory - start_memory

                metrics = PerformanceMetrics(
                    queries_per_second=len(vectors) / index_time,
                    latency_p50=0,  # Not applicable for indexing
                    latency_p95=0,
                    latency_p99=0,
                    memory_usage_mb=memory_usage,
                    cpu_usage_percent=psutil.cpu_percent(),
                    index_build_time=index_time,
                )

                results[batch_size] = metrics

                self.logger.info(
                    f"  Batch size {batch_size}: {len(vectors) / index_time:.2f} vectors/sec"
                )

            except Exception as e:
                self.logger.error(f"Indexing failed for batch size {batch_size}: {e}")
                results[batch_size] = None

        # Store results
        self._store_indexing_results(results)

        return results

    def benchmark_concurrent_search(
        self,
        model,
        query_vectors: np.ndarray,
        num_threads_list: List[int] = [1, 2, 4, 8, 16],
        queries_per_thread: int = 100,
        limit: int = 10,
    ) -> Dict[int, PerformanceMetrics]:
        """
        Benchmark concurrent search performance.

        Args:
            model: Vector search model
            query_vectors: Query vectors for testing
            num_threads_list: List of thread counts to test
            queries_per_thread: Number of queries per thread
            limit: Search limit

        Returns:
            results: Dictionary mapping thread count to performance metrics
        """
        self.logger.info("Starting concurrent search benchmark...")

        results = {}

        # Sample queries
        max_queries = len(query_vectors)
        total_queries_needed = max(num_threads_list) * queries_per_thread

        if max_queries < total_queries_needed:
            # Repeat queries if needed
            repeat_factor = (total_queries_needed // max_queries) + 1
            query_vectors = np.tile(query_vectors, (repeat_factor, 1))[
                :total_queries_needed
            ]

        for num_threads in num_threads_list:
            self.logger.info(f"Testing with {num_threads} threads...")

            def worker_function(thread_id: int) -> List[float]:
                """Worker function for concurrent search."""
                thread_latencies = []
                start_idx = thread_id * queries_per_thread
                end_idx = start_idx + queries_per_thread

                for i in range(start_idx, end_idx):
                    query = query_vectors[i]

                    query_start = time.time()
                    try:
                        if hasattr(model, "search"):
                            model.search(query, limit=limit)
                        query_end = time.time()
                        thread_latencies.append(query_end - query_start)
                    except Exception as e:
                        self.logger.warning(f"Thread {thread_id} query {i} failed: {e}")
                        thread_latencies.append(float("inf"))

                return thread_latencies

            # Run concurrent queries
            start_time = time.time()

            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [
                    executor.submit(worker_function, i) for i in range(num_threads)
                ]

                all_latencies = []
                for future in as_completed(futures):
                    try:
                        thread_latencies = future.result()
                        all_latencies.extend(thread_latencies)
                    except Exception as e:
                        self.logger.error(f"Thread failed: {e}")

            end_time = time.time()

            # Calculate metrics
            total_time = end_time - start_time
            valid_latencies = [l for l in all_latencies if l != float("inf")]

            queries_per_second = (
                len(valid_latencies) / total_time if total_time > 0 else 0
            )
            latency_p50 = np.percentile(valid_latencies, 50) if valid_latencies else 0
            latency_p95 = np.percentile(valid_latencies, 95) if valid_latencies else 0
            latency_p99 = np.percentile(valid_latencies, 99) if valid_latencies else 0

            metrics = PerformanceMetrics(
                queries_per_second=queries_per_second,
                latency_p50=latency_p50,
                latency_p95=latency_p95,
                latency_p99=latency_p99,
                memory_usage_mb=0,  # Not tracked for concurrent test
                cpu_usage_percent=psutil.cpu_percent(),
            )

            results[num_threads] = metrics

            self.logger.info(f"  {num_threads} threads: {queries_per_second:.2f} QPS")

        # Store results
        self._store_concurrent_results(results)

        return results

    def benchmark_scalability(
        self,
        model_factory: Callable[[], Any],
        vector_sizes: List[int] = [1000, 10000, 100000, 1000000],
        dimension: int = 768,
        num_queries: int = 1000,
    ) -> Dict[int, Dict[str, float]]:
        """
        Benchmark scalability with different dataset sizes.

        Args:
            model_factory: Function that creates a fresh model instance
            vector_sizes: List of dataset sizes to test
            dimension: Vector dimension
            num_queries: Number of queries for performance testing

        Returns:
            results: Dictionary mapping dataset size to metrics
        """
        self.logger.info("Starting scalability benchmark...")

        results = {}

        for size in vector_sizes:
            self.logger.info(f"Testing dataset size: {size}")

            # Generate synthetic data
            vectors = np.random.randn(size, dimension).astype(np.float32)
            query_vectors = np.random.randn(num_queries, dimension).astype(np.float32)

            try:
                # Create fresh model
                model = model_factory()

                # Benchmark indexing
                start_time = time.time()
                if hasattr(model, "add"):
                    model.add(vectors, list(range(size)))
                index_time = time.time() - start_time

                # Benchmark search
                search_start = time.time()
                for query in query_vectors[:100]:  # Sample for speed
                    if hasattr(model, "search"):
                        model.search(query, limit=10)
                search_time = time.time() - search_start

                # Calculate metrics
                indexing_throughput = size / index_time if index_time > 0 else 0
                search_qps = 100 / search_time if search_time > 0 else 0

                results[size] = {
                    "indexing_time": index_time,
                    "indexing_throughput": indexing_throughput,
                    "search_qps": search_qps,
                    "memory_usage_mb": psutil.Process().memory_info().rss / 1024 / 1024,
                }

                self.logger.info(
                    f"  Size {size}: {indexing_throughput:.0f} vec/s, {search_qps:.1f} QPS"
                )

            except Exception as e:
                self.logger.error(f"Scalability test failed for size {size}: {e}")
                results[size] = {"error": str(e)}

        # Store results
        self._store_scalability_results(results)

        return results

    def compare_models(
        self,
        models: Dict[str, Any],
        query_vectors: np.ndarray,
        ground_truth: Optional[List[set]] = None,
        num_queries: int = 1000,
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare multiple models on the same dataset.

        Args:
            models: Dictionary mapping model names to model instances
            query_vectors: Query vectors for testing
            ground_truth: Optional ground truth for quality metrics
            num_queries: Number of queries to run

        Returns:
            comparison: Dictionary mapping model names to their metrics
        """
        self.logger.info("Starting model comparison...")

        comparison = {}

        for model_name, model in models.items():
            self.logger.info(f"Evaluating model: {model_name}")

            try:
                # Performance metrics
                perf_metrics = self.benchmark_search_performance(
                    model, query_vectors, ground_truth, num_queries
                )

                # Quality metrics if ground truth is available
                quality_metrics = {}
                if ground_truth:
                    quality_metrics = self._evaluate_search_quality(
                        model, query_vectors, ground_truth
                    )

                # Combine metrics
                combined_metrics = {
                    "queries_per_second": perf_metrics.queries_per_second,
                    "latency_p50_ms": perf_metrics.latency_p50 * 1000,
                    "latency_p95_ms": perf_metrics.latency_p95 * 1000,
                    "latency_p99_ms": perf_metrics.latency_p99 * 1000,
                    "memory_usage_mb": perf_metrics.memory_usage_mb,
                    **quality_metrics,
                }

                comparison[model_name] = combined_metrics

            except Exception as e:
                self.logger.error(f"Model {model_name} failed: {e}")
                comparison[model_name] = {"error": str(e)}

        # Store and log comparison
        self._store_comparison_results(comparison)
        self._log_comparison_table(comparison)

        return comparison

    def _evaluate_search_quality(
        self,
        model,
        query_vectors: np.ndarray,
        ground_truth: List[set],
        num_queries: int = 1000,
    ) -> Dict[str, float]:
        """Evaluate search quality metrics."""
        # Sample queries
        if len(query_vectors) > num_queries:
            indices = np.random.choice(len(query_vectors), num_queries, replace=False)
            test_queries = query_vectors[indices]
            test_ground_truth = [ground_truth[i] for i in indices]
        else:
            test_queries = query_vectors
            test_ground_truth = ground_truth

        # Perform searches
        retrieved_ids = []
        for query in test_queries:
            if hasattr(model, "search"):
                results = model.search(query, limit=max(self.k_values))
                retrieved_doc_ids = [doc_id for doc_id, _, _ in results]
                retrieved_ids.append(retrieved_doc_ids)
            else:
                retrieved_ids.append([])

        # Compute metrics
        metrics = self.metric_calculator.compute_all(
            retrieved_ids, test_ground_truth, self.k_values
        )

        return metrics

    def _store_performance_metrics(self, test_name: str, metrics: PerformanceMetrics):
        """Store performance metrics."""
        timestamp = time.time()

        self.results.extend(
            [
                BenchmarkResult(
                    f"{test_name}_qps",
                    metrics.queries_per_second,
                    "queries/sec",
                    timestamp,
                    {},
                ),
                BenchmarkResult(
                    f"{test_name}_latency_p50",
                    metrics.latency_p50,
                    "seconds",
                    timestamp,
                    {},
                ),
                BenchmarkResult(
                    f"{test_name}_latency_p95",
                    metrics.latency_p95,
                    "seconds",
                    timestamp,
                    {},
                ),
                BenchmarkResult(
                    f"{test_name}_latency_p99",
                    metrics.latency_p99,
                    "seconds",
                    timestamp,
                    {},
                ),
                BenchmarkResult(
                    f"{test_name}_memory", metrics.memory_usage_mb, "MB", timestamp, {}
                ),
                BenchmarkResult(
                    f"{test_name}_cpu",
                    metrics.cpu_usage_percent,
                    "percent",
                    timestamp,
                    {},
                ),
            ]
        )

    def _store_indexing_results(self, results: Dict[int, PerformanceMetrics]):
        """Store indexing results."""
        timestamp = time.time()
        for batch_size, metrics in results.items():
            if metrics:
                self.results.append(
                    BenchmarkResult(
                        f"indexing_batch_{batch_size}",
                        metrics.queries_per_second,
                        "vectors/sec",
                        timestamp,
                        {"batch_size": batch_size},
                    )
                )

    def _store_concurrent_results(self, results: Dict[int, PerformanceMetrics]):
        """Store concurrent search results."""
        timestamp = time.time()
        for num_threads, metrics in results.items():
            self.results.append(
                BenchmarkResult(
                    f"concurrent_{num_threads}_threads",
                    metrics.queries_per_second,
                    "queries/sec",
                    timestamp,
                    {"num_threads": num_threads},
                )
            )

    def _store_scalability_results(self, results: Dict[int, Dict[str, float]]):
        """Store scalability results."""
        timestamp = time.time()
        for size, metrics in results.items():
            for metric_name, value in metrics.items():
                if metric_name != "error":
                    self.results.append(
                        BenchmarkResult(
                            f"scalability_{metric_name}",
                            value,
                            "varies",
                            timestamp,
                            {"dataset_size": size},
                        )
                    )

    def _store_comparison_results(self, comparison: Dict[str, Dict[str, float]]):
        """Store model comparison results."""
        timestamp = time.time()
        for model_name, metrics in comparison.items():
            for metric_name, value in metrics.items():
                if metric_name != "error":
                    self.results.append(
                        BenchmarkResult(
                            f"comparison_{model_name}_{metric_name}",
                            value,
                            "varies",
                            timestamp,
                            {"model": model_name},
                        )
                    )

    def _log_comparison_table(self, comparison: Dict[str, Dict[str, float]]):
        """Log comparison table."""
        if not comparison:
            return

        # Get all metric names
        metric_names = set()
        for metrics in comparison.values():
            metric_names.update(metrics.keys())
        metric_names = sorted([m for m in metric_names if m != "error"])

        # Create table
        self.logger.info(f"\nModel Comparison Results:")
        self.logger.info("-" * (30 + len(metric_names) * 20))

        # Header
        header = f"{'Model':<20}"
        for metric_name in metric_names:
            header += f"{metric_name:<20}"
        self.logger.info(header)
        self.logger.info("-" * (30 + len(metric_names) * 20))

        # Rows
        for model_name, metrics in comparison.items():
            if "error" in metrics:
                row = f"{model_name:<20}ERROR: {metrics['error']}"
            else:
                row = f"{model_name:<20}"
                for metric_name in metric_names:
                    value = metrics.get(metric_name, 0.0)
                    if "latency" in metric_name:
                        row += f"{value:<20.2f}"
                    elif "qps" in metric_name or "throughput" in metric_name:
                        row += f"{value:<20.1f}"
                    else:
                        row += f"{value:<20.4f}"
            self.logger.info(row)

        self.logger.info("-" * (30 + len(metric_names) * 20))

    def save_results(self, filename: str = None):
        """Save all benchmark results to file."""
        if filename is None:
            filename = f"benchmark_results_{int(time.time())}.json"

        output_path = self.output_dir / filename

        # Convert results to serializable format
        serializable_results = []
        for result in self.results:
            serializable_results.append(
                {
                    "metric_name": result.metric_name,
                    "value": result.value,
                    "unit": result.unit,
                    "timestamp": result.timestamp,
                    "metadata": result.metadata,
                }
            )

        with open(output_path, "w") as f:
            json.dump(serializable_results, f, indent=2)

        self.logger.info(f"Results saved to {output_path}")

    def generate_report(self, filename: str = None) -> str:
        """Generate a text report of all benchmark results."""
        if filename is None:
            filename = f"benchmark_report_{int(time.time())}.txt"

        output_path = self.output_dir / filename

        with open(output_path, "w") as f:
            f.write("QuantumDB Benchmark Report\n")
            f.write("=" * 50 + "\n\n")

            # Group results by test type
            grouped_results = {}
            for result in self.results:
                test_type = result.metric_name.split("_")[0]
                if test_type not in grouped_results:
                    grouped_results[test_type] = []
                grouped_results[test_type].append(result)

            # Write results for each test type
            for test_type, test_results in grouped_results.items():
                f.write(f"{test_type.upper()} RESULTS\n")
                f.write("-" * 30 + "\n")

                for result in test_results:
                    f.write(f"{result.metric_name}: {result.value:.4f} {result.unit}\n")

                f.write("\n")

        self.logger.info(f"Report generated: {output_path}")
        return str(output_path)
