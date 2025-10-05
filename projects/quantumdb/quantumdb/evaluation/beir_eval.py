"""
BEIR benchmark evaluation utilities.

This module provides utilities for evaluating models on the BEIR benchmark
datasets using standard information retrieval metrics.
"""

import os
import json
import logging
from typing import Dict, List, Set, Any, Optional, Tuple
from pathlib import Path

import numpy as np
from tqdm import tqdm

try:
    from beir import util
    from beir.datasets.data_loader import GenericDataLoader
    from beir.retrieval.evaluation import EvaluateRetrieval

    BEIR_AVAILABLE = True
except ImportError:
    BEIR_AVAILABLE = False

from .metrics import MetricCalculator, create_default_metrics


class BEIREvaluator:
    """
    Evaluator for BEIR benchmark datasets.

    This class provides utilities to evaluate retrieval models on BEIR datasets
    using standard IR metrics like NDCG@10, Recall@10, MAP, etc.
    """

    def __init__(
        self,
        dataset_name: str,
        data_dir: str = "beir_data",
        split: str = "test",
        k_values: List[int] = [1, 5, 10, 20, 100],
    ):
        if not BEIR_AVAILABLE:
            raise ImportError("BEIR is not installed. Install with: pip install beir")

        self.dataset_name = dataset_name
        self.data_dir = Path(data_dir)
        self.split = split
        self.k_values = k_values
        self.logger = logging.getLogger(__name__)

        # Load dataset
        self._load_dataset()

        # Setup metrics
        self.metric_calculator = create_default_metrics(k_values)

    def _load_dataset(self):
        """Load BEIR dataset."""
        data_path = self.data_dir / self.dataset_name

        if not data_path.exists():
            self.logger.info(f"Downloading {self.dataset_name} dataset...")
            url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{self.dataset_name}.zip"
            util.download_and_unzip(url, self.data_dir)

        self.corpus, self.queries, self.qrels = GenericDataLoader(
            data_folder=data_path
        ).load(split=self.split)

        self.logger.info(f"Loaded {self.dataset_name}:")
        self.logger.info(f"  Corpus size: {len(self.corpus)}")
        self.logger.info(f"  Queries: {len(self.queries)}")
        self.logger.info(f"  Qrels: {len(self.qrels)}")

    def get_corpus_texts(self) -> List[str]:
        """Get corpus document texts."""
        return [doc["text"] for doc in self.corpus.values()]

    def get_corpus_ids(self) -> List[str]:
        """Get corpus document IDs."""
        return list(self.corpus.keys())

    def get_query_texts(self) -> List[str]:
        """Get query texts."""
        return list(self.queries.values())

    def get_query_ids(self) -> List[str]:
        """Get query IDs."""
        return list(self.queries.keys())

    def get_qrels_for_query(self, query_id: str) -> Set[str]:
        """Get relevant document IDs for a query."""
        return set(self.qrels.get(query_id, {}).keys())

    def evaluate_retrieval_results(
        self,
        results: Dict[str, List[Tuple[str, float]]],
        save_path: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Evaluate retrieval results.

        Args:
            results: Dictionary mapping query_id to list of (doc_id, score) tuples
            save_path: Optional path to save results

        Returns:
            metrics: Dictionary of metric scores
        """
        # Prepare data for evaluation
        retrieved_ids = []
        relevant_ids = []

        for query_id in self.queries.keys():
            # Get retrieved documents for this query
            query_results = results.get(query_id, [])
            retrieved_doc_ids = [doc_id for doc_id, _ in query_results]
            retrieved_ids.append(retrieved_doc_ids)

            # Get relevant documents for this query
            relevant_doc_ids = self.get_qrels_for_query(query_id)
            relevant_ids.append(relevant_doc_ids)

        # Compute metrics
        metrics = self.metric_calculator.compute_all(
            retrieved_ids, relevant_ids, self.k_values
        )

        # Log results
        self.logger.info(f"Evaluation results for {self.dataset_name}:")
        for metric_name, score in metrics.items():
            self.logger.info(f"  {metric_name}: {score:.4f}")

        # Save results if requested
        if save_path:
            self._save_results(metrics, save_path)

        return metrics

    def evaluate_model(
        self,
        model,
        embedder,
        batch_size: int = 32,
        save_path: Optional[str] = None,
        cache_embeddings: bool = True,
    ) -> Dict[str, float]:
        """
        Evaluate a model on the BEIR dataset.

        Args:
            model: Retrieval model (e.g., QuantumDB instance)
            embedder: Embedding generator
            batch_size: Batch size for embedding generation
            save_path: Optional path to save results
            cache_embeddings: Whether to cache embeddings

        Returns:
            metrics: Dictionary of metric scores
        """
        # Generate embeddings
        corpus_texts = self.get_corpus_texts()
        query_texts = self.get_query_texts()

        self.logger.info("Generating corpus embeddings...")
        corpus_embeddings = embedder.encode(corpus_texts, batch_size=batch_size)

        self.logger.info("Generating query embeddings...")
        query_embeddings = embedder.encode(query_texts, batch_size=batch_size)

        # Add corpus to model if it's a QuantumDB instance
        if hasattr(model, "add"):
            corpus_ids = self.get_corpus_ids()
            model.add(corpus_embeddings, corpus_ids)

        # Perform retrieval
        results = {}
        corpus_ids = self.get_corpus_ids()
        query_ids = self.get_query_ids()

        self.logger.info("Performing retrieval...")
        for i, (query_id, query_embedding) in enumerate(
            tqdm(zip(query_ids, query_embeddings), total=len(query_ids))
        ):
            if hasattr(model, "search"):
                # QuantumDB search
                search_results = model.search(query_embedding, limit=max(self.k_values))
                results[query_id] = [
                    (doc_id, score) for doc_id, score, _ in search_results
                ]
            else:
                # Custom search logic
                similarities = np.dot(corpus_embeddings, query_embedding)
                top_k_indices = np.argsort(similarities)[::-1][: max(self.k_values)]
                results[query_id] = [
                    (corpus_ids[idx], similarities[idx]) for idx in top_k_indices
                ]

        # Evaluate results
        return self.evaluate_retrieval_results(results, save_path)

    def _save_results(self, metrics: Dict[str, float], save_path: str):
        """Save evaluation results to file."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        results = {
            "dataset": self.dataset_name,
            "split": self.split,
            "k_values": self.k_values,
            "metrics": metrics,
        }

        with open(save_path, "w") as f:
            json.dump(results, f, indent=2)

        self.logger.info(f"Results saved to {save_path}")

    def compare_models(
        self,
        models: Dict[str, Any],
        embedder,
        batch_size: int = 32,
        save_dir: str = "beir_results",
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare multiple models on the BEIR dataset.

        Args:
            models: Dictionary mapping model names to model instances
            embedder: Embedding generator
            batch_size: Batch size for embedding generation
            save_dir: Directory to save comparison results

        Returns:
            comparison: Dictionary mapping model names to their metrics
        """
        comparison = {}
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        for model_name, model in models.items():
            self.logger.info(f"Evaluating model: {model_name}")

            # Create fresh model instance for each evaluation
            if hasattr(model, "recreate_collection"):
                model.recreate_collection()

            # Evaluate model
            save_path = save_dir / f"{self.dataset_name}_{model_name}_results.json"
            metrics = self.evaluate_model(
                model, embedder, batch_size, save_path=str(save_path)
            )

            comparison[model_name] = metrics

        # Save comparison summary
        comparison_path = save_dir / f"{self.dataset_name}_comparison.json"
        with open(comparison_path, "w") as f:
            json.dump(comparison, f, indent=2)

        # Log comparison table
        self._log_comparison_table(comparison)

        return comparison

    def _log_comparison_table(self, comparison: Dict[str, Dict[str, float]]):
        """Log a comparison table of model results."""
        if not comparison:
            return

        # Get all metric names
        metric_names = set()
        for metrics in comparison.values():
            metric_names.update(metrics.keys())
        metric_names = sorted(metric_names)

        # Create table
        self.logger.info(f"\nComparison Table for {self.dataset_name}:")
        self.logger.info("-" * (80 + len(metric_names) * 15))

        # Header
        header = f"{'Model':<20}"
        for metric_name in metric_names:
            header += f"{metric_name:<15}"
        self.logger.info(header)
        self.logger.info("-" * (80 + len(metric_names) * 15))

        # Rows
        for model_name, metrics in comparison.items():
            row = f"{model_name:<20}"
            for metric_name in metric_names:
                score = metrics.get(metric_name, 0.0)
                row += f"{score:<15.4f}"
            self.logger.info(row)

        self.logger.info("-" * (80 + len(metric_names) * 15))


def evaluate_on_beir_datasets(
    model,
    embedder,
    datasets: List[str],
    data_dir: str = "beir_data",
    batch_size: int = 32,
    save_dir: str = "beir_results",
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate a model on multiple BEIR datasets.

    Args:
        model: Model to evaluate
        embedder: Embedding generator
        datasets: List of BEIR dataset names
        data_dir: Directory to store BEIR data
        batch_size: Batch size for embedding generation
        save_dir: Directory to save results

    Returns:
        all_results: Dictionary mapping dataset names to model metrics
    """
    all_results = {}

    for dataset_name in datasets:
        try:
            evaluator = BEIREvaluator(dataset_name, data_dir)

            # Create fresh model instance
            if hasattr(model, "recreate_collection"):
                model.recreate_collection()

            # Evaluate
            save_path = Path(save_dir) / f"{dataset_name}_results.json"
            metrics = evaluator.evaluate_model(
                model, embedder, batch_size, str(save_path)
            )

            all_results[dataset_name] = metrics

        except Exception as e:
            logging.error(f"Error evaluating {dataset_name}: {e}")
            all_results[dataset_name] = {"error": str(e)}

    # Save summary
    summary_path = Path(save_dir) / "beir_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)

    return all_results


# Common BEIR datasets for evaluation
COMMON_BEIR_DATASETS = [
    "trec-covid",
    "nfcorpus",
    "nq",
    "hotpotqa",
    "fiqa",
    "arguana",
    "webis-touche2020",
    "cqadupstack",
    "quora",
    "dbpedia-entity",
    "scidocs",
    "fever",
    "climate-fever",
    "scifact",
]
