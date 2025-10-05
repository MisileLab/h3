"""
Evaluation metrics for vector search systems.

This module implements common information retrieval metrics including
Recall@K, NDCG@K, MRR, Precision, and F1-score.
"""

import numpy as np
from typing import List, Dict, Set, Tuple, Any
from abc import ABC, abstractmethod


class BaseMetric(ABC):
    """Abstract base class for evaluation metrics."""

    @abstractmethod
    def compute(
        self,
        retrieved_ids: List[List[Any]],
        relevant_ids: List[Set[Any]],
        k: Optional[int] = None,
    ) -> float:
        """Compute the metric."""
        pass

    @abstractmethod
    def __str__(self) -> str:
        """Return metric name."""
        pass


class RecallAtK(BaseMetric):
    """Recall@K metric."""

    def __init__(self, k: int = 10):
        self.k = k

    def compute(
        self,
        retrieved_ids: List[List[Any]],
        relevant_ids: List[Set[Any]],
        k: Optional[int] = None,
    ) -> float:
        """
        Compute Recall@K.

        Args:
            retrieved_ids: List of retrieved document IDs for each query
            relevant_ids: List of relevant document ID sets for each query
            k: Cut-off position (uses self.k if None)

        Returns:
            recall: Recall@K score
        """
        k = k or self.k
        recalls = []

        for retrieved, relevant in zip(retrieved_ids, relevant_ids):
            if len(relevant) == 0:
                continue

            retrieved_k = set(retrieved[:k])
            relevant_retrieved = len(relevant.intersection(retrieved_k))
            recall = relevant_retrieved / len(relevant)
            recalls.append(recall)

        return np.mean(recalls) if recalls else 0.0

    def __str__(self) -> str:
        return f"Recall@{self.k}"


class PrecisionAtK(BaseMetric):
    """Precision@K metric."""

    def __init__(self, k: int = 10):
        self.k = k

    def compute(
        self,
        retrieved_ids: List[List[Any]],
        relevant_ids: List[Set[Any]],
        k: Optional[int] = None,
    ) -> float:
        """
        Compute Precision@K.

        Args:
            retrieved_ids: List of retrieved document IDs for each query
            relevant_ids: List of relevant document ID sets for each query
            k: Cut-off position (uses self.k if None)

        Returns:
            precision: Precision@K score
        """
        k = k or self.k
        precisions = []

        for retrieved, relevant in zip(retrieved_ids, relevant_ids):
            retrieved_k = retrieved[:k]
            if len(retrieved_k) == 0:
                continue

            relevant_retrieved = sum(1 for doc_id in retrieved_k if doc_id in relevant)
            precision = relevant_retrieved / len(retrieved_k)
            precisions.append(precision)

        return np.mean(precisions) if precisions else 0.0

    def __str__(self) -> str:
        return f"Precision@{self.k}"


class F1Score(BaseMetric):
    """F1-score metric."""

    def __init__(self, k: int = 10):
        self.k = k

    def compute(
        self,
        retrieved_ids: List[List[Any]],
        relevant_ids: List[Set[Any]],
        k: Optional[int] = None,
    ) -> float:
        """
        Compute F1-score@K.

        Args:
            retrieved_ids: List of retrieved document IDs for each query
            relevant_ids: List of relevant document ID sets for each query
            k: Cut-off position (uses self.k if None)

        Returns:
            f1: F1-score@K
        """
        k = k or self.k
        f1_scores = []

        for retrieved, relevant in zip(retrieved_ids, relevant_ids):
            retrieved_k = retrieved[:k]
            if len(retrieved_k) == 0 or len(relevant) == 0:
                continue

            relevant_retrieved = sum(1 for doc_id in retrieved_k if doc_id in relevant)
            precision = relevant_retrieved / len(retrieved_k)
            recall = relevant_retrieved / len(relevant)

            if precision + recall == 0:
                f1 = 0.0
            else:
                f1 = 2 * precision * recall / (precision + recall)

            f1_scores.append(f1)

        return np.mean(f1_scores) if f1_scores else 0.0

    def __str__(self) -> str:
        return f"F1@{self.k}"


class NDCGAtK(BaseMetric):
    """Normalized Discounted Cumulative Gain@K metric."""

    def __init__(self, k: int = 10):
        self.k = k

    def compute(
        self,
        retrieved_ids: List[List[Any]],
        relevant_ids: List[Set[Any]],
        k: Optional[int] = None,
        relevance_scores: Optional[List[Dict[Any, float]]] = None,
    ) -> float:
        """
        Compute NDCG@K.

        Args:
            retrieved_ids: List of retrieved document IDs for each query
            relevant_ids: List of relevant document ID sets for each query
            k: Cut-off position (uses self.k if None)
            relevance_scores: Optional relevance scores for each document

        Returns:
            ndcg: NDCG@K score
        """
        k = k or self.k
        ndcg_scores = []

        for i, (retrieved, relevant) in enumerate(zip(retrieved_ids, relevant_ids)):
            if len(relevant) == 0:
                continue

            retrieved_k = retrieved[:k]

            # Get relevance scores
            if relevance_scores and i < len(relevance_scores):
                rel_scores = relevance_scores[i]
            else:
                # Binary relevance: 1 for relevant, 0 for non-relevant
                rel_scores = {doc_id: 1.0 for doc_id in relevant}

            # Compute DCG
            dcg = 0.0
            for pos, doc_id in enumerate(retrieved_k):
                relevance = rel_scores.get(doc_id, 0.0)
                dcg += relevance / np.log2(pos + 2)  # +2 because log2(1) = 0

            # Compute IDCG (ideal DCG)
            ideal_relevances = sorted(
                [rel_scores.get(doc_id, 0.0) for doc_id in relevant], reverse=True
            )[:k]

            idcg = 0.0
            for pos, relevance in enumerate(ideal_relevances):
                idcg += relevance / np.log2(pos + 2)

            # Compute NDCG
            if idcg > 0:
                ndcg = dcg / idcg
                ndcg_scores.append(ndcg)

        return np.mean(ndcg_scores) if ndcg_scores else 0.0

    def __str__(self) -> str:
        return f"NDCG@{self.k}"


class MRR(BaseMetric):
    """Mean Reciprocal Rank metric."""

    def compute(
        self,
        retrieved_ids: List[List[Any]],
        relevant_ids: List[Set[Any]],
        k: Optional[int] = None,
    ) -> float:
        """
        Compute Mean Reciprocal Rank.

        Args:
            retrieved_ids: List of retrieved document IDs for each query
            relevant_ids: List of relevant document ID sets for each query
            k: Cut-off position (optional)

        Returns:
            mrr: MRR score
        """
        reciprocal_ranks = []

        for retrieved, relevant in zip(retrieved_ids, relevant_ids):
            if len(relevant) == 0:
                continue

            # Find first relevant document
            for pos, doc_id in enumerate(retrieved):
                if k is not None and pos >= k:
                    break
                if doc_id in relevant:
                    reciprocal_ranks.append(1.0 / (pos + 1))
                    break
            else:
                reciprocal_ranks.append(0.0)

        return np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0

    def __str__(self) -> str:
        return "MRR"


class MAP(BaseMetric):
    """Mean Average Precision metric."""

    def compute(
        self,
        retrieved_ids: List[List[Any]],
        relevant_ids: List[Set[Any]],
        k: Optional[int] = None,
    ) -> float:
        """
        Compute Mean Average Precision.

        Args:
            retrieved_ids: List of retrieved document IDs for each query
            relevant_ids: List of relevant document ID sets for each query
            k: Cut-off position (optional)

        Returns:
            map_score: MAP score
        """
        average_precisions = []

        for retrieved, relevant in zip(retrieved_ids, relevant_ids):
            if len(relevant) == 0:
                continue

            retrieved_k = retrieved[:k] if k else retrieved
            relevant_retrieved = 0
            precisions = []

            for pos, doc_id in enumerate(retrieved_k):
                if doc_id in relevant:
                    relevant_retrieved += 1
                    precision = relevant_retrieved / (pos + 1)
                    precisions.append(precision)

            if precisions:
                ap = np.mean(precisions)
                average_precisions.append(ap)

        return np.mean(average_precisions) if average_precisions else 0.0

    def __str__(self) -> str:
        return "MAP"


class MetricCalculator:
    """Utility class for calculating multiple metrics."""

    def __init__(self, metrics: List[BaseMetric]):
        self.metrics = metrics

    def compute_all(
        self,
        retrieved_ids: List[List[Any]],
        relevant_ids: List[Set[Any]],
        k_values: Optional[List[int]] = None,
        relevance_scores: Optional[List[Dict[Any, float]]] = None,
    ) -> Dict[str, float]:
        """
        Compute all metrics.

        Args:
            retrieved_ids: List of retrieved document IDs for each query
            relevant_ids: List of relevant document ID sets for each query
            k_values: List of k values to evaluate (for metrics that support it)
            relevance_scores: Optional relevance scores

        Returns:
            results: Dictionary of metric names to scores
        """
        results = {}
        k_values = k_values or [1, 5, 10, 20, 100]

        for metric in self.metrics:
            if isinstance(metric, (RecallAtK, PrecisionAtK, F1Score, NDCGAtK)):
                # Compute for multiple k values
                for k in k_values:
                    score = metric.compute(
                        retrieved_ids,
                        relevant_ids,
                        k=k,
                        relevance_scores=relevance_scores,
                    )
                    results[f"{metric.__class__.__name__}@{k}"] = score
            else:
                # Compute single value metric
                score = metric.compute(
                    retrieved_ids, relevant_ids, relevance_scores=relevance_scores
                )
                results[str(metric)] = score

        return results

    def compute_per_query(
        self,
        retrieved_ids: List[List[Any]],
        relevant_ids: List[Set[Any]],
        k_values: Optional[List[int]] = None,
        relevance_scores: Optional[List[Dict[Any, float]]] = None,
    ) -> Dict[str, List[float]]:
        """
        Compute metrics per query.

        Returns:
            per_query_results: Dictionary of metric names to list of scores per query
        """
        results = {}
        k_values = k_values or [1, 5, 10, 20, 100]

        for metric in self.metrics:
            if isinstance(metric, (RecallAtK, PrecisionAtK, F1Score, NDCGAtK)):
                for k in k_values:
                    per_query_scores = []
                    for i, (retrieved, relevant) in enumerate(
                        zip(retrieved_ids, relevant_ids)
                    ):
                        if len(relevant) == 0:
                            per_query_scores.append(0.0)
                            continue

                        # Compute for single query
                        single_retrieved = [retrieved]
                        single_relevant = [relevant]
                        single_relevance = None
                        if relevance_scores and i < len(relevance_scores):
                            single_relevance = [relevance_scores[i]]

                        score = metric.compute(
                            single_retrieved,
                            single_relevant,
                            k=k,
                            relevance_scores=single_relevance,
                        )
                        per_query_scores.append(score)

                    results[f"{metric.__class__.__name__}@{k}"] = per_query_scores
            else:
                per_query_scores = []
                for i, (retrieved, relevant) in enumerate(
                    zip(retrieved_ids, relevant_ids)
                ):
                    if len(relevant) == 0:
                        per_query_scores.append(0.0)
                        continue

                    single_retrieved = [retrieved]
                    single_relevant = [relevant]
                    single_relevance = None
                    if relevance_scores and i < len(relevance_scores):
                        single_relevance = [relevance_scores[i]]

                    score = metric.compute(
                        single_retrieved,
                        single_relevant,
                        relevance_scores=single_relevance,
                    )
                    per_query_scores.append(score)

                results[str(metric)] = per_query_scores

        return results


def create_default_metrics(
    k_values: List[int] = [1, 5, 10, 20, 100],
) -> MetricCalculator:
    """Create a metric calculator with default metrics."""
    metrics = [
        RecallAtK(k=10),  # Will be computed for all k_values
        PrecisionAtK(k=10),  # Will be computed for all k_values
        NDCGAtK(k=10),  # Will be computed for all k_values
        MRR(),
        MAP(),
    ]

    return MetricCalculator(metrics)
