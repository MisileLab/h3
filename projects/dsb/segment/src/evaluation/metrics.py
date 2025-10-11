"""
Evaluation metrics for jailbreak detection.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
from typing import Dict, List, Tuple
import torch


class JailbreakMetrics:
    """Metrics calculator for jailbreak detection."""
    
    def __init__(self):
        self.threshold = 0.5
    
    def compute_metrics(self, eval_pred) -> Dict[str, float]:
        """Compute metrics for evaluation."""
        predictions, labels = eval_pred
        
        # Convert logits to probabilities
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        
        # Get probabilities for positive class (unsafe)
        probs = torch.softmax(torch.tensor(predictions), dim=-1)[:, 1].numpy()
        
        # Get predicted labels
        pred_labels = (probs > self.threshold).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(labels, pred_labels)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, pred_labels, average='binary', zero_division=0
        )
        
        # Calculate AUC
        try:
            auc = roc_auc_score(labels, probs)
        except ValueError:
            auc = 0.0
        
        # Calculate confusion matrix
        tn, fp, fn, tp = confusion_matrix(labels, pred_labels).ravel()
        
        # Calculate additional metrics
        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auc": auc,
            "true_positives": tp,
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "false_positive_rate": false_positive_rate,
            "false_negative_rate": false_negative_rate
        }
    
    def compute_threshold_metrics(self, predictions: np.ndarray, labels: np.ndarray, 
                                 thresholds: List[float] = None) -> Dict[str, List]:
        """Compute metrics across different thresholds."""
        if thresholds is None:
            thresholds = np.arange(0.1, 1.0, 0.1)
        
        # Convert logits to probabilities
        probs = torch.softmax(torch.tensor(predictions), dim=-1)[:, 1].numpy()
        
        results = {
            "threshold": [],
            "accuracy": [],
            "precision": [],
            "recall": [],
            "f1": [],
            "false_positive_rate": [],
            "false_negative_rate": []
        }
        
        for threshold in thresholds:
            pred_labels = (probs > threshold).astype(int)
            
            accuracy = accuracy_score(labels, pred_labels)
            precision, recall, f1, _ = precision_recall_fscore_support(
                labels, pred_labels, average='binary', zero_division=0
            )
            
            tn, fp, fn, tp = confusion_matrix(labels, pred_labels).ravel()
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
            
            results["threshold"].append(threshold)
            results["accuracy"].append(accuracy)
            results["precision"].append(precision)
            results["recall"].append(recall)
            results["f1"].append(f1)
            results["false_positive_rate"].append(fpr)
            results["false_negative_rate"].append(fnr)
        
        return results
    
    def get_classification_report(self, predictions: np.ndarray, labels: np.ndarray) -> str:
        """Get detailed classification report."""
        probs = torch.softmax(torch.tensor(predictions), dim=-1)[:, 1].numpy()
        pred_labels = (probs > self.threshold).astype(int)
        
        return classification_report(
            labels, 
            pred_labels, 
            target_names=["safe", "unsafe"],
            digits=4
        )
    
    def find_optimal_threshold(self, predictions: np.ndarray, labels: np.ndarray) -> Tuple[float, Dict]:
        """Find optimal threshold based on F1 score."""
        probs = torch.softmax(torch.tensor(predictions), dim=-1)[:, 1].numpy()
        
        thresholds = np.arange(0.1, 1.0, 0.01)
        best_f1 = 0
        best_threshold = 0.5
        best_metrics = {}
        
        for threshold in thresholds:
            pred_labels = (probs > threshold).astype(int)
            precision, recall, f1, _ = precision_recall_fscore_support(
                labels, pred_labels, average='binary', zero_division=0
            )
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
                best_metrics = {
                    "threshold": threshold,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "accuracy": accuracy_score(labels, pred_labels)
                }
        
        return best_threshold, best_metrics