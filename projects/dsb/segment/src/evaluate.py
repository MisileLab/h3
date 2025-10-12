"""
Evaluation Script for ELECTRA-based Jailbreak Detection Model

This module implements comprehensive evaluation with visualization including
confusion matrices, ROC curves, and detailed error analysis.
"""

import argparse
import os
import sys
import yaml
import json
import logging
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.metrics import (
    classification_report, 
    roc_auc_score, 
    confusion_matrix,
    precision_recall_curve,
    auc
)
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from src.model import SegmentModel
from data.datasets import JailbreakDataset
from src.utils import setup_logging, get_device

logger = logging.getLogger(__name__)


def plot_confusion_matrix(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    save_path: str,
    labels: Optional[List[str]] = None
):
    """
    Plot and save confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        save_path: Path to save the plot
        labels: Label names for the plot
    """
    if labels is None:
        labels = ['Safe', 'Unsafe']
        
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels
    )
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Confusion matrix saved to: {save_path}")


def plot_roc_curve(
    y_true: np.ndarray, 
    y_proba: np.ndarray, 
    save_path: str
):
    """
    Plot and save ROC curve
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities for positive class
        save_path: Path to save the plot
    """
    from sklearn.metrics import roc_curve
    
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"ROC curve saved to: {save_path}")


def plot_precision_recall_curve(
    y_true: np.ndarray, 
    y_proba: np.ndarray, 
    save_path: str
):
    """
    Plot and save Precision-Recall curve
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities for positive class
        save_path: Path to save the plot
    """
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, 
             label=f'PR curve (AUC = {pr_auc:.2f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Precision-Recall curve saved to: {save_path}")


def analyze_false_predictions(
    model: SegmentModel, 
    tokenizer,
    test_data: List[Dict[str, Any]], 
    threshold: float = 0.5,
    max_examples: int = 10
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Analyze false positive and false negative predictions
    
    Args:
        model: Trained model
        tokenizer: Model tokenizer
        test_data: Test dataset
        threshold: Classification threshold
        max_examples: Maximum number of examples to analyze
        
    Returns:
        Dictionary with false positive and false negative examples
    """
    texts = [item['text'] for item in test_data]
    true_labels = [item['label'] for item in test_data]
    
    # Get predictions
    predictions = model.predict(texts, return_probabilities=True)
    pred_labels = predictions['predictions']
    pred_probs = predictions['unsafe_probabilities']
    
    false_positives = []
    false_negatives = []
    
    for i, (text, true_label, pred_label, pred_prob) in enumerate(
        zip(texts, true_labels, pred_labels, pred_probs)
    ):
        if len(false_positives) >= max_examples and len(false_negatives) >= max_examples:
            break
            
        # False positive: predicted unsafe but actually safe
        if pred_label == 1 and true_label == 0 and len(false_positives) < max_examples:
            false_positives.append({
                'text': text,
                'true_label': 'safe',
                'predicted_label': 'unsafe',
                'probability': pred_prob,
                'index': i
            })
            
        # False negative: predicted safe but actually unsafe
        elif pred_label == 0 and true_label == 1 and len(false_negatives) < max_examples:
            false_negatives.append({
                'text': text,
                'true_label': 'unsafe',
                'predicted_label': 'safe',
                'probability': pred_prob,
                'index': i
            })
    
    return {
        'false_positives': false_positives,
        'false_negatives': false_negatives
    }


def evaluate_model(
    model_path: str, 
    config_path: str,
    output_dir: Optional[str] = None,
    batch_size: int = 16
) -> Dict[str, Any]:
    """
    Comprehensive model evaluation
    
    Args:
        model_path: Path to trained model
        config_path: Path to configuration file
        output_dir: Directory to save evaluation results
        
    Returns:
        Dictionary with evaluation results
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup output directory
    if output_dir is None:
        output_dir = os.path.join(
            config.get('logging', {}).get('log_dir', './experiments/logs'),
            'evaluation',
            datetime.now().strftime('%Y%m%d_%H%M%S')
        )
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup logging
    setup_logging(output_dir)
    
    logger.info(f"Starting evaluation of model: {model_path}")
    logger.info(f"Results will be saved to: {output_dir}")
    
    # Load model
    device = get_device(config)
    model = SegmentModel.load_model(model_path, device=device)
    
    # Prepare test dataset
    dataset_manager = JailbreakDataset(config)
    datasets = dataset_manager.dataset
    
    # Use test split if available, otherwise use validation split
    if 'test' in datasets:
        test_dataset = datasets['test']
    elif 'validation' in datasets:
        test_dataset = datasets['validation']
        logger.warning("Test split not found, using validation split")
    else:
        # Split train dataset
        train_data = datasets['train']
        train_size = int(0.8 * len(train_data))
        test_dataset = train_data.select(range(train_size, len(train_data)))
        logger.warning("No test/validation split found, using portion of train data")
    
    # Preprocess test data
    label2id = {"benign": 0, "jailbreak": 1}
    if 'type' in test_dataset.column_names:
        test_dataset = test_dataset.map(lambda x: {"label": label2id.get(x["type"], -1)})

    processed_test = test_dataset.map(
        lambda x: dataset_manager._preprocess_function(x, model.tokenizer),
        batched=True,
    )
    
    # Get texts and labels for analysis
    texts = list(processed_test['prompt'])
    labels = processed_test['type'] if 'type' in processed_test.column_names else processed_test['label']
    
    # Convert labels to binary
    binary_labels = []
    for label in labels:
        if isinstance(label, str):
            binary_labels.append(1 if label.lower() in ['jailbreak', 'unsafe', 'malicious', 'harmful'] else 0)
        else:
            binary_labels.append(int(label))
    
    # Make predictions in batches to avoid OOM
    pred_labels = []
    pred_probs = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        predictions = model.predict(batch_texts, return_probabilities=True)
        pred_labels.extend(predictions['predictions'])
        pred_probs.extend(predictions['unsafe_probabilities'])
    
    # Calculate metrics
    accuracy = np.mean(np.array(pred_labels) == np.array(binary_labels))
    
    # Detailed classification report
    class_report = classification_report(
        binary_labels, 
        pred_labels, 
        target_names=['Safe', 'Unsafe'],
        output_dict=True
    )
    
    # ROC AUC
    roc_auc = roc_auc_score(binary_labels, pred_probs)
    
    # Confusion matrix
    cm = confusion_matrix(binary_labels, pred_labels)
    
    # Compile results
    results = {
        'accuracy': float(accuracy),
        'roc_auc': float(roc_auc),
        'classification_report': class_report,
        'confusion_matrix': cm.tolist(),
        'total_samples': len(binary_labels),
        'safe_samples': int(np.sum(np.array(binary_labels) == 0)),
        'unsafe_samples': int(np.sum(np.array(binary_labels) == 1)),
        'model_path': model_path,
        'evaluation_timestamp': datetime.now().isoformat()
    }
    
    # Save detailed results
    results_path = os.path.join(output_dir, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Generate visualizations
    plot_confusion_matrix(
        np.array(binary_labels), 
        np.array(pred_labels),
        os.path.join(output_dir, 'confusion_matrix.png')
    )
    
    plot_roc_curve(
        np.array(binary_labels), 
        np.array(pred_probs),
        os.path.join(output_dir, 'roc_curve.png')
    )
    
    plot_precision_recall_curve(
        np.array(binary_labels), 
        np.array(pred_probs),
        os.path.join(output_dir, 'precision_recall_curve.png')
    )
    
    # Analyze false predictions
    test_data_list = [{'text': text, 'label': label} for text, label in zip(texts, binary_labels)]
    false_predictions = analyze_false_predictions(
        model, model.tokenizer, test_data_list, max_examples=20
    )
    
    # Save false prediction analysis
    false_predictions_path = os.path.join(output_dir, 'false_predictions.json')
    with open(false_predictions_path, 'w') as f:
        json.dump(false_predictions, f, indent=2)
    
    # Log summary
    logger.info("Evaluation completed!")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"ROC AUC: {roc_auc:.4f}")
    logger.info(f"Precision (Unsafe): {class_report['unsafe']['precision']:.4f}")
    logger.info(f"Recall (Unsafe): {class_report['unsafe']['recall']:.4f}")
    logger.info(f"F1-Score (Unsafe): {class_report['unsafe']['f1-score']:.4f}")
    logger.info(f"False Positives: {false_predictions['false_positives'].__len__()}")
    logger.info(f"False Negatives: {false_predictions['false_negatives'].__len__()}")
    
    return results


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Evaluate ELECTRA-based jailbreak detection model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
    parser.add_argument("--output_dir", type=str, help="Directory to save evaluation results")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for evaluation")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Set debug level if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        results = evaluate_model(args.model_path, args.config, args.output_dir, args.batch_size)
        logger.info("Evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()
