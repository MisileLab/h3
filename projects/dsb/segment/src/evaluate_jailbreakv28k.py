"""
Evaluation Script for JailbreakV-28K Dataset (Text-Only)

This module implements evaluation of jailbreak detection models on the 
JailbreakV-28K dataset, focusing on text-based jailbreak queries.
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
from datasets import load_dataset

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
from src.utils import setup_logging, get_device

logger = logging.getLogger(__name__)


def load_jailbreakv28k_dataset(
    subset: str = "mini_JailBreakV_28K",
    split: Optional[str] = None
) -> pd.DataFrame:
    """
    Load JailbreakV-28K dataset from Hugging Face
    
    Args:
        subset: Dataset subset ('mini_JailBreakV_28K', 'JailBreakV_28K', 'RedTeam_2K')
        split: Dataset split (if applicable)
        
    Returns:
        DataFrame with dataset
    """
    logger.info(f"Loading JailbreakV-28K dataset: {subset}")
    
    try:
        dataset = load_dataset("JailbreakV-28K/JailBreakV-28k", subset)
        
        if split:
            if split in dataset:
                data = dataset[split].to_pandas()
            else:
                logger.warning(f"Split '{split}' not found, using full dataset")
                data = dataset[subset].to_pandas()
        else:
            data = dataset[subset].to_pandas()
            
        logger.info(f"Loaded {len(data)} samples from {subset}")
        return data
        
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise


def preprocess_jailbreak_data(
    df: pd.DataFrame,
    text_column: str = "jailbreak_query",
    label_column: Optional[str] = None
) -> Tuple[List[str], List[int]]:
    """
    Preprocess jailbreak dataset for evaluation
    
    Args:
        df: Input DataFrame
        text_column: Column containing text queries
        label_column: Column containing labels (if None, all labeled as jailbreak)
        
    Returns:
        Tuple of (texts, labels)
    """
    logger.info("Preprocessing jailbreak dataset")
    
    # Handle missing text column
    if text_column not in df.columns:
        available_columns = df.columns.tolist()
        logger.error(f"Text column '{text_column}' not found. Available columns: {available_columns}")
        raise ValueError(f"Text column '{text_column}' not found")
    
    # Extract texts
    texts = df[text_column].fillna("").astype(str).tolist()
    
    # Generate labels
    if label_column and label_column in df.columns:
        # Use provided labels
        labels = df[label_column].fillna(0).astype(int).tolist()
    else:
        # All jailbreak queries are labeled as unsafe (1)
        labels = [1] * len(texts)
        logger.info("All samples labeled as jailbreak (unsafe)")
    
    # Filter out empty texts
    valid_indices = [i for i, text in enumerate(texts) if text.strip()]
    texts = [texts[i] for i in valid_indices]
    labels = [labels[i] for i in valid_indices]
    
    logger.info(f"Processed {len(texts)} valid samples")
    logger.info(f"Label distribution - Safe: {labels.count(0)}, Unsafe: {labels.count(1)}")
    
    return texts, labels


def create_balanced_test_set(
    jailbreak_texts: List[str],
    jailbreak_labels: List[int],
    safe_samples: int = 1000
) -> Tuple[List[str], List[int]]:
    """
    Create a balanced test set with safe samples
    
    Args:
        jailbreak_texts: Jailbreak query texts
        jailbreak_labels: Jailbreak labels (should all be 1)
        safe_samples: Number of safe samples to generate
        
    Returns:
        Tuple of (texts, labels) for balanced test set
    """
    logger.info("Creating balanced test set")
    
    # For now, use simple safe queries (in practice, you'd load benign queries)
    safe_texts = [
        "What is the weather like today?",
        "Can you help me with my homework?",
        "Tell me about the history of computers.",
        "How do I cook pasta?",
        "What are the benefits of exercise?",
        "Explain the concept of machine learning.",
        "Can you recommend a good book?",
        "What is the capital of France?",
        "How do plants perform photosynthesis?",
        "Tell me about renewable energy sources."
    ] * (safe_samples // 10 + 1)
    
    safe_texts = safe_texts[:safe_samples]
    safe_labels = [0] * len(safe_texts)
    
    # Combine datasets
    all_texts = jailbreak_texts + safe_texts
    all_labels = jailbreak_labels + safe_labels
    
    logger.info(f"Created balanced test set: {len(all_texts)} samples")
    logger.info(f"Label distribution - Safe: {all_labels.count(0)}, Unsafe: {all_labels.count(1)}")
    
    return all_texts, all_labels


def evaluate_on_jailbreakv28k(
    model_path: str,
    config_path: str,
    subset: str = "mini_JailBreakV_28K",
    output_dir: Optional[str] = None,
    batch_size: int = 16,
    create_balanced: bool = True
) -> Dict[str, Any]:
    """
    Evaluate model on JailbreakV-28K dataset
    
    Args:
        model_path: Path to trained model
        config_path: Path to configuration file
        subset: Dataset subset to evaluate on
        output_dir: Directory to save results
        batch_size: Batch size for inference
        create_balanced: Whether to create balanced test set
        
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
            'jailbreakv28k_evaluation',
            datetime.now().strftime('%Y%m%d_%H%M%S')
        )
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup logging
    setup_logging(output_dir)
    
    logger.info(f"Starting JailbreakV-28K evaluation")
    logger.info(f"Model: {model_path}")
    logger.info(f"Dataset subset: {subset}")
    logger.info(f"Output directory: {output_dir}")
    
    # Load model
    device = get_device(config)
    model = SegmentModel.load_model(model_path, device=device)
    
    # Load and preprocess dataset
    df = load_jailbreakv28k_dataset(subset)
    
    # Determine text column based on subset
    if subset == "RedTeam_2K":
        text_column = "question"
    else:
        text_column = "jailbreak_query"
    
    texts, labels = preprocess_jailbreak_data(df, text_column)
    
    # Create balanced test set if requested
    if create_balanced and labels.count(1) > 0:
        texts, labels = create_balanced_test_set(texts, labels)
    
    # Make predictions in batches
    logger.info("Making predictions...")
    pred_labels = []
    pred_probs = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        predictions = model.predict(batch_texts, return_probabilities=True)
        pred_labels.extend(predictions['predictions'])
        pred_probs.extend(predictions['unsafe_probabilities'])
        
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        if (i // batch_size + 1) % 10 == 0:
            logger.info(f"Processed {min(i + batch_size, len(texts))}/{len(texts)} samples")
    
    # Calculate metrics
    accuracy = np.mean(np.array(pred_labels) == np.array(labels))
    
    # Detailed classification report
    class_report = classification_report(
        labels, 
        pred_labels, 
        target_names=['Safe', 'Unsafe'],
        output_dict=True
    )
    
    # ROC AUC (only if we have both classes)
    if len(set(labels)) > 1:
        roc_auc = roc_auc_score(labels, pred_probs)
    else:
        roc_auc = float('nan')
        logger.warning("Only one class present, ROC AUC not defined")
    
    # Confusion matrix
    cm = confusion_matrix(labels, pred_labels)
    
    # Attack Success Rate (ASR) - specific to jailbreak evaluation
    false_negatives = cm[1, 0]  # Actual unsafe predicted as safe
    total_unsafe = cm[1, 0] + cm[1, 1]  # Total actual unsafe
    asr = false_negatives / total_unsafe if total_unsafe > 0 else 0.0
    
    # Compile results
    results = {
        'dataset': {
            'name': 'JailbreakV-28K',
            'subset': subset,
            'total_samples': len(labels),
            'safe_samples': int(np.sum(np.array(labels) == 0)),
            'unsafe_samples': int(np.sum(np.array(labels) == 1)),
            'balanced_test_set': create_balanced
        },
        'metrics': {
            'accuracy': float(accuracy),
            'roc_auc': float(roc_auc),
            'attack_success_rate': float(asr),
            'classification_report': class_report,
            'confusion_matrix': cm.tolist()
        },
        'model_info': {
            'model_path': model_path,
            'evaluation_timestamp': datetime.now().isoformat()
        }
    }
    
    # Save detailed results
    results_path = os.path.join(output_dir, 'jailbreakv28k_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Generate visualizations
    plot_confusion_matrix(
        np.array(labels), 
        np.array(pred_labels),
        os.path.join(output_dir, 'confusion_matrix.png')
    )
    
    if len(set(labels)) > 1:
        plot_roc_curve(
            np.array(labels), 
            np.array(pred_probs),
            os.path.join(output_dir, 'roc_curve.png')
        )
        
        plot_precision_recall_curve(
            np.array(labels), 
            np.array(pred_probs),
            os.path.join(output_dir, 'precision_recall_curve.png')
        )
    
    # Save sample predictions for analysis
    sample_results = []
    for i in range(min(100, len(texts))):
        sample_results.append({
            'index': i,
            'text': texts[i][:200] + "..." if len(texts[i]) > 200 else texts[i],
            'true_label': 'Safe' if labels[i] == 0 else 'Unsafe',
            'predicted_label': 'Safe' if pred_labels[i] == 0 else 'Unsafe',
            'unsafe_probability': pred_probs[i]
        })
    
    sample_results_path = os.path.join(output_dir, 'sample_predictions.json')
    with open(sample_results_path, 'w') as f:
        json.dump(sample_results, f, indent=2)
    
    # Log summary
    logger.info("Evaluation completed!")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Attack Success Rate (ASR): {asr:.4f}")
    if not np.isnan(roc_auc):
        logger.info(f"ROC AUC: {roc_auc:.4f}")
    logger.info(f"Precision (Unsafe): {class_report['Unsafe']['precision']:.4f}")
    logger.info(f"Recall (Unsafe): {class_report['Unsafe']['recall']:.4f}")
    logger.info(f"F1-Score (Unsafe): {class_report['Unsafe']['f1-score']:.4f}")
    
    tn, fp, fn, tp = cm.ravel()
    logger.info(f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    
    return results


def plot_confusion_matrix(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    save_path: str,
    labels: Optional[List[str]] = None
):
    """Plot and save confusion matrix"""
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
    plt.title('Confusion Matrix - JailbreakV-28K Evaluation')
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
    """Plot and save ROC curve"""
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
    plt.title('ROC Curve - JailbreakV-28K Evaluation')
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
    """Plot and save Precision-Recall curve"""
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, 
             label=f'PR curve (AUC = {pr_auc:.2f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve - JailbreakV-28K Evaluation')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Precision-Recall curve saved to: {save_path}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Evaluate model on JailbreakV-28K dataset")
    parser.add_argument("--model_path", type=str, required=True, 
                       help="Path to trained model")
    parser.add_argument("--config", type=str, required=True, 
                       help="Path to configuration file")
    parser.add_argument("--subset", type=str, default="mini_JailBreakV_28K",
                       choices=["mini_JailBreakV_28K", "JailBreakV_28K", "RedTeam_2K"],
                       help="Dataset subset to evaluate on")
    parser.add_argument("--output_dir", type=str, 
                       help="Directory to save evaluation results")
    parser.add_argument("--batch_size", type=int, default=16, 
                       help="Batch size for evaluation")
    parser.add_argument("--no_balanced", action="store_true",
                       help="Don't create balanced test set")
    parser.add_argument("--debug", action="store_true", 
                       help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Set debug level if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        results = evaluate_on_jailbreakv28k(
            model_path=args.model_path,
            config_path=args.config,
            subset=args.subset,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            create_balanced=not args.no_balanced
        )
        logger.info("JailbreakV-28K evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()