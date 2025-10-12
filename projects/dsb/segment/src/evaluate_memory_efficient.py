"""
Memory-Efficient Evaluation Script for ELECTRA-based Jailbreak Detection Model

This version processes data in small chunks to avoid OOM errors during evaluation.
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


def evaluate_model_memory_efficient(
    model_path: str, 
    config_path: str,
    output_dir: Optional[str] = None,
    batch_size: int = 1,
    chunk_size: int = 100
) -> Dict[str, Any]:
    """
    Memory-efficient model evaluation that processes data in chunks
    
    Args:
        model_path: Path to trained model
        config_path: Path to configuration file
        output_dir: Directory to save evaluation results
        batch_size: Batch size for prediction (keep at 1 for OOM issues)
        chunk_size: Number of samples to process at a time
        
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
    
    logger.info(f"Starting memory-efficient evaluation of model: {model_path}")
    logger.info(f"Results will be saved to: {output_dir}")
    logger.info(f"Using batch_size={batch_size}, chunk_size={chunk_size}")
    
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
    
    # Get raw texts and labels - NO tokenization yet!
    texts = list(test_dataset['prompt'])
    
    # Handle labels
    if 'type' in test_dataset.column_names:
        raw_labels = test_dataset['type']
    else:
        raw_labels = test_dataset['label']
    
    # Convert labels to binary
    binary_labels = []
    for label in raw_labels:
        if isinstance(label, str):
            binary_labels.append(1 if label.lower() in ['jailbreak', 'unsafe', 'malicious', 'harmful'] else 0)
        else:
            binary_labels.append(int(label))
    
    logger.info(f"Processing {len(texts)} samples in chunks of {chunk_size}")
    
    # Process in chunks to avoid memory issues
    all_pred_labels = []
    all_pred_probs = []
    
    for chunk_start in range(0, len(texts), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(texts))
        chunk_texts = texts[chunk_start:chunk_end]
        
        logger.info(f"Processing chunk {chunk_start//chunk_size + 1}: samples {chunk_start}-{chunk_end}")
        
        # Process chunk in batches
        chunk_pred_labels = []
        chunk_pred_probs = []
        
        for i in range(0, len(chunk_texts), batch_size):
            batch_texts = chunk_texts[i:i + batch_size]
            
            try:
                predictions = model.predict(batch_texts, return_probabilities=True)
                chunk_pred_labels.extend(predictions['predictions'])
                chunk_pred_probs.extend(predictions['unsafe_probabilities'])
                
                # Aggressive memory cleanup
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.error(f"OOM error on batch {i}, trying even smaller batch...")
                    # Try single sample at a time
                    for text in batch_texts:
                        try:
                            pred = model.predict([text], return_probabilities=True)
                            chunk_pred_labels.extend(pred['predictions'])
                            chunk_pred_probs.extend(pred['unsafe_probabilities'])
                        except RuntimeError as e2:
                            logger.error(f"Failed to process single text: {e2}")
                            # Skip this sample but continue
                            continue
                else:
                    raise e
        
        # Add chunk results
        all_pred_labels.extend(chunk_pred_labels)
        all_pred_probs.extend(chunk_pred_probs)
        
        # Clear chunk memory
        del chunk_pred_labels, chunk_pred_probs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info(f"Completed chunk {chunk_start//chunk_size + 1}")
    
    # Ensure we have predictions for all samples
    if len(all_pred_labels) != len(binary_labels):
        logger.warning(f"Prediction count mismatch: {len(all_pred_labels)} vs {len(binary_labels)}")
        min_len = min(len(all_pred_labels), len(binary_labels))
        all_pred_labels = all_pred_labels[:min_len]
        all_pred_probs = all_pred_probs[:min_len]
        binary_labels = binary_labels[:min_len]
    
    # Calculate metrics
    accuracy = np.mean(np.array(all_pred_labels) == np.array(binary_labels))
    
    # Detailed classification report
    class_report = classification_report(
        binary_labels, 
        all_pred_labels, 
        target_names=['Safe', 'Unsafe'],
        output_dict=True
    )
    
    # ROC AUC
    roc_auc = roc_auc_score(binary_labels, all_pred_probs)
    
    # Confusion matrix
    cm = confusion_matrix(binary_labels, all_pred_labels)
    
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
        'evaluation_timestamp': datetime.now().isoformat(),
        'memory_efficient_mode': True,
        'chunk_size': chunk_size,
        'batch_size': batch_size
    }
    
    # Save detailed results
    results_path = os.path.join(output_dir, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Generate visualizations
    try:
        from src.evaluate import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
        
        plot_confusion_matrix(
            np.array(binary_labels), 
            np.array(all_pred_labels),
            os.path.join(output_dir, 'confusion_matrix.png')
        )
        
        plot_roc_curve(
            np.array(binary_labels), 
            np.array(all_pred_probs),
            os.path.join(output_dir, 'roc_curve.png')
        )
        
        plot_precision_recall_curve(
            np.array(binary_labels), 
            np.array(all_pred_probs),
            os.path.join(output_dir, 'precision_recall_curve.png')
        )
    except Exception as e:
        logger.warning(f"Could not generate visualizations: {e}")
    
    # Log summary
    logger.info("Memory-efficient evaluation completed!")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"ROC AUC: {roc_auc:.4f}")
    logger.info(f"Precision (Unsafe): {class_report['Unsafe']['precision']:.4f}")
    logger.info(f"Recall (Unsafe): {class_report['Unsafe']['recall']:.4f}")
    logger.info(f"F1-Score (Unsafe): {class_report['Unsafe']['f1-score']:.4f}")
    
    return results


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Memory-efficient evaluation for ELECTRA-based jailbreak detection model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
    parser.add_argument("--output_dir", type=str, help="Directory to save evaluation results")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for evaluation (default: 1)")
    parser.add_argument("--chunk_size", type=int, default=100, help="Number of samples to process at once (default: 100)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Set debug level if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        results = evaluate_model_memory_efficient(
            args.model_path, 
            args.config, 
            args.output_dir, 
            args.batch_size,
            args.chunk_size
        )
        logger.info("Evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()