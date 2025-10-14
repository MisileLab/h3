"""
Training Script for ELECTRA-based Jailbreak Detection Model

This module implements the training pipeline with early stopping, logging,
and support for both LoRA and full fine-tuning.
"""

import argparse
import os
import sys
import yaml
import json
import logging
from typing import Dict, Any, Optional
import numpy as np
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import (
    Trainer, 
    TrainingArguments, 
    EarlyStoppingCallback,
    get_linear_schedule_with_warmup
)
from transformers.trainer_utils import EvalPrediction
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import torch
from torch.utils.tensorboard import SummaryWriter

from src.model import SegmentModel
from data.datasets import JailbreakDataset
from src.data_augmentation import AdversarialAugmenter
from src.utils import setup_logging, save_config, get_device

logger = logging.getLogger(__name__)


def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
    """
    Compute evaluation metrics
    
    Args:
        eval_pred: Evaluation predictions
        
    Returns:
        Dictionary of metrics
    """
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    # Calculate metrics
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary'
    )
    
    # Calculate ROC AUC if we have probability scores
    try:
        # For ROC AUC, we need the probability of the positive class
        if hasattr(eval_pred, 'predictions'):
            probs = torch.softmax(torch.tensor(eval_pred.predictions), dim=-1)
            roc_auc = roc_auc_score(labels, probs[:, 1].numpy())
        else:
            roc_auc = 0.0
    except:
        roc_auc = 0.0
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc
    }
    
    # Debug logging
    logger.debug(f"[DEBUG] Computed metrics: {metrics}")
    
    return metrics


class CustomEarlyStoppingCallback(EarlyStoppingCallback):
    """Custom early stopping callback with detailed logging"""
    
    def __init__(self, patience: int = 5, threshold: float = 0.001):
        super().__init__(early_stopping_patience=patience)
        self.threshold = threshold
        self.best_metric = None
        self.counter = 0
        self.patience = patience
        
    def on_evaluate(self, args, state, control, model, logs=None, **kwargs):
        """
        Check for early stopping condition
        """
        # Debug logging
        logger.debug(f"[DEBUG] Early stopping callback - logs: {logs}")
        logger.debug(f"[DEBUG] State log history: {state.log_history[-5:] if state.log_history else 'None'}")
        
        metric_to_check = args.metric_for_best_model
        current_metric = None
        
        # Try to get metric from logs first
        if logs:
            current_metric = logs.get(metric_to_check)
            if current_metric is None:
                current_metric = logs.get(f"eval_{metric_to_check}")
            if current_metric is None and metric_to_check == "loss":
                current_metric = logs.get("eval_loss")
        
        # If not found in logs, try to get from the most recent evaluation in state.log_history
        if current_metric is None and state.log_history:
            # Get the most recent evaluation log
            for log in reversed(state.log_history):
                if any(key in log for key in [metric_to_check, f"eval_{metric_to_check}", "eval_loss"]):
                    current_metric = log.get(metric_to_check) or log.get(f"eval_{metric_to_check}") or log.get("eval_loss")
                    logger.debug(f"[DEBUG] Found metric in log_history: {current_metric}")
                    break
        
        if current_metric is None:
            logger.warning(f"Metric {metric_to_check} not found in evaluation logs")
            if logs:
                logger.info(f"Available in logs: {list(logs.keys())}")
            elif state.log_history:
                last_log = state.log_history[-1]
                logger.info(f"Available in last log_history: {list(last_log.keys())}")
            else:
                logger.info("No logs available")
            return
            
        # Check if we should save the model
        if self.best_metric is None:
            self.best_metric = current_metric
            logger.info(f"New best {metric_to_check}: {self.best_metric:.4f}")
        else:
            improvement = current_metric - self.best_metric
            
            if improvement > self.threshold:
                self.best_metric = current_metric
                self.counter = 0
                logger.info(f"New best {metric_to_check}: {self.best_metric:.4f} (improvement: {improvement:.4f})")
            else:
                self.counter += 1
                logger.info(f"No improvement. Counter: {self.counter}/{self.patience}")
                
                if self.counter >= self.patience:
                    logger.info(f"Early stopping triggered after {self.counter} evaluations without improvement")
                    control.should_training_stop = True


def setup_data_augmentation(config: Dict[str, Any]) -> Optional[AdversarialAugmenter]:
    """
    Setup data augmentation if enabled
    
    Args:
        config: Configuration dictionary
        
    Returns:
        AdversarialAugmenter instance or None
    """
    if config.get('data', {}).get('augmentation_enabled', False):
        aug_config = config.get('augmentation', {})
        return AdversarialAugmenter(aug_config)
    return None


def train(config_path: str, use_lora: bool = False, resume_from_checkpoint: Optional[str] = None):
    """
    Main training function
    
    Args:
        config_path: Path to configuration file
        use_lora: Whether to use LoRA fine-tuning
        resume_from_checkpoint: Path to checkpoint to resume from
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    # Setup logging
    log_dir = config.get('logging', {}).get('log_dir', './experiments/logs')
    os.makedirs(log_dir, exist_ok=True)
    setup_logging(log_dir)
    
    logger.info(f"Starting training with LoRA: {use_lora}")
    logger.info(f"Configuration loaded from: {config_path}")
    
    # Save configuration for reproducibility
    save_config(config, log_dir)
    
    # Initialize model
    device = get_device(config)
    model = SegmentModel(config, use_lora=use_lora, device=device)
    
    # Log model info
    model_info = model.get_model_info()
    logger.info(f"Model info: {json.dumps(model_info, indent=2)}")
    
    # Setup data augmentation
    augmenter = setup_data_augmentation(config)
    
    # Prepare datasets
    dataset_manager = JailbreakDataset(config)
    train_dataset, eval_dataset = dataset_manager.prepare_datasets(model.tokenizer)
    
    # Apply data augmentation if enabled
    if augmenter and config.get('data', {}).get('augmentation_enabled', False):
        logger.info("Applying data augmentation...")
        # Note: In practice, you might want to apply augmentation during training
        # This is a placeholder for augmentation logic
        pass
    
    # Setup training arguments
    training_config = config.get('training', {})
    
    # DEBUG: Check learning rate type
    lr = training_config.get('learning_rate')
    logger.info(f"[DEBUG] Learning rate from config: {lr} (type: {type(lr)})")
    
    # Ensure learning rate is a float
    try:
        learning_rate = float(lr)
    except (ValueError, TypeError):
        logger.error(f"Could not convert learning rate to float: {lr}")
        raise
        
    training_args = TrainingArguments(
        output_dir=training_config.get('output_dir', './models/checkpoints'),
        num_train_epochs=training_config.get('num_epochs', 100),
        per_device_train_batch_size=training_config.get('batch_size', 8),
        per_device_eval_batch_size=training_config.get('batch_size', 16),
        gradient_accumulation_steps=training_config.get('gradient_accumulation_steps', 4),
        learning_rate=learning_rate,  # Use validated learning rate
        weight_decay=training_config.get('weight_decay', 0.01),
        warmup_ratio=training_config.get('warmup_ratio', 0.05),
        max_grad_norm=training_config.get('max_grad_norm', 1.0),
        fp16=training_config.get('fp16', True),
        logging_steps=training_config.get('logging_steps', 100),
        save_steps=training_config.get('save_steps', 500),
        eval_steps=training_config.get('eval_steps', 500),
        save_total_limit=training_config.get('save_total_limit', 3),
        load_best_model_at_end=training_config.get('load_best_model_at_end', True),
        metric_for_best_model=training_config.get('metric_for_best_model', 'accuracy'),
        greater_is_better=training_config.get('greater_is_better', True),
        dataloader_num_workers=training_config.get('dataloader_num_workers', 4),
        remove_unused_columns=training_config.get('remove_unused_columns', False),
        eval_strategy="steps",
        save_strategy="steps",
        logging_dir=os.path.join(log_dir, 'tensorboard'),
        report_to=["tensorboard"],
        seed=config.get('data', {}).get('seed', 42),
        # Ensure evaluation metrics are logged
        evaluation_strategy="steps",
        logging_strategy="steps",
    )
    
    # Setup callbacks
    callbacks = []
    
    # Early stopping
    early_stopping_config = config.get('training', {})
    patience = early_stopping_config.get('early_stopping_patience', 5)
    threshold = early_stopping_config.get('early_stopping_threshold', 0.001)
    callbacks.append(CustomEarlyStoppingCallback(patience=patience, threshold=threshold))
    
    # Initialize trainer
    trainer = Trainer(
        model=model.model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
        # Ensure callbacks receive evaluation logs
        preprocess_logits_for_metrics=None,
    )
    
    # Resume from checkpoint if specified
    if resume_from_checkpoint:
        logger.info(f"Resuming training from checkpoint: {resume_from_checkpoint}")
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    else:
        logger.info("Starting training from scratch")
        trainer.train()
    
    # Save final model
    output_dir = training_config.get('output_dir', './models/checkpoints')
    final_model_path = os.path.join(output_dir, 'final_model')
    model.save_model(final_model_path)
    
    # Final evaluation
    logger.info("Running final evaluation...")
    eval_results = trainer.evaluate()
    
    # Log final results
    logger.info("Final evaluation results:")
    for metric, value in eval_results.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    # Save evaluation results
    results_path = os.path.join(log_dir, 'final_results.json')
    with open(results_path, 'w') as f:
        json.dump(eval_results, f, indent=2)
    
    logger.info(f"Training completed successfully!")
    logger.info(f"Model saved to: {final_model_path}")
    logger.info(f"Results saved to: {results_path}")
    
    return trainer, eval_results


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Train ELECTRA-based jailbreak detection model")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
    parser.add_argument("--lora", action="store_true", help="Use LoRA fine-tuning")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Set debug level if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        trainer, results = train(args.config, args.lora, args.resume)
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()