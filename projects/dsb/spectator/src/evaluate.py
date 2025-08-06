"""Evaluation script for ETH volatility and direction prediction model."""

import json
import logging
from pathlib import Path
from typing import Any

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
  accuracy_score, precision_score, recall_score, f1_score,
  roc_auc_score, confusion_matrix, classification_report,
  mean_absolute_error, mean_squared_error, r2_score
)
from safetensors.torch import load_file
from tqdm import tqdm

from .config import (
  EXPERIMENTS_DIR,
  MODEL_CHECKPOINT_FILE,
  LOG_LEVEL,
  LOG_FORMAT,
)
from .model import ETHVolatilityModel, create_model
from .dataset import create_data_loaders, DataSplitter
from .data_loader import CoinDeskDataLoader
from .preprocessing import DataPreprocessor
from .embedder import NewsEmbedder


# Configure logging
logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=LOG_FORMAT)
logger = logging.getLogger(__name__)


class ModelEvaluator:
  """Handles model evaluation and metrics computation."""

  def __init__(self, model: ETHVolatilityModel, device: torch.device) -> None:
    """
    Initialize the evaluator.
    
    Args:
      model: The trained model to evaluate
      device: Device to run evaluation on
    """
    self.model = model.to(device)
    self.device = device

  def evaluate_model(self, data_loader: torch.utils.data.DataLoader) -> dict[str, Any]:
    """
    Evaluate the model on a dataset.
    
    Args:
      data_loader: DataLoader for the dataset to evaluate
      
    Returns:
      Dictionary with evaluation metrics and predictions
    """
    self.model.eval()
    
    all_direction_preds = []
    all_direction_targets = []
    all_volatility_preds = []
    all_volatility_targets = []
    
    with torch.no_grad():
      with tqdm(data_loader, desc="Evaluating") as pbar:
        for batch in pbar:
          # Move data to device
          price_seq = batch["price_seq"].to(self.device)
          news_emb = batch["news_emb"].to(self.device)
          direction_target = batch["direction"].to(self.device)
          volatility_target = batch["volatility"].to(self.device)
          
          # Forward pass
          predictions = self.model(price_seq, news_emb)
          
          # Collect predictions and targets
          all_direction_preds.extend(predictions["direction"].cpu().numpy())
          all_direction_targets.extend(direction_target.cpu().numpy())
          all_volatility_preds.extend(predictions["volatility"].cpu().numpy())
          all_volatility_targets.extend(volatility_target.cpu().numpy())
    
    # Convert to numpy arrays
    direction_preds = np.array(all_direction_preds).flatten()
    direction_targets = np.array(all_direction_targets).flatten()
    volatility_preds = np.array(all_volatility_preds).flatten()
    volatility_targets = np.array(all_volatility_targets).flatten()
    
    # Calculate metrics
    direction_metrics = self._calculate_direction_metrics(direction_preds, direction_targets)
    volatility_metrics = self._calculate_volatility_metrics(volatility_preds, volatility_targets)
    
    return {
      "direction_metrics": direction_metrics,
      "volatility_metrics": volatility_metrics,
      "predictions": {
        "direction_preds": direction_preds.tolist(),
        "direction_targets": direction_targets.tolist(),
        "volatility_preds": volatility_preds.tolist(),
        "volatility_targets": volatility_targets.tolist(),
      },
    }

  def _calculate_direction_metrics(self, preds: np.ndarray, targets: np.ndarray) -> dict[str, Any]:
    """Calculate direction prediction metrics."""
    # Convert probabilities to binary predictions
    binary_preds = (preds > 0.5).astype(int)
    
    # Basic metrics
    accuracy = accuracy_score(targets, binary_preds)
    precision = precision_score(targets, binary_preds, zero_division=0)
    recall = recall_score(targets, binary_preds, zero_division=0)
    f1 = f1_score(targets, binary_preds, zero_division=0)
    
    # AUC-ROC
    try:
      auc = roc_auc_score(targets, preds)
    except ValueError:
      auc = 0.5  # Handle case where all targets are the same class
    
    # Confusion matrix
    cm = confusion_matrix(targets, binary_preds)
    
    # Classification report
    class_report = classification_report(targets, binary_preds, output_dict=True, zero_division=0)
    
    return {
      "accuracy": accuracy,
      "precision": precision,
      "recall": recall,
      "f1_score": f1,
      "auc_roc": auc,
      "confusion_matrix": cm.tolist(),
      "classification_report": class_report,
    }

  def _calculate_volatility_metrics(self, preds: np.ndarray, targets: np.ndarray) -> dict[str, Any]:
    """Calculate volatility prediction metrics."""
    # Basic regression metrics
    mae = mean_absolute_error(targets, preds)
    mse = mean_squared_error(targets, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(targets, preds)
    
    # Mean Absolute Percentage Error
    mask = targets != 0
    if np.any(mask):
      mape = np.mean(np.abs((targets[mask] - preds[mask]) / targets[mask])) * 100
    else:
      mape = 0.0
    
    # Additional statistics
    mean_target = np.mean(targets)
    mean_pred = np.mean(preds)
    std_target = np.std(targets)
    std_pred = np.std(preds)
    
    # Correlation
    correlation = np.corrcoef(targets, preds)[0, 1] if len(targets) > 1 else 0.0
    
    return {
      "mae": mae,
      "mse": mse,
      "rmse": rmse,
      "r2_score": r2,
      "mape": mape,
      "mean_target": mean_target,
      "mean_prediction": mean_pred,
      "std_target": std_target,
      "std_prediction": std_pred,
      "correlation": correlation,
    }

  def create_evaluation_plots(self, results: dict[str, Any], save_dir: Path) -> None:
    """Create evaluation plots and save them."""
    save_dir.mkdir(parents=True, exist_ok=True)
    
    predictions = results["predictions"]
    direction_preds = np.array(predictions["direction_preds"])
    direction_targets = np.array(predictions["direction_targets"])
    volatility_preds = np.array(predictions["volatility_preds"])
    volatility_targets = np.array(predictions["volatility_targets"])
    
    # Set style
    plt.style.use('seaborn-v0_8')
    
    # 1. Direction confusion matrix
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    cm = results["direction_metrics"]["confusion_matrix"]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title('Direction Prediction Confusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_xticklabels(['Down', 'Up'])
    ax.set_yticklabels(['Down', 'Up'])
    plt.tight_layout()
    plt.savefig(save_dir / 'direction_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Direction probability distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Histogram of predictions by class
    up_probs = direction_preds[direction_targets == 1]
    down_probs = direction_preds[direction_targets == 0]
    
    ax1.hist(down_probs, bins=20, alpha=0.7, label='Down (0)', color='red', density=True)
    ax1.hist(up_probs, bins=20, alpha=0.7, label='Up (1)', color='green', density=True)
    ax1.set_xlabel('Predicted Probability')
    ax1.set_ylabel('Density')
    ax1.set_title('Direction Prediction Probability Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # ROC curve would go here if we had the data for it
    ax2.scatter(direction_targets, direction_preds, alpha=0.6)
    ax2.plot([0, 1], [0, 1], 'r--', alpha=0.8)
    ax2.set_xlabel('Actual Direction')
    ax2.set_ylabel('Predicted Probability')
    ax2.set_title('Direction: Actual vs Predicted')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'direction_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Volatility scatter plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Scatter plot
    ax1.scatter(volatility_targets, volatility_preds, alpha=0.6)
    min_val = min(volatility_targets.min(), volatility_preds.min())
    max_val = max(volatility_targets.max(), volatility_preds.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
    ax1.set_xlabel('Actual Volatility')
    ax1.set_ylabel('Predicted Volatility')
    ax1.set_title('Volatility: Actual vs Predicted')
    ax1.grid(True, alpha=0.3)
    
    # Residuals plot
    residuals = volatility_preds - volatility_targets
    ax2.scatter(volatility_targets, residuals, alpha=0.6)
    ax2.axhline(y=0, color='r', linestyle='--', alpha=0.8)
    ax2.set_xlabel('Actual Volatility')
    ax2.set_ylabel('Residuals (Predicted - Actual)')
    ax2.set_title('Volatility Prediction Residuals')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'volatility_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Error distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Volatility error histogram
    ax1.hist(residuals, bins=20, alpha=0.7, color='blue', density=True)
    ax1.set_xlabel('Prediction Error')
    ax1.set_ylabel('Density')
    ax1.set_title('Volatility Prediction Error Distribution')
    ax1.axvline(x=0, color='r', linestyle='--', alpha=0.8)
    ax1.grid(True, alpha=0.3)
    
    # Time series of errors (if we had time index)
    ax2.plot(range(len(residuals)), residuals, alpha=0.7)
    ax2.axhline(y=0, color='r', linestyle='--', alpha=0.8)
    ax2.set_xlabel('Sample Index')
    ax2.set_ylabel('Prediction Error')
    ax2.set_title('Volatility Prediction Errors Over Time')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'error_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Evaluation plots saved to {save_dir}")


def load_best_model(model_path: Path, price_input_size: int, device: torch.device) -> ETHVolatilityModel:
  """Load the best trained model."""
  # Create model
  model = create_model(price_input_size)
  
  # Load checkpoint
  checkpoint = load_file(str(model_path))
  model.load_state_dict(checkpoint["model_state_dict"])
  
  return model


def main() -> None:
  """Main evaluation pipeline."""
  # Set device
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  logger.info(f"Using device: {device}")
  
  try:
    # Load and preprocess data
    logger.info("Loading data...")
    loader = CoinDeskDataLoader()
    price_df, news_df = loader.fetch_all_data()
    
    logger.info("Preprocessing data...")
    preprocessor = DataPreprocessor()
    windows = preprocessor.process_data(price_df, news_df)
    
    # Generate embeddings
    logger.info("Generating embeddings...")
    embedder = NewsEmbedder()
    windows = embedder.process_windows(windows)
    
    # Split data
    logger.info("Splitting data...")
    splitter = DataSplitter()
    train_windows, val_windows, test_windows = splitter.time_based_split(windows)
    
    # Create data loaders
    logger.info("Creating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(
      train_windows, val_windows, test_windows
    )
    
    # Determine input size
    sample_batch = next(iter(train_loader))
    price_input_size = sample_batch["price_seq"].shape[-1]
    
    # Load best model
    model_path = EXPERIMENTS_DIR / f"best_{MODEL_CHECKPOINT_FILE}"
    if not model_path.exists():
      model_path = EXPERIMENTS_DIR / MODEL_CHECKPOINT_FILE
    
    if not model_path.exists():
      raise FileNotFoundError(f"No trained model found at {model_path}")
    
    logger.info(f"Loading model from {model_path}")
    model = load_best_model(model_path, price_input_size, device)
    
    # Create evaluator
    evaluator = ModelEvaluator(model, device)
    
    # Evaluate on all splits
    logger.info("Evaluating on training set...")
    train_results = evaluator.evaluate_model(train_loader)
    
    logger.info("Evaluating on validation set...")
    val_results = evaluator.evaluate_model(val_loader)
    
    logger.info("Evaluating on test set...")
    test_results = evaluator.evaluate_model(test_loader)
    
    # Combine results
    evaluation_results = {
      "train": train_results,
      "validation": val_results,
      "test": test_results,
    }
    
    # Print summary
    def print_metrics(split_name: str, results: dict[str, Any]) -> None:
      dir_metrics = results["direction_metrics"]
      vol_metrics = results["volatility_metrics"]
      
      logger.info(f"\n{split_name.upper()} SET RESULTS:")
      logger.info("Direction Prediction:")
      logger.info(f"  Accuracy: {dir_metrics['accuracy']:.4f}")
      logger.info(f"  Precision: {dir_metrics['precision']:.4f}")
      logger.info(f"  Recall: {dir_metrics['recall']:.4f}")
      logger.info(f"  F1-Score: {dir_metrics['f1_score']:.4f}")
      logger.info(f"  AUC-ROC: {dir_metrics['auc_roc']:.4f}")
      
      logger.info("Volatility Prediction:")
      logger.info(f"  MAE: {vol_metrics['mae']:.4f}")
      logger.info(f"  RMSE: {vol_metrics['rmse']:.4f}")
      logger.info(f"  R²: {vol_metrics['r2_score']:.4f}")
      logger.info(f"  MAPE: {vol_metrics['mape']:.2f}%")
      logger.info(f"  Correlation: {vol_metrics['correlation']:.4f}")
    
    print_metrics("Train", train_results)
    print_metrics("Validation", val_results)
    print_metrics("Test", test_results)
    
    # Save evaluation results
    results_file = EXPERIMENTS_DIR / "evaluation_results.json"
    with open(results_file, "w") as f:
      # Convert numpy types to Python types for JSON serialization
      def convert_types(obj: Any) -> Any:
        if hasattr(obj, 'item'):
          return obj.item()
        elif isinstance(obj, dict):
          return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
          return [convert_types(v) for v in obj]
        else:
          return obj
      
      json.dump(convert_types(evaluation_results), f, indent=2, default=str)
    
    logger.info(f"Evaluation results saved to {results_file}")
    
    # Create evaluation plots
    logger.info("Creating evaluation plots...")
    plots_dir = EXPERIMENTS_DIR / "plots"
    evaluator.create_evaluation_plots(test_results, plots_dir)
    
    logger.info("Evaluation pipeline completed successfully!")
    
  except Exception as e:
    logger.error(f"Evaluation failed with error: {e}")
    raise


if __name__ == "__main__":
  main()