"""
Trainer for learnable product quantization models.

This module provides a comprehensive training pipeline with support for
hyperparameter tuning, experiment tracking, model evaluation and early stopping.
"""

import os
import time
import logging
from typing import Dict, List, Optional, Tuple, Any, Callable
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm

try:
    import optuna
    from optuna.integration import PyTorchLightningPruningCallback

    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from .model import LearnablePQ
from .losses import CombinedLoss


class VectorDataset(Dataset):
    """Simple dataset for vector data."""

    def __init__(self, vectors: np.ndarray):
        self.vectors = torch.FloatTensor(vectors)

    def __len__(self):
        return len(self.vectors)

    def __getitem__(self, idx):
        return self.vectors[idx]


class Trainer:
    """
    Trainer for LearnablePQ models with support for hyperparameter tuning.

    This class handles the complete training pipeline including data loading,
    model training, validation, early stopping and experiment tracking.
    """

    def __init__(
        self,
        model: LearnablePQ,
        device: Optional[str] = None,
        log_dir: str = "logs",
        experiment_name: str = "quantumdb",
        use_wandb: bool = False,
        wandb_project: str = "quantumdb",
    ):
        self.model = model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.experiment_name = experiment_name

        # Setup logging
        self.setup_logging()

        # Experiment tracking
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        if self.use_wandb:
            wandb.init(
                project=wandb_project,
                name=experiment_name,
                config=self.get_model_config(),
            )

        # Training state
        self.current_epoch = 0
        self.best_loss = float("inf")
        self.training_history = []

    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(self.log_dir / f"{self.experiment_name}.log"),
                logging.StreamHandler(),
            ],
        )
        self.logger = logging.getLogger(self.experiment_name)

    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration for logging."""
        return {
            "input_dim": self.model.input_dim,
            "target_dim": self.model.target_dim,
            "n_subvectors": self.model.n_subvectors,
            "codebook_size": self.model.codebook_size,
            "compression_ratio": self.model.get_compression_ratio(),
            "model_size": self.model.get_model_size(),
        }

    def create_data_loaders(
        self,
        train_data: np.ndarray,
        val_data: Optional[np.ndarray] = None,
        batch_size: int = 256,
        num_workers: int = 4,
        val_split: float = 0.1,
    ) -> Tuple[DataLoader, Optional[DataLoader]]:
        """
        Create training and validation data loaders.

        Args:
            train_data: Training vectors
            val_data: Validation vectors (optional)
            batch_size: Batch size for training
            num_workers: Number of worker processes
            val_split: Fraction of training data to use for validation if val_data is None

        Returns:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
        """
        # Create validation split if not provided
        if val_data is None:
            n_val = int(len(train_data) * val_split)
            if n_val > 0:
                indices = np.random.permutation(len(train_data))
                val_indices, train_indices = indices[:n_val], indices[n_val:]

                val_data = train_data[val_indices]
                train_data = train_data[train_indices]
            else:
                val_data = None

        # Create datasets
        train_dataset = VectorDataset(train_data)
        val_dataset = VectorDataset(val_data) if val_data is not None else None

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )

        val_loader = None
        if val_dataset is not None:
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )

        self.logger.info(f"Training samples: {len(train_dataset)}")
        if val_loader:
            self.logger.info(f"Validation samples: {len(val_dataset)}")

        return train_loader, val_loader

    def train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: optim.Optimizer,
        loss_fn: nn.Module,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
    ) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        loss_components = {}

        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch + 1}")

        for batch_idx, vectors in enumerate(pbar):
            vectors = vectors.to(self.device)

            # Forward pass
            optimizer.zero_grad()

            encoded = self.model.encode(vectors)
            quantized, codes = self.model.quantize(encoded)

            # Calculate loss
            loss, components = loss_fn(
                original=vectors,
                encoded=encoded,
                quantized=quantized,
                codes=codes,
            )

            # Backward pass
            loss.backward()
            optimizer.step()

            # Update metrics
            total_loss += loss.item()
            for key, value in components.items():
                if key not in loss_components:
                    loss_components[key] = 0.0
                loss_components[key] += value.item()

            # Update progress bar
            pbar.set_postfix({"loss": loss.item()})

            # Log to wandb
            if self.use_wandb and batch_idx % 100 == 0:
                wandb.log(
                    {
                        "batch_loss": loss.item(),
                        "batch": self.current_epoch * len(train_loader) + batch_idx,
                    }
                )

        # Average losses
        avg_loss = total_loss / len(train_loader)
        avg_components = {k: v / len(train_loader) for k, v in loss_components.items()}

        # Update learning rate (once per epoch)
        if scheduler:
            scheduler.step()

        return {"total_loss": avg_loss, **avg_components}

    def validate(
        self,
        val_loader: DataLoader,
        loss_fn: nn.Module,
    ) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        loss_components = {}

        with torch.no_grad():
            for vectors in tqdm(val_loader, desc="Validation"):
                vectors = vectors.to(self.device)

                # Forward pass
                encoded = self.model.encode(vectors)
                quantized, codes = self.model.quantize(encoded)

                # Calculate loss
                loss, components = loss_fn(
                    original=vectors,
                    encoded=encoded,
                    quantized=quantized,
                    codes=codes,
                )

                # Update metrics
                total_loss += loss.item()
                for key, value in components.items():
                    if key not in loss_components:
                        loss_components[key] = 0.0
                    loss_components[key] += value.item()

        # Average losses
        avg_loss = total_loss / len(val_loader)
        avg_components = {k: v / len(val_loader) for k, v in loss_components.items()}

        return {"total_loss": avg_loss, **avg_components}

    def fit(
        self,
        train_data: np.ndarray,
        val_data: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        loss_weights: Optional[Dict[str, float]] = None,
        save_best: bool = True,
        save_dir: str = "models",
        early_stopping: bool = True,
        patience: int = 10,
        min_delta: float = 0.0,
    ) -> Dict[str, List[float]]:
        """
        Train the model.

        Args:
            train_data: Training vectors
            val_data: Validation vectors
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            weight_decay: Weight decay
            loss_weights: Weights for different loss components
            save_best: Whether to save the best model
            save_dir: Directory to save models
            early_stopping: Whether to apply early stopping
            patience: Number of epochs with no improvement before stopping
            min_delta: Minimum change in the monitored metric to qualify as improvement

        Returns:
            history: Training history
        """
        # Create data loaders
        train_loader, val_loader = self.create_data_loaders(
            train_data, val_data, batch_size
        )

        # Setup optimizer and loss function
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        loss_weights = loss_weights or {
            "quantization_weight": 1.0,
            "reconstruction_weight": 1.0,
            "triplet_weight": 0.0,
            "diversity_weight": 0.1,
        }

        loss_fn = CombinedLoss(
            **loss_weights,
            n_subvectors=self.model.n_subvectors,
            codebook_size=self.model.codebook_size,
        )

        # Training loop
        history = {"train_loss": [], "val_loss": []}

        # Early stopping state
        epochs_no_improve = 0
        # Initialize best_loss such that is bigger than any real loss
        # If using internal state from previous calls, we still want to start fresh here
        best_monitored = float("inf")

        for epoch in range(epochs):
            self.current_epoch = epoch

            # Train
            train_metrics = self.train_epoch(
                train_loader, optimizer, loss_fn, scheduler
            )

            # Validate
            val_metrics = {}
            if val_loader:
                val_metrics = self.validate(val_loader, loss_fn)

            # Determine monitored metric (prefer val_loss if available)
            if val_metrics:
                monitored_value = val_metrics["total_loss"]
                monitor_name = "val_loss"
            else:
                monitored_value = train_metrics["total_loss"]
                monitor_name = "train_loss"

            # Update history
            history["train_loss"].append(train_metrics["total_loss"])
            if val_metrics:
                history["val_loss"].append(val_metrics["total_loss"])

            # Check improvement
            improved = (best_monitored - monitored_value) > min_delta
            if improved:
                best_monitored = monitored_value
                epochs_no_improve = 0
                # Save best model immediately
                if save_best:
                    self.best_loss = best_monitored
                    self.save_model(save_dir, f"{self.experiment_name}_best")
            else:
                epochs_no_improve += 1

            # Log metrics
            log_msg = (
                f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_metrics['total_loss']:.4f}"
            )
            if val_metrics:
                log_msg += f" - Val Loss: {val_metrics['total_loss']:.4f}"
            log_msg += f" - Monitored ({monitor_name}): {monitored_value:.4f}"
            log_msg += f" - Best: {best_monitored:.4f}"
            log_msg += f" - Epochs_no_improve: {epochs_no_improve}/{patience}"
            self.logger.info(log_msg)

            # Log to wandb
            if self.use_wandb:
                wandb_log = {
                    "epoch": epoch + 1,
                    "train_loss": train_metrics["total_loss"],
                    **{
                        f"train_{k}": v
                        for k, v in train_metrics.items()
                        if k != "total_loss"
                    },
                    **{f"val_{k}": v for k, v in val_metrics.items()},
                }
                wandb.log(wandb_log)

            # Early stopping check
            if early_stopping and epochs_no_improve >= patience:
                self.logger.info(
                    f"Early stopping triggered (no improvement in {patience} epochs)."
                )
                break

        # Save final model
        if save_best:
            self.save_model(save_dir, f"{self.experiment_name}_final")

        return history

    def save_model(self, save_dir: str, model_name: str):
        """Save model using SafeTensors."""
        from safetensors.torch import save_file

        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True, parents=True)

        # Save model state
        state_dict = self.model.state_dict()
        save_file(state_dict, save_path / f"{model_name}.safetensors")

        # Save metadata
        metadata = {
            "model_config": self.get_model_config(),
            "training_config": {
                "best_loss": self.best_loss,
                "current_epoch": self.current_epoch,
            },
        }

        import json

        with open(save_path / f"{model_name}_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        self.logger.info(f"Model saved to {save_path / f'{model_name}.safetensors'}")

    def load_model(self, model_path: str):
        """Load model from SafeTensors."""
        from safetensors.torch import load_file

        state_dict = load_file(model_path)
        self.model.load_state_dict(state_dict)

        self.logger.info(f"Model loaded from {model_path}")

    def hyperparameter_tune(
        self,
        train_data: np.ndarray,
        val_data: Optional[np.ndarray] = None,
        n_trials: int = 50,
        timeout: Optional[int] = None,
        study_name: str = "quantumdb_optimization",
        early_stopping: bool = True,
        patience: int = 5,
        min_delta: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning using Optuna.

        Args:
            train_data: Training vectors
            val_data: Validation vectors
            n_trials: Number of optimization trials
            timeout: Timeout in seconds
            study_name: Name of the Optuna study
            early_stopping: Whether to use early stopping during trials
            patience: Early stopping patience for trials
            min_delta: Minimal improvement considered as progress

        Returns:
            best_params: Best hyperparameters found
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError(
                "Optuna is not installed. Install with: pip install optuna"
            )

        def objective(trial):
            # Sample hyperparameters
            config = {
                "learning_rate": trial.suggest_float(
                    "learning_rate", 1e-5, 1e-2, log=True
                ),
                "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512]),
                "weight_decay": trial.suggest_float(
                    "weight_decay", 1e-6, 1e-3, log=True
                ),
                "n_subvectors": trial.suggest_int("n_subvectors", 8, 32),
                "codebook_size": trial.suggest_categorical(
                    "codebook_size", [128, 256, 512]
                ),
                "target_dim": trial.suggest_categorical("target_dim", [128, 256, 384]),
            }

            # Create model with sampled hyperparameters
            model = LearnablePQ(
                input_dim=self.model.input_dim,
                target_dim=config["target_dim"],
                n_subvectors=config["n_subvectors"],
                codebook_size=config["codebook_size"],
            )

            # Create trainer
            trainer = Trainer(
                model,
                device=self.device,
                experiment_name=f"{self.experiment_name}_trial_{trial.number}",
                use_wandb=False,  # Disable wandb for tuning
            )

            # Train for fewer epochs during tuning
            history = trainer.fit(
                train_data=train_data,
                val_data=val_data,
                epochs=20,  # Fewer epochs for faster tuning
                batch_size=config["batch_size"],
                learning_rate=config["learning_rate"],
                weight_decay=config["weight_decay"],
                save_best=False,
                early_stopping=early_stopping,
                patience=patience,
                min_delta=min_delta,
            )

            # Return best validation loss
            return (
                min(history["val_loss"])
                if history["val_loss"]
                else history["train_loss"][-1]
            )

        # Create study
        study = optuna.create_study(
            study_name=study_name,
            direction="minimize",
            sampler=optuna.samplers.TPESampler(),
        )

        # Optimize
        study.optimize(objective, n_trials=n_trials, timeout=timeout)

        # Log results
        self.logger.info(f"Best trial: {study.best_trial.number}")
        self.logger.info(f"Best value: {study.best_trial.value}")
        self.logger.info(f"Best params: {study.best_trial.params}")

        return study.best_trial.params

