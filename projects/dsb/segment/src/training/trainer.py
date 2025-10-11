"""
Training script for ELECTRA jailbreak detection model.
"""

import os
import torch
import yaml
from transformers import (
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    AutoTokenizer
)
from datasets import DatasetDict
from loguru import logger
from typing import Dict, Optional

from ..models.electra_classifier import ElectraJailbreakClassifier
from ..data.dataset_loader import JailbreakDatasetLoader
from ..data.preprocessor import TextPreprocessor
from ..data.data_collator import JailbreakDataCollator
from ..evaluation.metrics import JailbreakMetrics


class JailbreakTrainer:
    """Trainer for ELECTRA jailbreak detection model."""
    
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Initialize components
        self.dataset_loader = JailbreakDatasetLoader()
        self.preprocessor = TextPreprocessor(self.config["model"]["name"])
        self.metrics_calculator = JailbreakMetrics()
        
        # Initialize model and tokenizer
        self.model = None
        self.tokenizer = None
        self.trainer = None
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def prepare_data(self) -> DatasetDict:
        """Prepare training data."""
        logger.info("Loading and preparing data...")
        
        # Load datasets
        dataset = self.dataset_loader.load_all_datasets()
        
        # Get statistics
        stats = self.dataset_loader.get_dataset_statistics(dataset)
        logger.info(f"Dataset statistics: {stats}")
        
        # Create splits
        dataset_dict = self.dataset_loader.create_train_val_test_split(
            dataset,
            train_split=self.config["data"]["train_split"],
            val_split=self.config["data"]["val_split"],
            test_split=self.config["data"]["test_split"],
            seed=self.config["data"]["seed"]
        )
        
        # Preprocess and tokenize
        for split in dataset_dict:
            dataset_dict[split] = self.preprocessor.preprocess_dataset(
                dataset_dict[split], 
                text_column=self.config["data"]["column_mapping"]["text"]
            )
            dataset_dict[split] = self.preprocessor.tokenize_dataset(
                dataset_dict[split],
                text_column=self.config["data"]["column_mapping"]["text"],
                max_length=self.config["model"]["max_length"]
            )
        
        return dataset_dict
    
    def prepare_model(self, use_lora: bool = False):
        """Prepare model for training."""
        logger.info("Initializing model...")
        
        self.model = ElectraJailbreakClassifier(
            model_name=self.config["model"]["name"],
            num_labels=self.config["model"]["num_labels"],
            dropout=self.config["model"]["dropout"],
            use_lora=use_lora,
            lora_config=self.config.get("lora") if use_lora else None
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.config["model"]["name"])
        
        # Move to device
        self.model.to(self.device)
        
        # Print model info
        trainable_params, total_params = self.model.get_trainable_parameters()
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable ratio: {trainable_params/total_params:.4f}")
    
    def setup_trainer(self, dataset_dict: DatasetDict, output_dir: str):
        """Setup Hugging Face Trainer."""
        # Data collator
        data_collator = JailbreakDataCollator(self.tokenizer)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            evaluation_strategy=self.config["training"]["evaluation_strategy"],
            save_strategy=self.config["training"]["save_strategy"],
            learning_rate=self.config["training"]["learning_rate"],
            per_device_train_batch_size=self.config["training"]["batch_size"],
            per_device_eval_batch_size=self.config["training"]["batch_size"],
            gradient_accumulation_steps=self.config["training"]["gradient_accumulation_steps"],
            num_train_epochs=self.config["training"]["num_epochs"],
            weight_decay=self.config["training"]["weight_decay"],
            warmup_ratio=self.config["training"]["warmup_ratio"],
            fp16=self.config["training"]["fp16"] and torch.cuda.is_available(),
            dataloader_num_workers=self.config["training"]["dataloader_num_workers"],
            load_best_model_at_end=self.config["training"]["load_best_model_at_end"],
            metric_for_best_model=self.config["training"]["metric_for_best_model"],
            greater_is_better=self.config["training"]["greater_is_better"],
            logging_dir=f"{output_dir}/logs",
            logging_steps=100,
            save_total_limit=3,
            report_to="wandb" if self.config["logging"]["use_wandb"] else None,
        )
        
        # Initialize trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset_dict["train"],
            eval_dataset=dataset_dict["validation"],
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.metrics_calculator.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=self.config["training"]["early_stopping_patience"])]
        )
    
    def train(self, dataset_dict: DatasetDict, output_dir: str):
        """Train the model."""
        logger.info("Starting training...")
        
        self.setup_trainer(dataset_dict, output_dir)
        
        # Train
        train_result = self.trainer.train()
        
        # Save model
        self.model.save_model(os.path.join(output_dir, "final_model"))
        
        # Evaluate
        eval_result = self.trainer.evaluate()
        
        logger.info("Training completed!")
        logger.info(f"Training loss: {train_result.training_loss}")
        logger.info(f"Evaluation results: {eval_result}")
        
        return train_result, eval_result
    
    def evaluate(self, dataset_dict: DatasetDict):
        """Evaluate model on test set."""
        if not self.trainer:
            raise ValueError("Trainer not initialized. Call train() first.")
        
        logger.info("Evaluating on test set...")
        test_result = self.trainer.evaluate(dataset_dict["test"])
        
        logger.info(f"Test results: {test_result}")
        return test_result
    
    def run_full_training(self, output_dir: str = "./checkpoints", use_lora: bool = False):
        """Run complete training pipeline."""
        # Prepare data
        dataset_dict = self.prepare_data()
        
        # Prepare model
        self.prepare_model(use_lora=use_lora)
        
        # Train
        train_result, eval_result = self.train(dataset_dict, output_dir)
        
        # Final evaluation
        test_result = self.evaluate(dataset_dict)
        
        return {
            "train_result": train_result,
            "eval_result": eval_result,
            "test_result": test_result
        }