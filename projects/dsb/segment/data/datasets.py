"""
Dataset loading utilities for the Segment project.
"""

import os
import logging
from typing import Dict, Any, Tuple

import numpy as np
from datasets import load_from_disk, Dataset
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)

class JailbreakDataset:
    """
    Handles loading and preprocessing of the jailbreak detection dataset.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the dataset manager.

        Args:
            config: Configuration dictionary.
        """
        self.config = config
        self.dataset_path = os.path.join(
            config.get("data", {}).get("processed_path", "data/processed"),
            "combined_dataset"
        )
        self.dataset = self._load_dataset()

    def _load_dataset(self) -> Dataset:
        """
        Loads the dataset from disk.
        """
        if not os.path.exists(self.dataset_path):
            logger.error(f"Dataset not found at: {self.dataset_path}")
            logger.error("Please run 'python scripts/download_datasets.py' first.")
            raise FileNotFoundError(f"Dataset not found at: {self.dataset_path}")
        
        logger.info(f"Loading dataset from: {self.dataset_path}")
        return load_from_disk(self.dataset_path)

    def _preprocess_function(self, examples: Dict[str, Any], tokenizer: PreTrainedTokenizer) -> Dict[str, Any]:
        """
        Tokenizes the input text.
        """
        return tokenizer(
            examples["prompt"],
            truncation=True,
            padding="max_length",
            max_length=self.config.get("data", {}).get("max_length", 512),
        )

    def prepare_datasets(self, tokenizer: PreTrainedTokenizer) -> Tuple[Dataset, Dataset]:
        """
        Prepares the train and evaluation datasets.

        Args:
            tokenizer: The tokenizer to use for preprocessing.

        Returns:
            A tuple containing the tokenized train and evaluation datasets.
        """
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Please check the dataset path.")

        # Map labels to integers
        label2id = {"benign": 0, "jailbreak": 1}
        self.dataset = self.dataset.map(lambda x: {"label": label2id.get(x["type"], -1)})

        # Tokenize the dataset
        tokenized_dataset = self.dataset.map(
            lambda x: self._preprocess_function(x, tokenizer),
            batched=True,
        )

        # Split into train and eval
        train_dataset = tokenized_dataset["train"]
        eval_dataset = tokenized_dataset["validation"]

        logger.info(f"Train dataset size: {len(train_dataset)}")
        logger.info(f"Evaluation dataset size: {len(eval_dataset)}")

        return train_dataset, eval_dataset

    def get_dataset_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the dataset.
        
        Returns:
            Dictionary containing dataset statistics.
        """
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Please check the dataset path.")
        
        stats = {}
        
        for split_name, split_data in self.dataset.items():
            if 'type' in split_data.column_names:
                labels = split_data['type']
                unique_labels, counts = np.unique(labels, return_counts=True)
                label_counts = dict(zip(unique_labels, counts))
                
                stats[split_name] = {
                    'total_examples': len(split_data),
                    'label_distribution': label_counts,
                    'columns': split_data.column_names
                }
            else:
                stats[split_name] = {
                    'total_examples': len(split_data),
                    'columns': split_data.column_names
                }
        
        return stats
