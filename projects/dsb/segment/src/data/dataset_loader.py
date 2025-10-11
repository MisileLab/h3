"""
Dataset loader for jailbreak detection datasets.
Supports multiple public datasets from Hugging Face and other sources.
"""

import os
import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset
from typing import Dict, List, Optional, Tuple
from loguru import logger


class JailbreakDatasetLoader:
    """Load and combine multiple jailbreak detection datasets."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir
        self.datasets = {}
        
    def load_jailbreak_classification(self) -> Dataset:
        """Load jackhhao/jailbreak-classification dataset."""
        logger.info("Loading jackhhao/jailbreak-classification dataset...")
        dataset = load_dataset("jackhhao/jailbreak-classification", cache_dir=self.cache_dir)
        return dataset["train"]
    
    def load_jailbreak_reasoning(self) -> Dataset:
        """Load dvilasuero/jailbreak-classification-reasoning dataset."""
        logger.info("Loading dvilasuero/jailbreak-classification-reasoning dataset...")
        dataset = load_dataset("dvilasuero/jailbreak-classification-reasoning", cache_dir=self.cache_dir)
        return dataset["train"]
    
    def load_jigsaw_toxic(self) -> Dataset:
        """Load Jigsaw Toxic Comment dataset for auxiliary training."""
        logger.info("Loading Jigsaw Toxic Comment dataset...")
        dataset = load_dataset("jigsaw_toxicity_pred", cache_dir=self.cache_dir)
        return dataset["train"]
    
    def load_anthropic_hh(self) -> Dataset:
        """Load Anthropic Helpful-Harmless dataset."""
        logger.info("Loading Anthropic HH dataset...")
        dataset = load_dataset("Anthropic/hh-rlhf", cache_dir=self.cache_dir)
        return dataset["train"]
    
    def load_redteam_prompts(self) -> Dataset:
        """Load redteam-llm-prompts dataset."""
        logger.info("Loading redteam-llm-prompts dataset...")
        dataset = load_dataset("redteam-llm-prompts", cache_dir=self.cache_dir)
        return dataset["train"]
    
    def load_do_not_answer(self) -> Dataset:
        """Load Do-Not-Answer dataset."""
        logger.info("Loading Do-Not-Answer dataset...")
        dataset = load_dataset("LibrAI/do-not-answer", cache_dir=self.cache_dir)
        return dataset["train"]
    
    def standardize_dataset(self, dataset: Dataset, dataset_name: str) -> Dataset:
        """Standardize dataset format to have 'text' and 'label' columns."""
        logger.info(f"Standardizing {dataset_name} dataset...")
        
        def standardize_example(example):
            if dataset_name == "jailbreak_classification":
                return {
                    "text": example["prompt"],
                    "label": 1 if example["type"] == "jailbreak" else 0
                }
            elif dataset_name == "jailbreak_reasoning":
                return {
                    "text": example["prompt"],
                    "label": 1 if example["is_jailbreak"] else 0
                }
            elif dataset_name == "jigsaw_toxic":
                return {
                    "text": example["comment_text"],
                    "label": 1 if example["toxic"] == 1 else 0
                }
            elif dataset_name == "anthropic_hh":
                return {
                    "text": example["chosen"],
                    "label": 0  # helpful responses are safe
                }
            elif dataset_name == "redteam_prompts":
                return {
                    "text": example["prompt"],
                    "label": 1 if example["is_harmful"] else 0
                }
            elif dataset_name == "do_not_answer":
                return {
                    "text": example["question"],
                    "label": 1 if example["is_harmful"] else 0
                }
            else:
                return example
        
        return dataset.map(standardize_example, remove_columns=dataset.column_names)
    
    def load_all_datasets(self) -> Dataset:
        """Load and combine all available datasets."""
        datasets = []
        
        try:
            # Load jailbreak-specific datasets
            jailbreak_ds = self.load_jailbreak_classification()
            datasets.append(self.standardize_dataset(jailbreak_ds, "jailbreak_classification"))
        except Exception as e:
            logger.warning(f"Failed to load jailbreak_classification: {e}")
        
        try:
            reasoning_ds = self.load_jailbreak_reasoning()
            datasets.append(self.standardize_dataset(reasoning_ds, "jailbreak_reasoning"))
        except Exception as e:
            logger.warning(f"Failed to load jailbreak_reasoning: {e}")
        
        try:
            redteam_ds = self.load_redteam_prompts()
            datasets.append(self.standardize_dataset(redteam_ds, "redteam_prompts"))
        except Exception as e:
            logger.warning(f"Failed to load redteam_prompts: {e}")
        
        try:
            dna_ds = self.load_do_not_answer()
            datasets.append(self.standardize_dataset(dna_ds, "do_not_answer"))
        except Exception as e:
            logger.warning(f"Failed to load do_not_answer: {e}")
        
        # Load auxiliary datasets
        try:
            jigsaw_ds = self.load_jigsaw_toxic()
            # Sample a subset to avoid imbalance
            jigsaw_sampled = jigsaw_ds.shuffle(seed=42).select(range(min(10000, len(jigsaw_ds))))
            datasets.append(self.standardize_dataset(jigsaw_sampled, "jigsaw_toxic"))
        except Exception as e:
            logger.warning(f"Failed to load jigsaw_toxic: {e}")
        
        if not datasets:
            raise ValueError("No datasets could be loaded successfully")
        
        # Combine all datasets
        combined_dataset = Dataset.from_dict({
            "text": [],
            "label": []
        })
        
        for ds in datasets:
            combined_dataset = Dataset.from_dict({
                "text": combined_dataset["text"] + ds["text"],
                "label": combined_dataset["label"] + ds["label"]
            })
        
        logger.info(f"Combined dataset size: {len(combined_dataset)}")
        return combined_dataset
    
    def create_train_val_test_split(
        self, 
        dataset: Dataset, 
        train_split: float = 0.8, 
        val_split: float = 0.1, 
        test_split: float = 0.1,
        seed: int = 42
    ) -> DatasetDict:
        """Create train/validation/test splits."""
        assert abs(train_split + val_split + test_split - 1.0) < 1e-6, "Splits must sum to 1.0"
        
        # Shuffle dataset
        dataset = dataset.shuffle(seed=seed)
        
        # Calculate split sizes
        total_size = len(dataset)
        train_size = int(total_size * train_split)
        val_size = int(total_size * val_split)
        
        # Split dataset
        train_dataset = dataset.select(range(train_size))
        val_dataset = dataset.select(range(train_size, train_size + val_size))
        test_dataset = dataset.select(range(train_size + val_size, total_size))
        
        return DatasetDict({
            "train": train_dataset,
            "validation": val_dataset,
            "test": test_dataset
        })
    
    def get_dataset_statistics(self, dataset: Dataset) -> Dict:
        """Get dataset statistics."""
        labels = dataset["label"]
        total = len(labels)
        safe_count = labels.count(0)
        unsafe_count = labels.count(1)
        
        return {
            "total_samples": total,
            "safe_samples": safe_count,
            "unsafe_samples": unsafe_count,
            "safe_ratio": safe_count / total,
            "unsafe_ratio": unsafe_count / total
        }