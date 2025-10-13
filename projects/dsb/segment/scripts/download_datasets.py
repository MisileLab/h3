#!/usr/bin/env python3
"""
Dataset Download Script for Segment Project

This script downloads and prepares the datasets required for jailbreak detection training.
It handles the main jailbreak classification dataset and any additional datasets.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

try:
    from datasets import load_dataset, DatasetDict, Dataset, concatenate_datasets
    from datasets.arrow_dataset import Dataset as ArrowDataset
    from datasets.dataset_dict import DatasetDict as ArrowDatasetDict
    import pandas as pd
    import numpy as np
except ImportError as e:
    print(f"Error: Required packages not installed. Please run: uv pip install -r requirements.txt")
    print(f"Missing package: {e}")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def download_jailbreak_dataset(save_path: str = "data/raw") -> DatasetDict:
    """
    Download the main jailbreak classification dataset
    
    Args:
        save_path: Path to save the dataset
        
    Returns:
        Downloaded dataset
    """
    logger.info("Downloading jailbreak classification dataset...")
    
    try:
        # Load the dataset from Hugging Face
        dataset = load_dataset("jackhhao/jailbreak-classification")
        
        # Create save directory
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save dataset to disk
        dataset.save_to_disk(str(save_dir / "jailbreak_classification"))
        
        logger.info(f"Dataset downloaded and saved to: {save_dir / 'jailbreak_classification'}")
        
        # Print dataset info
        for split_name, split_data in dataset.items():
            logger.info(f"{split_name}: {len(split_data)} examples")
            
        return dataset
        
    except Exception as e:
        logger.error(f"Failed to download jailbreak dataset: {e}")
        raise


def download_jailbreak_reasoning_dataset(save_path: str = "data/raw") -> DatasetDict:
    """
    Download the jailbreak classification reasoning dataset
    
    Args:
        save_path: Path to save the dataset
        
    Returns:
        Downloaded dataset
    """
    logger.info("Downloading jailbreak classification reasoning dataset...")
    
    try:
        # Load the reasoning dataset from Hugging Face
        dataset = load_dataset("dvilasuero/jailbreak-classification-reasoning")
        
        # Create save directory
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save dataset to disk
        dataset.save_to_disk(str(save_dir / "jailbreak_reasoning"))
        
        logger.info(f"Reasoning dataset downloaded and saved to: {save_dir / 'jailbreak_reasoning'}")
        
        # Print dataset info
        for split_name, split_data in dataset.items():
            logger.info(f"{split_name}: {len(split_data)} examples")
            
        return dataset
        
    except Exception as e:
        logger.error(f"Failed to download jailbreak reasoning dataset: {e}")
        raise


def download_wildjailbreak_dataset(save_path: str = "data/raw"):
    """
    Download the WildJailbreak dataset and combine train/eval splits
    
    Args:
        save_path: Path to save the dataset
        
    Returns:
        Downloaded and processed dataset with train/validation/test splits
    """
    logger.info("Downloading WildJailbreak dataset...")
    
    try:
        # Create save directory
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Load the WildJailbreak training and evaluation sets using the user's code
        logger.info("Loading WildJailbreak training set...")
        train_data = load_dataset("allenai/wildjailbreak", "train", delimiter="\t", keep_default_na=False)
        
        logger.info("Loading WildJailbreak evaluation set...")
        eval_data = load_dataset("allenai/wildjailbreak", "eval", delimiter="\t", keep_default_na=False)
        
        # Convert to pandas for easier processing
        # DatasetDict doesn't have to_pandas(), need to access the train split first
        train_df = train_data['train'].to_pandas()
        eval_df = eval_data['train'].to_pandas()
        
        # Combine both datasets
        combined_df = pd.concat([train_df, eval_df], ignore_index=True)
        logger.info(f"Combined dataset has {len(combined_df)} examples")
        
        # Process to unified format
        processed_data = []
        for _, row in combined_df.iterrows():
            # Extract prompt text - try different column names
            prompt_text = row.get('prompt', row.get('text', ''))
            
            # Determine label type
            label_type = row.get('type', row.get('label', ''))
            
            # Map to our binary classification
            if pd.isna(label_type):
                unified_type = 'benign'  # Default for missing labels
            elif str(label_type).lower() in ['harmful', 'jailbreak', 'adversarial_harmful', 'vanilla_harmful', '1', '1.0']:
                unified_type = 'jailbreak'
            elif str(label_type).lower() in ['benign', 'safe', 'adversarial_benign', 'vanilla_benign', '0', '0.0']:
                unified_type = 'benign'
            else:
                unified_type = 'benign'
                logger.warning(f"Unknown label type '{label_type}', defaulting to 'benign'")
            
            processed_data.append({
                'prompt': str(prompt_text),
                'type': unified_type
            })
        
        # Create Dataset from processed data
        combined_dataset = Dataset.from_list(processed_data)
        
        # Shuffle the combined dataset
        combined_dataset = combined_dataset.shuffle(seed=42)
        
        # Create train/validation/test splits (80/10/10)
        total_size = len(combined_dataset)
        train_size = int(0.8 * total_size)
        val_size = int(0.1 * total_size)
        
        train_split = combined_dataset.select(range(train_size))
        val_split = combined_dataset.select(range(train_size, train_size + val_size))
        test_split = combined_dataset.select(range(train_size + val_size, total_size))
        
        # Create DatasetDict
        wildjailbreak_dataset = DatasetDict({
            'train': train_split,
            'validation': val_split,
            'test': test_split
        })
        
        # Save to disk
        wildjailbreak_dataset.save_to_disk(str(save_dir / "wildjailbreak"))
        
        logger.info(f"WildJailbreak dataset downloaded and saved to: {save_dir / 'wildjailbreak'}")
        
        # Print dataset info and label distribution
        for split_name, split_data in wildjailbreak_dataset.items():
            logger.info(f"{split_name}: {len(split_data)} examples")
            
            # Count labels
            if 'type' in split_data.column_names:
                labels = split_data['type']
                unique_labels, counts = np.unique(labels, return_counts=True)
                label_counts = dict(zip(unique_labels, counts))
                logger.info(f"  Label distribution: {label_counts}")
        
        return wildjailbreak_dataset
        
    except Exception as e:
        logger.error(f"Failed to download WildJailbreak dataset: {e}")
        raise


def download_additional_datasets(save_path: str = "data/raw") -> Dict[str, Any]:
    """
    Download additional datasets for enhanced training
    
    Args:
        save_path: Path to save the datasets
        
    Returns:
        Dictionary of additional datasets
    """
    logger.info("Downloading additional datasets...")
    
    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    additional_datasets = {}
    
    # Dataset 1: Toxicity dataset (for negative examples)
    try:
        logger.info("Downloading toxicity dataset...")
        toxicity_dataset = load_dataset("civil_comments/toxicity", split="train")
        
        # Sample a subset for faster training
        toxicity_subset = toxicity_dataset.shuffle(seed=42).select(range(1000))
        
        # Save to disk
        toxicity_subset.save_to_disk(str(save_dir / "toxicity"))
        additional_datasets['toxicity'] = toxicity_subset
        
        logger.info(f"Toxicity dataset downloaded: {len(toxicity_subset)} examples")
        
    except Exception as e:
        logger.warning(f"Could not download toxicity dataset: {e}")
    
    # Dataset 2: Synthetic safe prompts
    try:
        logger.info("Creating synthetic safe prompts...")
        
        # Create synthetic safe prompts
        safe_prompts = [
            "Tell me about the history of artificial intelligence.",
            "Explain the concept of machine learning.",
            "What are the benefits of renewable energy?",
            "Describe the process of photosynthesis.",
            "How does the internet work?",
            "What are the main principles of democracy?",
            "Explain the theory of evolution.",
            "Describe the solar system and its planets.",
            "What is climate change and its effects?",
            "How do vaccines work?",
            "Explain the basics of quantum physics.",
            "What are the major world religions?",
            "Describe the human digestive system.",
            "How do economic markets function?",
            "What is the history of space exploration?",
            "Explain the concept of blockchain technology.",
            "Describe different types of ecosystems.",
            "What are the principles of good writing?",
            "How does the human brain process language?",
            "Explain the importance of biodiversity."
        ]
        
        # Create variations
        synthetic_safe = []
        for prompt in safe_prompts:
            # Add variations
            synthetic_safe.append(prompt)
            synthetic_safe.append(f"Can you {prompt.lower()}")
            synthetic_safe.append(f"I would like to know {prompt.lower()}")
            synthetic_safe.append(f"Please explain {prompt.lower()}")
        
        # Create dataset
        synthetic_data = {
            'prompt': synthetic_safe,
            'type': ['benign'] * len(synthetic_safe)
        }
        
        synthetic_dataset = Dataset.from_dict(synthetic_data)
        
        # Save to disk
        synthetic_dataset.save_to_disk(str(save_dir / "synthetic_safe"))
        additional_datasets['synthetic_safe'] = synthetic_dataset
        
        logger.info(f"Synthetic safe prompts created: {len(synthetic_dataset)} examples")
        
    except Exception as e:
        logger.warning(f"Could not create synthetic safe prompts: {e}")
    
    return additional_datasets


def create_combined_dataset_with_wildjailbreak(
    jailbreak_dataset: DatasetDict,
    reasoning_dataset: DatasetDict,
    wildjailbreak_dataset: DatasetDict,
    additional_datasets: Dict[str, Any],
    save_path: str = "data/processed"
) -> DatasetDict:
    """
    Create a combined dataset from all sources
    
    Args:
        jailbreak_dataset: Main jailbreak dataset
        additional_datasets: Additional datasets
        save_path: Path to save the combined dataset
        
    Returns:
        Combined dataset
    """
    logger.info("Creating combined dataset...")
    
    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Start with the main dataset
    combined_train = jailbreak_dataset['train']
    combined_test = jailbreak_dataset.get('test', None)
    
    # Add reasoning dataset
    if reasoning_dataset and 'train' in reasoning_dataset:
        reasoning_train = reasoning_dataset['train']
        
        # Convert reasoning dataset to match format
        reasoning_prompts = reasoning_train['prompt']
        reasoning_types = reasoning_train['type']
        
        reasoning_df = Dataset.from_dict({
            'prompt': reasoning_prompts,
            'type': reasoning_types
        })
        
        combined_train = concatenate_datasets([combined_train, reasoning_df])
        logger.info(f"Added {len(reasoning_df)} reasoning examples")
    
    # Add toxicity data (as safe examples)
    if 'toxicity' in additional_datasets:
        toxicity_data = additional_datasets['toxicity']
        
        # Convert to jailbreak format
        toxicity_prompts = toxicity_data['comment_text']
        toxicity_labels = ['benign'] * len(toxicity_prompts)
        
        toxicity_df = Dataset.from_dict({
            'prompt': toxicity_prompts,
            'type': toxicity_labels
        })
        
        combined_train = concatenate_datasets([combined_train, toxicity_df])
        logger.info(f"Added {len(toxicity_df)} toxicity examples")
    
    # Add synthetic safe prompts
    if 'synthetic_safe' in additional_datasets:
        synthetic_data = additional_datasets['synthetic_safe']
        combined_train = concatenate_datasets([combined_train, synthetic_data])
        logger.info(f"Added {len(synthetic_data)} synthetic safe examples")
    
    # Shuffle the combined dataset
    combined_train = combined_train.shuffle(seed=42)
    
    # Create train/val/test split if no test set exists
    if combined_test is None:
        # 80% train, 10% val, 10% test
        total_size = len(combined_train)
        train_size = int(0.8 * total_size)
        val_size = int(0.1 * total_size)
        
        train_dataset = combined_train.select(range(train_size))
        val_dataset = combined_train.select(range(train_size, train_size + val_size))
        test_dataset = combined_train.select(range(train_size + val_size, total_size))
        
        combined_dataset = DatasetDict({
            'train': train_dataset,
            'validation': val_dataset,
            'test': test_dataset
        })
    else:
        # Use smaller validation set (5% of train data)
        val_size = min(200, int(0.1 * len(combined_train)))
        combined_dataset = DatasetDict({
            'train': combined_train,
            'validation': combined_train.select(range(val_size)),
            'test': combined_test
        })
    
    # Save combined dataset
    combined_dataset.save_to_disk(str(save_dir / "combined_dataset"))
    
    logger.info(f"Combined dataset saved to: {save_dir / 'combined_dataset'}")
    
    # Print final statistics
    for split_name, split_data in combined_dataset.items():
        logger.info(f"{split_name}: {len(split_data)} examples")
        
        # Count labels
        if 'type' in split_data.column_names:
            labels = split_data['type']
            unique_labels, counts = np.unique(labels, return_counts=True)
            label_counts = dict(zip(unique_labels, counts))
            logger.info(f"  Label distribution: {label_counts}")
    
    return combined_dataset


def create_combined_dataset_with_wildjailbreak(
    jailbreak_dataset: DatasetDict,
    reasoning_dataset: DatasetDict,
    wildjailbreak_dataset: DatasetDict,
    additional_datasets: Dict[str, Any],
    save_path: str = "data/processed"
) -> DatasetDict:
    """
    Create a combined dataset from all sources including WildJailbreak
    
    Args:
        jailbreak_dataset: Main jailbreak dataset
        reasoning_dataset: Reasoning dataset
        wildjailbreak_dataset: WildJailbreak dataset
        additional_datasets: Additional datasets
        save_path: Path to save the combined dataset
        
    Returns:
        Combined dataset
    """
    logger.info("Creating combined dataset with WildJailbreak...")
    
    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Start with the main dataset
    combined_train = jailbreak_dataset['train']
    combined_test = jailbreak_dataset.get('test', None)
    
    # Add reasoning dataset
    if reasoning_dataset and 'train' in reasoning_dataset:
        reasoning_train = reasoning_dataset['train']
        
        # Convert reasoning dataset to match format
        reasoning_prompts = reasoning_train['prompt']
        reasoning_types = reasoning_train['type']
        
        reasoning_df = Dataset.from_dict({
            'prompt': reasoning_prompts,
            'type': reasoning_types
        })
        
        combined_train = concatenate_datasets([combined_train, reasoning_df])
        logger.info(f"Added {len(reasoning_df)} reasoning examples")
    
    # Add WildJailbreak dataset
    if wildjailbreak_dataset and 'train' in wildjailbreak_dataset:
        wildjailbreak_train = wildjailbreak_dataset['train']
        combined_train = concatenate_datasets([combined_train, wildjailbreak_train])
        logger.info(f"Added {len(wildjailbreak_train)} WildJailbreak examples")
    
    # Add toxicity data (as safe examples)
    if 'toxicity' in additional_datasets:
        toxicity_data = additional_datasets['toxicity']
        
        # Convert to jailbreak format
        toxicity_prompts = toxicity_data['comment_text']
        toxicity_labels = ['benign'] * len(toxicity_prompts)
        
        toxicity_df = Dataset.from_dict({
            'prompt': toxicity_prompts,
            'type': toxicity_labels
        })
        
        combined_train = concatenate_datasets([combined_train, toxicity_df])
        logger.info(f"Added {len(toxicity_df)} toxicity examples")
    
    # Add synthetic safe prompts
    if 'synthetic_safe' in additional_datasets:
        synthetic_data = additional_datasets['synthetic_safe']
        combined_train = concatenate_datasets([combined_train, synthetic_data])
        logger.info(f"Added {len(synthetic_data)} synthetic safe examples")
    
    # Shuffle the combined dataset
    combined_train = combined_train.shuffle(seed=42)
    
    # Create train/val/test split if no test set exists
    if combined_test is None:
        # 80% train, 10% val, 10% test
        total_size = len(combined_train)
        train_size = int(0.8 * total_size)
        val_size = int(0.1 * total_size)
        
        train_dataset = combined_train.select(range(train_size))
        val_dataset = combined_train.select(range(train_size, train_size + val_size))
        test_dataset = combined_train.select(range(train_size + val_size, total_size))
        
        combined_dataset = DatasetDict({
            'train': train_dataset,
            'validation': val_dataset,
            'test': test_dataset
        })
    else:
        # Use smaller validation set (5% of train data)
        val_size = min(200, int(0.1 * len(combined_train)))
        combined_dataset = DatasetDict({
            'train': combined_train,
            'validation': combined_train.select(range(val_size)),
            'test': combined_test
        })
    
    # Save combined dataset
    combined_dataset.save_to_disk(str(save_dir / "combined_dataset"))
    
    logger.info(f"Combined dataset saved to: {save_dir / 'combined_dataset'}")
    
    # Print final statistics
    for split_name, split_data in combined_dataset.items():
        logger.info(f"{split_name}: {len(split_data)} examples")
        
        # Count labels
        if 'type' in split_data.column_names:
            labels = split_data['type']
            unique_labels, counts = np.unique(labels, return_counts=True)
            label_counts = dict(zip(unique_labels, counts))
            logger.info(f"  Label distribution: {label_counts}")
    
    return combined_dataset


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Download datasets for jailbreak detection")
    parser.add_argument(
        "--raw_path", 
        type=str, 
        default="data/raw",
        help="Path to save raw datasets"
    )
    parser.add_argument(
        "--processed_path", 
        type=str, 
        default="data/processed",
        help="Path to save processed datasets"
    )
    parser.add_argument(
        "--skip_additional", 
        action="store_true",
        help="Skip downloading additional datasets"
    )
    parser.add_argument(
        "--wildjailbreak_only", 
        action="store_true",
        help="Only download WildJailbreak dataset"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("Starting dataset download process...")
    
    try:
        if args.wildjailbreak_only:
            # Only download WildJailbreak dataset
            logger.info("Downloading only WildJailbreak dataset...")
            wildjailbreak_dataset = download_wildjailbreak_dataset(args.raw_path)
            
            # Save as the combined dataset directly
            wildjailbreak_dataset.save_to_disk(str(Path(args.processed_path) / "combined_dataset"))
            
            logger.info("WildJailbreak dataset download completed successfully!")
            logger.info(f"Dataset saved to: {args.processed_path}")
        else:
            # Download main dataset
            jailbreak_dataset = download_jailbreak_dataset(args.raw_path)
            
            # Download reasoning dataset
            reasoning_dataset = download_jailbreak_reasoning_dataset(args.raw_path)
            
            # Download WildJailbreak dataset
            logger.info("Downloading WildJailbreak dataset...")
            wildjailbreak_dataset = download_wildjailbreak_dataset(args.raw_path)
            
            # Download additional datasets
            additional_datasets = {}
            if not args.skip_additional:
                additional_datasets = download_additional_datasets(args.raw_path)
            
            # Create combined dataset with all datasets including WildJailbreak
            combined_dataset = create_combined_dataset_with_wildjailbreak(
                jailbreak_dataset,
                reasoning_dataset,
                wildjailbreak_dataset,
                additional_datasets, 
                args.processed_path
            )
            
            logger.info("Dataset download process completed successfully!")
            logger.info(f"Raw datasets saved to: {args.raw_path}")
            logger.info(f"Processed dataset saved to: {args.processed_path}")
        
    except Exception as e:
        logger.error(f"Dataset download failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
