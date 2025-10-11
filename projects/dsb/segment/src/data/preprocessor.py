"""
Text preprocessing utilities for jailbreak detection.
"""

import re
import html
import unicodedata
from typing import List, Optional
from transformers import AutoTokenizer


class TextPreprocessor:
    """Text preprocessing for jailbreak detection."""
    
    def __init__(self, tokenizer_name: str = "google/electra-large-discriminator"):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        if not isinstance(text, str):
            text = str(text)
        
        # Remove HTML entities
        text = html.unescape(text)
        
        # Normalize Unicode
        text = unicodedata.normalize('NFKC', text)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove control characters except newlines and tabs
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def preprocess_dataset(self, dataset, text_column: str = "text") -> List[str]:
        """Preprocess text column in dataset."""
        def preprocess_example(example):
            cleaned_text = self.clean_text(example[text_column])
            return {"text": cleaned_text}
        
        return dataset.map(preprocess_example)
    
    def tokenize_texts(self, texts: List[str], max_length: int = 256) -> dict:
        """Tokenize texts for model input."""
        return self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt"
        )
    
    def tokenize_dataset(self, dataset, text_column: str = "text", max_length: int = 256):
        """Tokenize dataset for training."""
        def tokenize_function(examples):
            return self.tokenizer(
                examples[text_column],
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors=None
            )
        
        return dataset.map(tokenize_function, batched=True)