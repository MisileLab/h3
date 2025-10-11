"""
Data collator for jailbreak detection training.
"""

from transformers import DataCollatorWithPadding
from typing import Dict, List, Any


class JailbreakDataCollator(DataCollatorWithPadding):
    """Data collator for jailbreak detection with padding."""
    
    def __init__(self, tokenizer, return_tensors: str = "pt"):
        super().__init__(tokenizer=tokenizer, return_tensors=return_tensors)
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate batch of features."""
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        
        # Ensure labels are properly formatted
        if "label" in batch:
            batch["labels"] = batch["label"]
            del batch["label"]
        
        return batch