"""
ELECTRA-based jailbreak detection classifier.
"""

import torch
import torch.nn as nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoConfig
)
from peft import LoraConfig, get_peft_model, TaskType
from typing import Dict, List, Optional, Tuple
import numpy as np


class ElectraJailbreakClassifier(nn.Module):
    """ELECTRA-based classifier for jailbreak detection."""
    
    def __init__(
        self,
        model_name: str = "google/electra-large-discriminator",
        num_labels: int = 2,
        dropout: float = 0.1,
        use_lora: bool = False,
        lora_config: Optional[Dict] = None
    ):
        super().__init__()
        self.model_name = model_name
        self.num_labels = num_labels
        self.use_lora = use_lora
        
        # Load base model
        self.config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
        self.base_model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            config=self.config
        )
        
        # Add dropout if specified
        if dropout > 0:
            self.base_model.classifier.dropout = nn.Dropout(dropout)
        
        # Apply LoRA if specified
        if use_lora and lora_config:
            lora_cfg = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                **lora_config
            )
            self.base_model = get_peft_model(self.base_model, lora_cfg)
            self.base_model.print_trainable_parameters()
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, 
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        return outputs
    
    def predict(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Make predictions."""
        with torch.no_grad():
            outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            return probabilities
    
    def predict_single(self, text: str, tokenizer, max_length: int = 256) -> Dict:
        """Predict for a single text input."""
        inputs = tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt"
        )
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
            self.cuda()
        
        with torch.no_grad():
            outputs = self.base_model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            
            # Get prediction and confidence
            predicted_class = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0][predicted_class].item()
            unsafe_prob = probabilities[0][1].item()  # Probability of class 1 (unsafe)
            
            return {
                "predicted_class": predicted_class,
                "confidence": confidence,
                "unsafe_probability": unsafe_prob,
                "safe_probability": probabilities[0][0].item(),
                "logits": logits[0].cpu().numpy().tolist(),
                "probabilities": probabilities[0].cpu().numpy().tolist()
            }
    
    def save_model(self, save_path: str, save_format: str = "safetensors"):
        """Save model to disk."""
        if self.use_lora:
            self.base_model.save_pretrained(save_path, safe_serialization=(save_format == "safetensors"))
        else:
            self.base_model.save_pretrained(save_path, safe_serialization=(save_format == "safetensors"))
        
        # Save tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        tokenizer.save_pretrained(save_path)
    
    @classmethod
    def load_model(cls, model_path: str, use_lora: bool = False):
        """Load model from disk."""
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Load model
        if use_lora:
            from peft import PeftModel
            base_model = AutoModelForSequenceClassification.from_pretrained(
                "google/electra-large-discriminator"
            )
            model = PeftModel.from_pretrained(base_model, model_path)
        else:
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
        
        # Create classifier instance
        classifier = cls.__new__(cls)
        classifier.model_name = model_path
        classifier.num_labels = model.config.num_labels
        classifier.use_lora = use_lora
        classifier.config = model.config
        classifier.base_model = model
        
        return classifier, tokenizer
    
    def get_trainable_parameters(self) -> Tuple[int, int]:
        """Get number of trainable and total parameters."""
        if self.use_lora:
            trainable_params = sum(p.numel() for p in self.base_model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.base_model.parameters())
        else:
            trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.parameters())
        
        return trainable_params, total_params
    
    def freeze_base_model(self):
        """Freeze base model parameters (only train classifier head)."""
        if self.use_lora:
            # For LoRA, only LoRA parameters are trainable by default
            pass
        else:
            for param in self.base_model.electra.parameters():
                param.requires_grad = False
    
    def unfreeze_all(self):
        """Unfreeze all parameters for full fine-tuning."""
        for param in self.base_model.parameters():
            param.requires_grad = True