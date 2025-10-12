"""
ELECTRA-based Jailbreak Detection Model

This module implements the core ELECTRA classification model with LoRA support
for detecting prompt jailbreak attacks.
"""

import torch
import torch.nn as nn
from transformers import (
    ElectraForSequenceClassification,
    ElectraTokenizer,
    AutoConfig,
    AutoTokenizer
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from safetensors.torch import save_file, load_file
import os
from typing import Dict, Any, Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)


class SegmentModel:
    """ELECTRA-based Jailbreak Guard Model with LoRA support"""
    
    def __init__(
        self, 
        config: Dict[str, Any], 
        use_lora: bool = False,
        device: Optional[str] = None
    ):
        """
        Initialize the Segment model
        
        Args:
            config: Model configuration dictionary
            use_lora: Whether to use LoRA fine-tuning
            device: Device to use ('cuda', 'cpu', or 'auto')
        """
        self.config = config
        self.use_lora = use_lora
        self.model_name = config.get('model', {}).get('name', 'google/electra-large-discriminator')
        self.num_labels = config.get('model', {}).get('num_labels', 2)
        
        # Set device
        if device == 'auto' or device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        logger.info(f"Using device: {self.device}")
        
        # Initialize tokenizer and model
        self._init_tokenizer()
        self._init_model()
        
    def _init_tokenizer(self):
        """Initialize the tokenizer"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                use_fast=True
            )
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            logger.info(f"Tokenizer loaded: {self.model_name}")
            
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            raise
            
    def _init_model(self):
        """Initialize the model"""
        try:
            # Load model configuration
            model_config = AutoConfig.from_pretrained(
                self.model_name,
                num_labels=self.num_labels,
                problem_type="single_label_classification"
            )
            
            # Set dropout rates
            dropout = self.config.get('model', {}).get('dropout', 0.1)
            model_config.hidden_dropout_prob = dropout
            model_config.attention_probs_dropout_prob = dropout
            model_config.classifier_dropout = dropout
            
            # Load model
            self.model = ElectraForSequenceClassification.from_pretrained(
                self.model_name,
                config=model_config
            )
            
            # Apply LoRA if requested
            if self.use_lora:
                self._apply_lora()
                
            # Move to device
            self.model.to(self.device)
            
            logger.info(f"Model loaded: {self.model_name} (LoRA: {self.use_lora})")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
            
    def _apply_lora(self):
        """Apply LoRA configuration to the model"""
        lora_config = self.config.get('lora', {})
        
        lora = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=lora_config.get('r', 16),
            lora_alpha=lora_config.get('lora_alpha', 32),
            lora_dropout=lora_config.get('lora_dropout', 0.1),
            target_modules=lora_config.get('target_modules', ['query', 'value']),
            bias=lora_config.get('bias', 'none'),
        )
        
        self.model = get_peft_model(self.model, lora)
        
        # Enable training mode for LoRA parameters
        self.model.print_trainable_parameters()
        
        logger.info("LoRA configuration applied")
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model
        
        Args:
            input_ids: Tokenized input IDs
            attention_mask: Attention mask
            
        Returns:
            Model logits
        """
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            
        return outputs.logits
        
    def predict(
        self, 
        texts: List[str], 
        max_length: int = 512,
        return_probabilities: bool = True
    ) -> Dict[str, Any]:
        """
        Make predictions on input texts
        
        Args:
            texts: List of input texts
            max_length: Maximum sequence length
            return_probabilities: Whether to return probability scores
            
        Returns:
            Dictionary with predictions and optionally probabilities
        """
        self.model.eval()
        
        # Tokenize inputs
        inputs = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt"
        ).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            
        # Convert to probabilities
        probabilities = torch.softmax(logits, dim=-1)
        predictions = torch.argmax(probabilities, dim=-1)
        
        results = {
            'predictions': predictions.cpu().numpy().tolist(),
            'labels': ['safe' if pred == 0 else 'unsafe' for pred in predictions.cpu().numpy()]
        }
        
        if return_probabilities:
            results['probabilities'] = probabilities.cpu().numpy().tolist()
            results['unsafe_probabilities'] = probabilities[:, 1].cpu().numpy().tolist()
            
        return results
        
    def save_model(self, path: str):
        """
        Save the model to disk using safetensors format
        
        Args:
            path: Directory path to save the model
        """
        os.makedirs(path, exist_ok=True)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(path)
        
        # Save model
        if self.use_lora:
            # Save LoRA model
            self.model.save_pretrained(path, safe_serialization=True)
        else:
            # Save full model
            self.model.save_pretrained(path, safe_serialization=True)
            
        # Save configuration
        import json
        with open(os.path.join(path, 'segment_config.json'), 'w') as f:
            json.dump(self.config, f, indent=2)
            
        logger.info(f"Model saved to: {path}")
        
    @classmethod
    def load_model(cls, path: str, device: Optional[str] = None) -> 'SegmentModel':
        """
        Load a saved model from disk
        
        Args:
            path: Directory path containing the saved model
            device: Device to use ('cuda', 'cpu', or 'auto')
            
        Returns:
            Loaded SegmentModel instance
        """
        # Load configuration
        import json
        config_path = os.path.join(path, 'segment_config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            # Default configuration if not found
            config = {
                'model': {
                    'name': 'google/electra-large-discriminator',
                    'num_labels': 2,
                    'dropout': 0.1
                }
            }
            
        # Create model instance
        model = cls(config, device=device)
        
        # Load tokenizer
        model.tokenizer = AutoTokenizer.from_pretrained(path)
        
        # Load model
        try:
            # Try loading as PEFT model first
            model.model = PeftModel.from_pretrained(model.model, path)
            model.use_lora = True
        except:
            # Load as regular model
            model.model = ElectraForSequenceClassification.from_pretrained(path)
            model.use_lora = False
            
        model.model.to(model.device)
        
        logger.info(f"Model loaded from: {path}")
        
        return model
        
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model
        
        Returns:
            Dictionary with model information
        """
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'model_name': self.model_name,
            'num_labels': self.num_labels,
            'device': str(self.device),
            'use_lora': self.use_lora,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'trainable_percentage': (trainable_params / total_params) * 100 if total_params > 0 else 0
        }