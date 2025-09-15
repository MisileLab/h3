import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import TypedDict, override, final
from tensor_logic import TensorLogicModule


class TensorLogicOutputs(TypedDict):
  """Structure for tensor logic module outputs"""
  var_dependency: torch.Tensor
  type_consistency: torch.Tensor
  scope_validity: torch.Tensor
  logic_flow: torch.Tensor


class ModelOutput(TypedDict):
  """Structure for model forward pass outputs"""
  logits: torch.Tensor
  loss: torch.Tensor | None
  tensor_outputs: TensorLogicOutputs


class GemmaWithTensorLogic(nn.Module):
  """
  Gemma model enhanced with Tensor Logic Module for structural code reasoning.
  Inserts the tensor logic module between specified transformer layers.
  """

  @final
  def __init__(
    self,
    model_name: str = "google/gemma-3-270m",
    insertion_layer: int = 12,  # Middle layer for 4B model
    num_types: int = 128
  ):
    super().__init__() # pyright: ignore[reportUnknownMemberType]
    
    # Load base Gemma model
    self.base_model = AutoModelForCausalLM.from_pretrained(model_name)
    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Get model configuration
    config = self.base_model.config
    self.hidden_dim = config.hidden_size
    
    # Split transformer layers
    self.pre_layers = nn.ModuleList(list(self.base_model.model.layers[:insertion_layer]))
    self.post_layers = nn.ModuleList(list(self.base_model.model.layers[insertion_layer:]))
    
    # Tensor logic module
    self.tensor_module = TensorLogicModule(self.hidden_dim)
    
    # Preserve original model components
    self.model = self.base_model.model
    self.lm_head = self.base_model.lm_head

  @override
  def forward(
    self,
    input_ids: torch.LongTensor,
    attention_mask: torch.Tensor | None = None,
    labels: torch.LongTensor | None = None
  ) -> ModelOutput:
    """
    Forward pass with tensor logic integration

    Args:
        input_ids: Token IDs [batch_size, seq_len]
        attention_mask: Attention mask [batch_size, seq_len]
        labels: Optional labels for loss calculation

    Returns:
        Dictionary containing logits, optional loss, and tensor outputs
    """
    # Get embeddings
    hidden_states = self.model.embed_tokens(input_ids)

    # Process through pre-layers
    for layer in self.pre_layers:
      hidden_states = layer(hidden_states, attention_mask=attention_mask)[0]

    # Apply tensor logic module
    hidden_states, tensor_outputs = self.tensor_module(hidden_states, attention_mask)

    # Process through post-layers
    for layer in self.post_layers:
      hidden_states = layer(hidden_states, attention_mask=attention_mask)[0]

    # Final normalization and LM head
    hidden_states = self.model.norm(hidden_states)
    logits = self.lm_head(hidden_states)

    # Calculate loss if labels provided
    loss = None
    if labels is not None:
      shift_logits = logits[..., :-1, :].contiguous()
      shift_labels = labels[..., 1:].contiguous()
      loss_fct = nn.CrossEntropyLoss()
      loss = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
      )

    return {
      'logits': logits,
      'loss': loss,
      'tensor_outputs': tensor_outputs
    }

  def generate(
    self,
    prompt: str,
    max_tokens: int = 512,
    **kwargs
  ) -> str:
    """
    Generate code with tensor-enhanced reasoning

    Args:
        prompt: Input prompt for code generation
        max_tokens: Maximum tokens to generate

    Returns:
        Generated code string
    """
    inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
    
    outputs = self.base_model.generate(
      **inputs,
      max_new_tokens=max_tokens,
      **kwargs
    )
    
    return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

  @property
  def device(self):
    return next(self.parameters()).device
