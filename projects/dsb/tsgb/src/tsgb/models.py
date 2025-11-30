"""LLM model wrappers for the self-play framework.

This module provides abstracted interfaces for:
- Attacker LLM (A): Generates adversarial prompts
- Guard LLM (G): Decides whether to allow/block prompts
- Surrogate Target LLM (T_s): Small open-source model for training
- Black-box Target LLM (T_b): Interface for production LLMs (Stage 2)

NOTE: This scaffold does NOT generate actual harmful jailbreak content.
All attack scenarios are abstracted and identified only by scenario IDs.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from tsgb.logging import get_logger

logger = get_logger(__name__)

# Global accelerator instance for multi-GPU support
_accelerator: Accelerator | None = None


def get_accelerator() -> Accelerator:
    """Get or create the global Accelerator instance for multi-GPU support."""
    global _accelerator
    if _accelerator is None:
        _accelerator = Accelerator()
        logger.info(
            "accelerator_initialized",
            device=str(_accelerator.device),
            num_processes=_accelerator.num_processes,
            distributed_type=str(_accelerator.distributed_type),
        )
    return _accelerator


class LLMRole(str, Enum):
    """Role of an LLM in the self-play framework."""

    ATTACKER = "attacker"
    GUARD = "guard"
    TARGET = "target"


@dataclass
class LLMConfig:
    """Configuration for an LLM instance."""

    model_name: str
    role: LLMRole
    max_new_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True
    device: str = "auto"
    torch_dtype: str = "auto"
    trust_remote_code: bool = False
    use_accelerate: bool = True  # Enable multi-GPU by default


@dataclass
class GenerationOutput:
    """Output from LLM generation."""

    text: str
    input_ids: torch.Tensor | None = None
    output_ids: torch.Tensor | None = None
    logits: torch.Tensor | None = None


class HuggingFaceLM:
    """Wrapper for HuggingFace causal language models.

    Provides a unified interface for text generation with support for
    returning logits (needed for RL training). Supports multi-GPU via accelerate.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        config: LLMConfig,
        accelerator: Accelerator | None = None,
    ) -> None:
        """Initialize the LLM wrapper.

        Args:
            model: The loaded HuggingFace model.
            tokenizer: The corresponding tokenizer.
            config: Configuration for generation.
            accelerator: Optional Accelerator for multi-GPU support.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.accelerator = accelerator
        self._device = next(model.parameters()).device

        # Ensure tokenizer has pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        role: LLMRole,
        device: str = "auto",
        torch_dtype: str = "auto",
        use_accelerate: bool = True,
        **kwargs: Any,
    ) -> "HuggingFaceLM":
        """Load a model from HuggingFace Hub or local path.

        Args:
            model_name: HuggingFace model ID or local path.
            role: The role of this model in the framework.
            device: Device to load the model on.
            torch_dtype: Data type for model weights.
            use_accelerate: Whether to use accelerate for multi-GPU support.
            **kwargs: Additional arguments for LLMConfig.

        Returns:
            An initialized HuggingFaceLM instance.
        """
        logger.info("loading_model", model_name=model_name, role=role.value)

        # Get accelerator if using multi-GPU
        accelerator = get_accelerator() if use_accelerate else None

        # Resolve torch dtype
        dtype = None
        if torch_dtype == "auto":
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        elif torch_dtype == "float16":
            dtype = torch.float16
        elif torch_dtype == "bfloat16":
            dtype = torch.bfloat16
        else:
            dtype = torch.float32

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        multi_gpu_available = torch.cuda.is_available() and torch.cuda.device_count() > 1
        max_memory = None

        # Load model with device_map for multi-GPU
        if use_accelerate and accelerator is not None:
            # Use accelerate's device for single GPU, or device_map for multi-GPU
            if accelerator.num_processes > 1 or multi_gpu_available:
                device_map = "auto"
            else:
                device_map = accelerator.device
        else:
            if device != "auto":
                device_map = device
            elif multi_gpu_available:
                device_map = "auto"
            else:
                device_map = "cuda" if torch.cuda.is_available() else "cpu"

        if device_map == "auto" and multi_gpu_available:
            max_memory = {
                i: int(torch.cuda.get_device_properties(i).total_memory * 0.9)
                for i in range(torch.cuda.device_count())
            }

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dtype,
            device_map=device_map,
            max_memory=max_memory,
            trust_remote_code=kwargs.get("trust_remote_code", False),
        )

        config = LLMConfig(
            model_name=model_name,
            role=role,
            device=device,
            torch_dtype=torch_dtype,
            use_accelerate=use_accelerate,
            **kwargs,
        )

        logger.info(
            "model_loaded",
            model_name=model_name,
            device=str(device_map),
            num_gpus=accelerator.num_processes if accelerator else 1,
        )
        return cls(model, tokenizer, config, accelerator)

    def generate(
        self,
        prompt: str,
        return_logits: bool = False,
        **kwargs: Any,
    ) -> GenerationOutput:
        """Generate text from a prompt.

        Args:
            prompt: The input prompt string.
            return_logits: Whether to return logits (for RL training).
            **kwargs: Override generation parameters.

        Returns:
            GenerationOutput with generated text and optional tensors.
        """
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=kwargs.get("max_length", self.tokenizer.model_max_length),
        ).to(self._device)

        input_ids = inputs["input_ids"]

        # Generation parameters
        gen_kwargs = {
            "max_new_tokens": kwargs.get("max_new_tokens", self.config.max_new_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
            "top_p": kwargs.get("top_p", self.config.top_p),
            "do_sample": kwargs.get("do_sample", self.config.do_sample),
            "pad_token_id": self.tokenizer.pad_token_id,
            "output_scores": return_logits,
            "return_dict_in_generate": True,
            "attention_mask": inputs.get("attention_mask"),
        }

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(input_ids, **gen_kwargs)

        # Extract generated tokens (excluding input)
        output_ids = outputs.sequences[:, input_ids.shape[1] :]

        # Decode to text
        generated_text = self.tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True,
        )

        # Get logits if requested
        logits = None
        if return_logits and hasattr(outputs, "scores") and outputs.scores:
            # Stack all step scores into a tensor
            logits = torch.stack(outputs.scores, dim=1)

        return GenerationOutput(
            text=generated_text,
            input_ids=input_ids,
            output_ids=output_ids,
            logits=logits,
        )

    def get_log_probs(
        self,
        input_ids: torch.Tensor,
        target_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Compute log probabilities for target tokens given input.

        Used for PPO advantage computation.

        Args:
            input_ids: Input token IDs.
            target_ids: Target token IDs to compute log probs for.

        Returns:
            Tensor of log probabilities.
        """
        # Combine input and target
        full_ids = torch.cat([input_ids, target_ids], dim=1)

        with torch.no_grad():
            outputs = self.model(full_ids)
            logits = outputs.logits

        # Get logits for the target positions
        target_logits = logits[:, input_ids.shape[1] - 1 : -1, :]

        # Compute log probabilities
        log_probs = torch.nn.functional.log_softmax(target_logits, dim=-1)

        # Gather log probs for actual target tokens
        target_log_probs = log_probs.gather(
            dim=-1,
            index=target_ids.unsqueeze(-1),
        ).squeeze(-1)

        return target_log_probs

    @property
    def device(self) -> torch.device:
        """Get the device the model is on."""
        return self._device
