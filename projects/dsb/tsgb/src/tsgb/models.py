"""LLM model wrappers for the self-play framework.

This module provides abstracted interfaces for:
- Attacker LLM (A): Generates adversarial prompts
- Guard LLM (G): Decides whether to allow/block prompts
- Surrogate Target LLM (T_s): Small open-source model for training
- Black-box Target LLM (T_b): Interface for production LLMs (Stage 2)

NOTE: This scaffold does NOT generate actual harmful jailbreak content.
All attack scenarios are abstracted and identified only by scenario IDs.
"""

from contextlib import nullcontext
from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from tsgb.logging import get_logger

logger = get_logger(__name__)


def _resolve_max_length(tokenizer: PreTrainedTokenizer, fallback: int = 4096) -> int:
    """Clamp tokenizer max_length when models expose sentinel values."""
    max_length = tokenizer.model_max_length
    if max_length is None:
        return fallback
    if max_length > 1_000_000:
        logger.warning(
            "tokenizer_max_length_clamped",
            original=max_length,
            used=fallback,
        )
        return fallback
    return int(max_length)


# Global accelerator instance for multi-GPU support
_accelerator: Accelerator | None = None


def get_accelerator(
    *,
    mixed_precision: str | None = "fp16",
    gradient_accumulation_steps: int | None = None,
) -> Accelerator:
    """Get or create the global Accelerator instance for multi-GPU support."""
    global _accelerator
    if _accelerator is None:
        accelerator_kwargs: dict[str, Any] = {}
        if mixed_precision:
            accelerator_kwargs["mixed_precision"] = mixed_precision
        if gradient_accumulation_steps:
            accelerator_kwargs["gradient_accumulation_steps"] = gradient_accumulation_steps
        _accelerator = Accelerator(**accelerator_kwargs)
        logger.info(
            "accelerator_initialized",
            device=str(_accelerator.device),
            num_processes=_accelerator.num_processes,
            distributed_type=str(_accelerator.distributed_type),
            mixed_precision=mixed_precision,
            gradient_accumulation_steps=gradient_accumulation_steps,
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
    attn_implementation: str | None = "flash_attention_2"
    gradient_checkpointing: bool = True
    use_mixed_precision: bool = True


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

        self.max_length = _resolve_max_length(self.tokenizer)

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        role: LLMRole,
        device: str = "auto",
        torch_dtype: str = "auto",
        use_accelerate: bool = True,
        accelerator: Accelerator | None = None,
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
        accel = accelerator or (get_accelerator() if use_accelerate else None)

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
        if use_accelerate and accel is not None:
            # Use accelerate's device for single GPU, or device_map for multi-GPU
            if accel.num_processes > 1 or multi_gpu_available:
                device_map = "auto"
            else:
                device_map = accel.device
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

        attn_impl = kwargs.pop("attn_implementation", "flash_attention_2")
        gradient_checkpointing = kwargs.pop("gradient_checkpointing", True)
        use_mixed_precision = kwargs.pop("use_mixed_precision", True)

        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=dtype,
                device_map=device_map,
                max_memory=max_memory,
                trust_remote_code=kwargs.get("trust_remote_code", False),
                attn_implementation=attn_impl,
            )
        except TypeError:
            logger.warning(
                "attn_impl_not_supported",
                model_name=model_name,
                attn_implementation=attn_impl,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=dtype,
                device_map=device_map,
                max_memory=max_memory,
                trust_remote_code=kwargs.get("trust_remote_code", False),
            )

        if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
            try:
                model.gradient_checkpointing_enable()
                if hasattr(model.config, "use_cache"):
                    model.config.use_cache = False
                logger.info("gradient_checkpointing_enabled", model_name=model_name)
            except Exception as checkpoint_error:  # pragma: no cover - runtime safety
                logger.warning("gradient_checkpointing_failed", error=str(checkpoint_error))

        config = LLMConfig(
            model_name=model_name,
            role=role,
            device=device,
            torch_dtype=torch_dtype,
            use_accelerate=use_accelerate,
            attn_implementation=attn_impl,
            gradient_checkpointing=gradient_checkpointing,
            use_mixed_precision=use_mixed_precision,
            **kwargs,
        )

        logger.info(
            "model_loaded",
            model_name=model_name,
            device=str(device_map),
            num_gpus=accel.num_processes if accel else 1,
        )
        return cls(model, tokenizer, config, accel)

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
        requested_max_length = kwargs.get("max_length")
        max_length = (
            min(int(requested_max_length), self.max_length)
            if requested_max_length is not None
            else self.max_length
        )

        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
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

        use_amp = self.config.use_mixed_precision and torch.cuda.is_available()
        autocast_context = (
            (
                self.accelerator.autocast()
                if self.accelerator
                else torch.autocast("cuda", dtype=torch.float16)
            )
            if use_amp
            else nullcontext()
        )

        # Generate
        with torch.no_grad():
            with autocast_context:
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

    def generate_batch(
        self,
        prompts: list[str],
        return_logits: bool = False,
        **kwargs: Any,
    ) -> list[GenerationOutput]:
        """Generate batched outputs for a list of prompts."""
        if not prompts:
            return []

        requested_max_length = kwargs.get("max_length")
        max_length = (
            min(int(requested_max_length), self.max_length)
            if requested_max_length is not None
            else self.max_length
        )

        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(self._device)

        attention_mask = inputs.get("attention_mask")
        input_ids = inputs["input_ids"]

        gen_kwargs = {
            "max_new_tokens": kwargs.get("max_new_tokens", self.config.max_new_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
            "top_p": kwargs.get("top_p", self.config.top_p),
            "do_sample": kwargs.get("do_sample", self.config.do_sample),
            "pad_token_id": self.tokenizer.pad_token_id,
            "output_scores": return_logits,
            "return_dict_in_generate": True,
            "attention_mask": attention_mask,
        }

        use_amp = self.config.use_mixed_precision and torch.cuda.is_available()
        autocast_context = (
            (
                self.accelerator.autocast()
                if self.accelerator
                else torch.autocast("cuda", dtype=torch.float16)
            )
            if use_amp
            else nullcontext()
        )

        with torch.no_grad():
            with autocast_context:
                outputs = self.model.generate(input_ids=input_ids, **gen_kwargs)

        sequences = outputs.sequences
        if attention_mask is not None:
            input_lengths = attention_mask.sum(dim=1)
        else:
            input_lengths = torch.full(
                (sequences.size(0),), input_ids.shape[1], device=sequences.device
            )

        logits_tensor = None
        if return_logits and hasattr(outputs, "scores") and outputs.scores:
            logits_tensor = torch.stack(outputs.scores, dim=1)

        results: list[GenerationOutput] = []
        for idx, prompt in enumerate(prompts):
            start = int(input_lengths[idx].item())
            output_ids = sequences[idx, start:]
            sample_logits = logits_tensor[idx] if logits_tensor is not None else None
            results.append(
                GenerationOutput(
                    text=self.tokenizer.decode(output_ids, skip_special_tokens=True),
                    input_ids=input_ids[idx : idx + 1].detach().cpu(),
                    output_ids=output_ids.unsqueeze(0).detach().cpu(),
                    logits=sample_logits.detach().cpu() if sample_logits is not None else None,
                )
            )

        return results

        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
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

        use_amp = self.config.use_mixed_precision and torch.cuda.is_available()
        autocast_context = (
            (
                self.accelerator.autocast()
                if self.accelerator
                else torch.autocast("cuda", dtype=torch.float16)
            )
            if use_amp
            else nullcontext()
        )

        # Generate
        with torch.no_grad():
            with autocast_context:
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

        pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id or 0
        attention_mask = (full_ids != pad_token_id).long()

        use_amp = self.config.use_mixed_precision and self._device.type == "cuda"
        autocast_context = (
            (
                self.accelerator.autocast()
                if self.accelerator
                else torch.autocast("cuda", dtype=torch.float16)
            )
            if use_amp
            else nullcontext()
        )

        with autocast_context:
            outputs = self.model(full_ids, attention_mask=attention_mask)
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
