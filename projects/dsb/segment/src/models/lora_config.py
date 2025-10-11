"""
LoRA configuration for ELECTRA fine-tuning.
"""

from peft import LoraConfig, TaskType


def get_lora_config(
    r: int = 16,
    lora_alpha: int = 32,
    target_modules: list = None,
    lora_dropout: float = 0.1,
    bias: str = "none"
) -> LoraConfig:
    """Get LoRA configuration for ELECTRA model."""
    
    if target_modules is None:
        target_modules = ["query", "value"]
    
    return LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias=bias,
    )


def get_electra_target_modules():
    """Get target modules for ELECTRA model."""
    return [
        "query",
        "value", 
        "dense",
        "classifier"
    ]