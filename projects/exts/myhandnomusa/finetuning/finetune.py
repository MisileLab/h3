import os
import torch
import trackio as wandb  # trackio는 wandb와 API 호환
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Mxfp4Config
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

# --- Configuration ---
MODEL_ID = "openai/gpt-oss-120b"  # Updated model
MIN_VRAM_GB = 80  # Minimum required GPU VRAM
OUTPUT_DIR = "./gpt-oss-120b-finetuned"
HUB_DATASET_ID = "misilelab/korean-law-dataset"

def check_gpu_vram():
    """Checks if the available GPU VRAM meets the minimum requirement."""
    if not torch.cuda.is_available():
        print("CUDA is not available. Fine-tuning requires a GPU.")
        return False, 0
    
    num_gpus = torch.cuda.device_count()
    print(f"Found {num_gpus} GPU(s)")
    
    total_vram = 0
    gpu_info = []
    
    for i in range(num_gpus):
        gpu_props = torch.cuda.get_device_properties(i)
        gpu_vram = gpu_props.total_memory / (1024**3)
        total_vram += gpu_vram
        gpu_info.append(f"GPU {i}: {gpu_props.name} - {gpu_vram:.2f}GB")
        print(gpu_info[-1])
    
    print(f"Total VRAM across all GPUs: {total_vram:.2f}GB")
    
    if total_vram < MIN_VRAM_GB:
        print(f"Error: Insufficient total GPU VRAM. Found {total_vram:.2f}GB, but require {MIN_VRAM_GB}GB.")
        return False, num_gpus
    
    print(f"Sufficient GPU VRAM found: {total_vram:.2f}GB across {num_gpus} GPU(s)")
    return True, num_gpus

def load_and_prepare_data():
    """Loads the training dataset from the Hugging Face Hub."""
    print(f"Loading dataset '{HUB_DATASET_ID}' from the Hub...")
    # Load the 'train' split
    dataset = load_dataset(HUB_DATASET_ID, split="train")
    print("Dataset loaded successfully.")
    return dataset

def format_dataset_for_gpt_oss(examples):
    """Convert dataset to GPT-OSS format with messages structure."""
    formatted_messages = []
    
    for text in examples["text"]:
        # Create messages format for GPT-OSS
        messages = [
            {"role": "user", "content": text}
        ]
        formatted_messages.append(messages)
    
    return {"messages": formatted_messages}

def main():
    """Main fine-tuning script."""
    # trackio 실험 시작
    wandb.init(
        project="gpt-oss-finetuning",
        name="korean-law-finetune",
        config={
            "model_id": MODEL_ID,
            "dataset": HUB_DATASET_ID,
            "min_vram_gb": MIN_VRAM_GB,
            "output_dir": OUTPUT_DIR
        }
    )
    
    vram_ok, num_gpus = check_gpu_vram()
    if not vram_ok:
        wandb.finish()
        return
    
    # GPU 개수에 따른 배치 사이즈 조정
    per_device_batch_size = 4
    if num_gpus > 1:
        print(f"Multi-GPU setup detected. Using DataParallel across {num_gpus} GPUs")
        print(f"Effective batch size will be: {per_device_batch_size * num_gpus}")
    
    wandb.config.update({
        "num_gpus": num_gpus,
        "per_device_batch_size": per_device_batch_size,
        "effective_batch_size": per_device_batch_size * num_gpus
    })
    
    print("Loading and preparing dataset...")
    dataset = load_and_prepare_data()
    if dataset is None:
        wandb.finish()
        return

    # Format dataset for GPT-OSS (convert to messages format)
    print("Formatting dataset for GPT-OSS...")
    formatted_dataset = dataset.map(format_dataset_for_gpt_oss, batched=True)

    # --- Load Tokenizer ---
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # --- Model Loading and Configuration ---
    print("Loading GPT-OSS model...")
   
    quantization_config = Mxfp4Config(dequantize=True)
    # GPT-OSS specific model loading (with Mxfp4 quantization)
    model_kwargs = {
        "quantization_config": quantization_config,
        "attn_implementation": "eager",  # Better performance for training
        "dtype": "auto",
        "use_cache": False,  # Disable cache for training with gradient checkpointing
        "device_map": "auto",  # Distribute across GPUs
        "trust_remote_code": True
    }
    
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **model_kwargs)
    print("Model loaded successfully (using Mxfp4 quantization)")

    # --- LoRA Configuration for GPT-OSS ---
    print("Setting up LoRA configuration...")
    lora_config = LoraConfig(
        r=8,  # Low-rank dimension
        lora_alpha=32,  # LoRA scaling parameter
        target_modules="all-linear",  # Target all linear layers for MoE architecture
        target_parameters=["mlp.experts.down_proj", "mlp.experts.gate_up_proj"],  # Expert-specific layers
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Apply LoRA to the model
    peft_model = get_peft_model(model, lora_config)
    print("LoRA configuration applied successfully")

    # --- Training Configuration ---
    print("Setting up training configuration...")
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        learning_rate=2e-4,
        gradient_checkpointing=True,
        num_train_epochs=1,
        logging_steps=1,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=4,
        max_length=2048,  # GPT-OSS specific sequence length
        warmup_ratio=0.03,
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr_rate": 0.1},
        report_to="none",  # Using trackio instead
        save_strategy="epoch",
        save_total_limit=2,
        dataloader_pin_memory=False,
        dataloader_num_workers=min(4, num_gpus * 2),
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
    )

    # --- Training ---
    print("Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=peft_model,
        args=training_args,
        train_dataset=formatted_dataset,
        processing_class=tokenizer,  # Use processing_class instead of tokenizer
    )

    print("Starting fine-tuning...")
    trainer.train()
    print("Fine-tuning finished.")
    
    # 훈련 메트릭 로깅
    wandb.log({"training_completed": True, "final_epoch": 1})

    # --- Save Model ---
    print("Saving LoRA adapter...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print(f"Model saved to {OUTPUT_DIR}")
    
    # --- Test Generation ---
    print("Testing generation with fine-tuned model...")
    test_messages = [
        {"role": "user", "content": "한국의 수도는 어디인가요?"}
    ]
    
    input_ids = tokenizer.apply_chat_template(
        test_messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(peft_model.device)
    
    gen_kwargs = {
        "max_new_tokens": 256,
        "do_sample": True,
        "temperature": 0.6,
        "top_p": 0.9
    }
    
    output_ids = peft_model.generate(input_ids, **gen_kwargs)
    response = tokenizer.batch_decode(output_ids)[0]
    print("Sample generation:")
    print(response)
    
    # trackio 실험 종료
    wandb.finish()

if __name__ == "__main__":
    # 멀티 GPU 환경을 위한 환경 변수 설정
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    main()
