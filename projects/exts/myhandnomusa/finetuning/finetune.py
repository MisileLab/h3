import os
import torch
import trackio as wandb  # trackio는 wandb와 API 호환
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from safetensors.torch import save_file

# --- Configuration ---
MODEL_ID = "gpt-oss/gpt-oss-120b"  # Example model
MIN_VRAM_GB = 80  # Minimum required GPU VRAM
OUTPUT_DIR = "./gpt-oss-120b-finetuned"
HUB_DATASET_ID = "misilelab/korean-law-dataset"

def check_gpu_vram():
    """Checks if the available GPU VRAM meets the minimum requirement."""
    if not torch.cuda.is_available():
        print("CUDA is not available. Fine-tuning requires a GPU.")
        return False
    total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    if total_vram < MIN_VRAM_GB:
        print(f"Error: Insufficient GPU VRAM. Found {total_vram:.2f}GB, but require {MIN_VRAM_GB}GB.")
        return False
    print(f"Sufficient GPU VRAM found: {total_vram:.2f}GB")
    return True

def load_and_prepare_data():
    """Loads the training dataset from the Hugging Face Hub."""
    print(f"Loading dataset '{HUB_DATASET_ID}' from the Hub...")
    # The Trainer API expects a single dataset for training.
    # We load the 'train' split here.
    dataset = load_dataset(HUB_DATASET_ID, split="train")
    print("Dataset loaded successfully.")
    return dataset

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
    
    if not check_gpu_vram():
        wandb.finish()
        return
    print("Loading and preparing dataset...")
    dataset = load_and_prepare_data()
    if dataset is None:
        return

    # --- Tokenization ---
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # --- Model Loading and Configuration ---
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=quantization_config,
        device_map="auto",
    )

    model = prepare_model_for_kbit_training(model)

    # --- LoRA Configuration ---
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)

    # --- Training ---
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        save_total_limit=2,
        report_to="none", # trackio 사용으로 wandb 리포팅 비활성화
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )

    print("Starting fine-tuning...")
    trainer.train()
    print("Fine-tuning finished.")
    
    # 훈련 메트릭 로깅
    wandb.log({"training_completed": True, "final_epoch": 1})

    # --- Save Model with SafeTensors ---
    print("Saving model with safetensors...")
    model.save_pretrained(OUTPUT_DIR)
    # If you want to save only the LoRA weights
    # save_file(model.state_dict(), os.path.join(OUTPUT_DIR, "adapter_model.safetensors"))
    
    # To save the full model, you might need to merge the weights first
    merged_model = model.merge_and_unload()
    save_file(merged_model.state_dict(), os.path.join(OUTPUT_DIR, "model.safetensors"))
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print(f"Model saved to {OUTPUT_DIR}")
    
    # trackio 실험 종료
    wandb.finish()

if __name__ == "__main__":
    main()
