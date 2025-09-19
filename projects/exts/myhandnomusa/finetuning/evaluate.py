
import torch
import polars as pl
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm
import asyncio

from datasets import load_dataset

# --- Configuration ---
BASE_MODEL_ID = "gpt-oss/gpt-oss-120b"  # Pre-trained model ID used for fine-tuning
ADAPTER_DIR = "./gpt-oss-120b-finetuned"  # Directory where the fine-tuned adapter is saved
HUB_DATASET_ID = "misilelab/korean-law-dataset" # Dataset repository on the Hub
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_model():
    """Loads the base model and applies the fine-tuned LoRA adapter."""
    print(f"Loading base model: {BASE_MODEL_ID}")
    
    # Use the same quantization config as in training
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=quantization_config,
        device_map="auto",
    )
    
    print(f"Loading LoRA adapter from: {ADAPTER_DIR}")
    model = PeftModel.from_pretrained(model, ADAPTER_DIR)
    
    print("Merging adapter weights into the base model...")
    model = model.merge_and_unload()
    
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)
    tokenizer.pad_token = tokenizer.eos_token
    
    print("Model and tokenizer loaded successfully.")
    return model, tokenizer

def calculate_perplexity(model, tokenizer, num_samples=50, stride=512):
    """Calculates the perplexity of the model on a given dataset."""
    print(f"\n--- Calculating Perplexity on {HUB_DATASET_ID} ---")
    try:
        # Load the test split from the Hub
        dataset = load_dataset(HUB_DATASET_ID, split="test")
        texts = dataset.select(range(num_samples))['text']
    except Exception as e:
        print(f"Failed to load or process test data from the Hub: {e}")
        return float('inf')

    encodings = tokenizer("\n\n".join(texts), return_tensors="pt")
    max_length = model.config.max_position_embeddings
    seq_len = encodings.input_ids.size(1)

    nlls = []
    prev_end_loc = 0
    for begin_loc in tqdm(range(0, seq_len, stride), desc="Calculating PPL"):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(DEVICE)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood)
        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).mean())
    print(f"--- Perplexity: {ppl.item():.4f} ---")
    return ppl.item()

def generate_text(model, tokenizer, prompt, max_new_tokens=150):
    """Generates text from a given prompt."""
    print(f"\n--- Generating text for prompt: '{prompt}' ---")
    
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("Generated Text:")
    print(response)
    print("--- End of Generation ---")
    return response

async def main():
    """Main evaluation function."""
    model, tokenizer = load_model()
    
    # 1. Quantitative Evaluation (Perplexity)
    calculate_perplexity(model, tokenizer, TEST_DATA_FILE)
    
    # 2. Qualitative Evaluation (Text Generation)
    # You can change this prompt to test different things.
    prompt = "대한민국 헌법 제32조는"
    generate_text(model, tokenizer, prompt)

    prompt_2 = "부당해고 구제신청에 대하여"
    generate_text(model, tokenizer, prompt_2)


if __name__ == "__main__":
    asyncio.run(main())
