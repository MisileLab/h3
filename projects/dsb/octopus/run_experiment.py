import argparse
import os
import random
from datasets import load_dataset
import polars as pl
from src.compression import rule_based_compression, llm_based_compression, hybrid_compression

def get_token_count(text):
    """Simple word count as a proxy for token count."""
    if not isinstance(text, str):
        return 0
    return len(text.split())

def print_section_header(title):
    """Prints a formatted section header."""
    print("\n" + "="*80)
    print(f"# {title}")
    print("="*80)

def main():
    parser = argparse.ArgumentParser(description="Run reasoning compression experiments.")
    parser.add_argument(
        "--dataset", 
        type=str, 
        choices=["gsm8k", "scicode"], 
        default="gsm8k", 
        help="The dataset to use for the experiment (gsm8k or scicode)."
    )
    parser.add_argument(
        "--sample_id",
        type=int,
        default=None,
        help="Specify a sample index to use. If not provided, a random sample is chosen."
    )
    args = parser.parse_args()

    print(f"Running experiment on the '{args.dataset}' dataset...")

    # --- 1. Load Dataset ---
    try:
        if args.dataset == "gsm8k":
            dataset = load_dataset("gsm8k", "main", split="train")
            question_col = "question"
            trace_col = "answer"
        elif args.dataset == "scicode":
            dataset = load_dataset("google/scicode", "main", split="train")
            question_col = "step_description_prompt"
            trace_col = "ground_truth_code"
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # --- 2. Select Sample ---
    if args.sample_id is not None and 0 <= args.sample_id < len(dataset):
        sample_index = args.sample_id
    else:
        sample_index = random.randint(0, len(dataset) - 1)
    
    sample = dataset[sample_index]
    question = sample[question_col]
    original_trace = sample[trace_col]

    print_section_header(f"Sample #{sample_index}: {args.dataset.upper()} Problem")
    print(question)

    print_section_header("Original Trace")
    print(original_trace)

    # --- 3. Apply Compressions ---
    print("\nApplying compression algorithms...")
    rule_based_compressed = rule_based_compression(original_trace)
    llm_based_compressed = llm_based_compression(original_trace)
    hybrid_compressed = hybrid_compression(original_trace)

    print_section_header("Rule-Based Compressed Trace")
    print(rule_based_compressed)

    print_section_header("LLM-Based Compressed Trace")
    if llm_based_compressed.startswith("Error:"):
        print(f"NOTE: {llm_based_compressed}")
    else:
        print(llm_based_compressed)

    print_section_header("Hybrid Compressed Trace")
    if hybrid_compressed.startswith("Error:"):
        print(f"NOTE: {hybrid_compressed}")
    else:
        print(hybrid_compressed)

    # --- 4. Show Statistics ---
    original_tokens = get_token_count(original_trace)
    results = {
        "Rule-Based": rule_based_compressed,
        "LLM-Based": llm_based_compressed,
        "Hybrid": hybrid_compressed,
    }

    stats = []
    for name, compressed_trace in results.items():
        token_count = get_token_count(compressed_trace)
        reduction = "N/A"
        if not compressed_trace.startswith("Error:") and original_tokens > 0:
            reduction_pct = ((original_tokens - token_count) / original_tokens) * 100
            reduction = f"{reduction_pct:.2f}%"
        stats.append([name, token_count, reduction])

    print_section_header("Compression Statistics")
    print(f"Original Token Count: {original_tokens}\n")
    
    # Using polars for a clean table print
    stats_df = pl.DataFrame(
        stats,
        schema=["Compression Method", "Token Count", "Reduction (%)"]
    )
    print(stats_df)

if __name__ == "__main__":
    main()
