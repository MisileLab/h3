'''
Downloads and processes datasets for code generation.

- Training set: sentence-transformers/codesearchnet
- Validation and Test sets: SciCode1/SciCode

This script performs the following steps:
1. Downloads both datasets.
2. Splits the SciCode dataset for validation and testing.
3. Applies appropriate transformations to each dataset to create a common format.
4. Filters the data based on content and length.
5. Saves the final train, validation, and test sets to the 'data/' directory.

Usage:
    python scripts/preprocess_data.py
'''
import os
import sys
import re
from datasets import load_dataset, DatasetDict

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.tokenizer import get_tokenizer

def process_codesearchnet(tokenizer):
    '''Loads and processes the training data from codesearchnet.'''
    print("Processing training data from sentence-transformers/codesearchnet...")
    dataset = load_dataset("sentence-transformers/codesearchnet", split='train')

    def transform(example):
        comment = example['comment']
        cleaned_comment = '\n'.join([line for line in comment.split('\n') if not line.strip().startswith('@')])
        cleaned_comment = cleaned_comment.strip()
        return {
            "text": f"Write a function that {cleaned_comment}",
            "target_code": example['code']
        }

    def filter_example(example):
        comment = example['comment']
        code = example['code']
        if not isinstance(comment, str) or not isinstance(code, str) or not comment or not code:
            return False
        stripped_comment = comment.strip().lower()
        if stripped_comment in ('// started', '// completed'):
            return False
        if not any(c.isalnum() for c in comment):
            return False
        # Removed token length limits
        return True

    num_procs = os.cpu_count() or 1
    filtered = dataset.filter(filter_example, num_proc=num_procs)
    transformed = filtered.map(transform, num_proc=num_procs, remove_columns=dataset.column_names)
    print("Finished processing training data.")
    return transformed

def process_scicode(tokenizer):
    '''Loads and processes the validation/test data from SciCode.'''
    print("Processing validation/test data from SciCode1/SciCode...")
    
    validation_dataset = load_dataset("SciCode1/SciCode", split='validation')
    test_dataset = load_dataset("SciCode1/SciCode", split='test')
    
    # --- Validation Set Processing ---
    def transform_validation(example):
        description = example['problem_description_main']
        background = str(example.get('problem_background_main', ''))
        full_description = description
        if background:
            full_description += "\n" + background

        if description.lower().startswith('write a python function'):
            text = full_description
        else:
            text = f"Write a Python function to solve the following problem: {full_description}"
        return {
            "text": text,
            "target_code": example['general_solution']
        }

    def filter_validation(example):
        description = example['text']
        solution = example['target_code']
        if not isinstance(description, str) or not isinstance(solution, str) or not description or not solution:
            return False
        if not any(c.isalnum() for c in description):
            return False
        # Removed token length limits
        return True

    # --- Test Set Processing (FIXED) ---
    def transform_test(example):
        description = example['problem_description_main']
        background = str(example.get('problem_background_main', ''))
        full_description = description
        if background:
            full_description += "\n" + background

        if description.lower().startswith('write a python function'):
            text = full_description
        else:
            text = f"Write a Python function to solve the following problem: {full_description}"
        
        # Convert list of tests to string
        tests = example['general_tests']
        if isinstance(tests, list):
            target_code = '\n'.join(tests)
        else:
            target_code = str(tests)
            
        return {
            "text": text,
            "target_code": target_code
        }

    def filter_test(example):
        description = example['text']
        tests = example['target_code']  # FIXED: Use general_tests instead of general_solution
        
        # Debug: Print first few examples
        global test_debug_count
        if not hasattr(filter_test, 'debug_count'):
            filter_test.debug_count = 0
        
        if filter_test.debug_count < 5:
            print(f"\n--- Test Example {filter_test.debug_count} Debug ---")
            print(f"Description type: {type(description)}")
            print(f"Tests type: {type(tests)}")
            print(f"Description: {str(description)[:200]}..." if description else "None")
            print(f"Tests: {str(tests)[:200]}..." if tests else "None")
            filter_test.debug_count += 1
        
        if not isinstance(description, str) or not isinstance(tests, str) or not description or not tests:
            return False
        if not any(c.isalnum() for c in description):
            return False
        # Additional check: make sure general_tests is not empty/whitespace
        if not tests.strip():
            return False
        # Removed token length limits
        return True

    num_procs = os.cpu_count() or 1
    
    print("Processing validation set...")
    processed_validation = validation_dataset.map(transform_validation, num_proc=num_procs, remove_columns=validation_dataset.column_names).filter(filter_validation, num_proc=num_procs)
    
    print("Processing test set...")
    print(f"Test dataset size before filtering: {len(test_dataset)}")
    processed_test = test_dataset.map(transform_test, num_proc=num_procs, remove_columns=test_dataset.column_names).filter(filter_test, num_proc=num_procs)
    print(f"Test dataset size after filtering: {len(processed_test)}")
    
    print("Finished processing validation/test data.")
    return processed_validation, processed_test

def main():
    '''Main function to run the data preprocessing pipeline.'''
    tokenizer = get_tokenizer()

    train_data = process_codesearchnet(tokenizer)
    validation_data, test_data = process_scicode(tokenizer)

    final_datasets = DatasetDict({
        'train': train_data,
        'validation': validation_data,
        'test': test_data
    })

    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)

    for split, data in final_datasets.items():
        if len(data) == 0:
            print(f"WARNING: The '{split}' split is empty after filtering and will be skipped.")
            continue
        output_path = os.path.join(output_dir, f"{split}.parquet")
        print(f"Saving {split} split to {output_path}...")
        data.to_parquet(output_path)
        print(f"Finished saving {split} split ({len(data)} examples).")

    print("\nPreprocessing complete. All data is saved in the 'data/' directory.")

if __name__ == "__main__":
    main()
