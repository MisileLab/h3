dataset = load_dataset("SciCode1/SciCode", split='test')
    
    valid_test_split = dataset.train_test_split(test_size=0.5, seed=42)
    
    def transform(example):
        return {
            "text": f"Write a Python function to solve the following problem: {example['problem_description_main']}",
            "target_code": example['general_solution']
        }

    def filter_example(example):
        description = example.get('problem_description_main', '')
        solution = example.get('general_solution', '')
        if not description or not solution:
            return False
        if not any(c.isalnum() for c in description):
            return False
        code_tokens = tokenizer.encode(solution)
        return 10 <= len(code_tokens) <= 500

    num_procs = os.cpu_count() or 1
    
    validation_set = valid_test_split['train']
    test_set = valid_test_split['test']

    print("Processing validation set...")
    processed_validation = validation_set.filter(filter_example, num_proc=num_procs).map(transform, num_proc=num_procs, remove_columns=validation_set.column_names)
    
    print("Processing test set...")
    processed_test = test_set.filter(filter_example, num_proc=num_procs).map(transform, num_proc=num_procs, remove_columns=test_set.column_names)
    
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
        print(f"Finished saving {split} split.")

    print("\nPreprocessing complete. All data is saved in the 'data/' directory.")

if __name__ == "__main__":
    main()