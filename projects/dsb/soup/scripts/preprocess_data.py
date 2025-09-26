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
        description = example['problem_description_main']
        solution = example['general_solution']
        if not isinstance(description, str) or not isinstance(solution, str) or not description or not solution:
            return False
        if not any(c.isalnum() for c in description):
            return False
        code_tokens = tokenizer.encode(solution)
        return 5 <= len(code_tokens) <= 1024

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
        return {
            "text": text,
            "target_code": example['general_tests']
        }

    def filter_test(example):
        description = example['problem_description_main']
        tests = example['general_tests']  # Using general_tests consistently
        if not isinstance(description, str) or not isinstance(tests, str) or not description or not tests:
            return False
        if not any(c.isalnum() for c in description):
            return False
        code_tokens = tokenizer.encode(tests)
        return 5 <= len(code_tokens) <= 1024

    num_procs = os.cpu_count() or 1
    
    print("Processing validation set...")
    processed_validation = validation_dataset.filter(filter_validation, num_proc=num_procs).map(transform_validation, num_proc=num_procs, remove_columns=validation_dataset.column_names)
    
    print("Processing test set...")
    processed_test = test_dataset.filter(filter_test, num_proc=num_procs).map(transform_test, num_proc=num_procs, remove_columns=test_dataset.column_names)
    
    print("Finished processing validation/test data.")
    return processed_validation, processed_test
