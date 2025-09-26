'''
PyTorch Dataset class for loading the preprocessed data.
'''
import os
import torch
from torch.utils.data import Dataset
from datasets import load_dataset

class CodeDataset(Dataset):
    '''
    A PyTorch Dataset for the preprocessed data, loaded from a Parquet file.
    '''
    def __init__(self, data_path, tokenizer, max_length=2048):
        '''
        Args:
            data_path (str): Path to the preprocessed .parquet file.
            tokenizer: The tokenizer to use for encoding the text.
            max_length (int): Maximum sequence length for the model.
        '''
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        if not os.path.exists(data_path):
            # If the file doesn't exist, create a dummy dataset for demonstration
            print(f"Warning: Data file not found at {data_path}. Using a dummy dataset.")
            self.dataset = self._create_dummy_dataset()
        else:
            # Load the dataset from the Parquet file using the datasets library.
            # This creates a memory-mappable dataset, which is highly efficient.
            self.dataset = load_dataset('parquet', data_files=data_path, split='train')

    def _create_dummy_dataset(self):
        '''Creates a small dummy dataset for testing purposes.'''
        from datasets import Dataset as HuggingFaceDataset
        dummy_data = {
            "text": ["Write a Python function that returns 'hello world'", "Create a JavaScript function to add two numbers"],
            "target_code": ["def hello():\n  return 'hello world'", "function add(a, b) {\n  return a + b;\n}"]
        }
        return HuggingFaceDataset.from_dict(dummy_data)

    def __len__(self):
        '''
        Returns the number of samples in the dataset.
        '''
        return len(self.dataset)

    def __getitem__(self, idx):
        '''
        Retrieves a single sample from the dataset and prepares it for the model.
        '''
        sample = self.dataset[idx]
        text = sample['text']
        target_code = sample['target_code']

        # Format the input text according to the model's training format
        input_text = f"<|startoftext|>{text}<|endoftext|><|startofcode|>{target_code}<|endofcode|>"

        # Tokenize the input text
        tokens = self.tokenizer.encode(input_text, allowed_special={'<|startoftext|>', '<|endoftext|>', '<|startofcode|>', '<|endofcode|>'})

        # Truncate tokens to max_length
        tokens = tokens[:self.max_length]

        # Create input and target tensors
        input_ids = torch.tensor(tokens, dtype=torch.long)
        
        # The target is the input shifted by one token. We use -100 for tokens to be ignored by the loss function.
        target_ids = input_ids.clone()
        target_ids[:-1] = input_ids[1:]
        target_ids[-1] = -100 # The last token has no target

        return {
            "input_ids": input_ids,
            "labels": target_ids
        }

if __name__ == '__main__':
    # This block is for demonstration purposes.
    # You will need to install pandas and pyarrow: pip install pandas pyarrow
    import pandas as pd
    from tokenizer import get_tokenizer

    # Initialize tokenizer
    tokenizer = get_tokenizer()

    # Create a dummy data file for testing
    dummy_data_path = "dummy_data.parquet"
    dummy_data = [
        {"text": "Write a test function", "target_code": "def test(): pass"},
        {"text": "Write another test function", "target_code": "def test2(): return 1"}
    ]
    pd.DataFrame(dummy_data).to_parquet(dummy_data_path)

    # Initialize dataset
    print(f"Loading data from {dummy_data_path}...")
    dataset = CodeDataset(data_path=dummy_data_path, tokenizer=tokenizer)

    # Get a sample
    if len(dataset) > 0:
        sample = dataset[0]
        print("\nSample from CodeDataset:")
        print(f"Input IDs: {sample['input_ids']}")
        print(f"Labels: {sample['labels']}")
        print(f"\nDecoded Input: {tokenizer.decode(sample['input_ids'].tolist())}")
    else:
        print("Dataset is empty.")

    # Clean up the dummy file
    os.remove(dummy_data_path)