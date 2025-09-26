'''
PyTorch Dataset class for loading the preprocessed data.
'''
import os
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset

# Define the padding token ID. Using the ID for <|endoftext|> from the o200k_base tokenizer.
PAD_TOKEN_ID = 199999

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
            print(f"Warning: Data file not found at {data_path}. Using a dummy dataset.")
            self.dataset = self._create_dummy_dataset()
        else:
            self.dataset = load_dataset('parquet', data_files=data_path, split='train')

    def _create_dummy_dataset(self):
        from datasets import Dataset as HuggingFaceDataset
        dummy_data = {
            "text": ["Write a Python function that returns 'hello world'", "Create a JavaScript function to add two numbers"],
            "target_code": ["def hello():\n  return 'hello world'", "function add(a, b) {\n  return a + b;\n}"]
        }
        return HuggingFaceDataset.from_dict(dummy_data)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        text = sample['text']
        target_code = sample['target_code']

        input_text = f"<|startoftext|>{text}<|endoftext|><|startofcode|>{target_code}<|endofcode|>"

        tokens = self.tokenizer.encode(input_text, allowed_special={'<|startoftext|>', '<|endoftext|>', '<|startofcode|>', '<|endofcode|>'} )
        tokens = tokens[:self.max_length]

        input_ids = torch.tensor(tokens, dtype=torch.long)
        target_ids = input_ids.clone()
        target_ids[:-1] = input_ids[1:]
        target_ids[-1] = -100

        return {
            "input_ids": input_ids,
            "labels": target_ids
        }

def collate_batch(batch):
    '''
    Pads sequences in a batch to the length of the longest sequence.
    '''
    input_ids_list = [item['input_ids'] for item in batch]
    labels_list = [item['labels'] for item in batch]

    # Pad input_ids with the pad_token_id
    padded_input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=PAD_TOKEN_ID)
    
    # Pad labels with -100, which is ignored by the loss function
    padded_labels = pad_sequence(labels_list, batch_first=True, padding_value=-100)

    return {
        "input_ids": padded_input_ids,
        "labels": padded_labels
    }

if __name__ == '__main__':
    import pandas as pd
    from tokenizer import get_tokenizer
    from torch.utils.data import DataLoader

    tokenizer = get_tokenizer()
    dummy_data_path = "dummy_data.parquet"
    dummy_data = [
        {"text": "Short function", "target_code": "def f(): pass"},
        {"text": "A much longer function description for testing padding", "target_code": "def a_longer_function():\n    print('This is longer')\n    return True"}
    ]
    pd.DataFrame(dummy_data).to_parquet(dummy_data_path)

    dataset = CodeDataset(data_path=dummy_data_path, tokenizer=tokenizer)
    
    # Demonstrate the collate function
    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_batch)
    batch = next(iter(loader))

    print("---")
    print(f"Input IDs shape: {batch['input_ids'].shape}")
    print(f"Labels shape: {batch['labels'].shape}")
    print(f"Input IDs:\n{batch['input_ids']}")

    os.remove(dummy_data_path)
