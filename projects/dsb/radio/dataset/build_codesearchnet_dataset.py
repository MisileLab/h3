
import os
import json
import numpy as np
from datasets import load_dataset
from argdantic import ArgParser
from pydantic import BaseModel
from tqdm import tqdm

from common import PuzzleDatasetMetadata

cli = ArgParser()

class DataProcessConfig(BaseModel):
    output_dir: str = "data/codesearchnet"
    max_seq_len: int = 512
    subsample_size: int = 10000

@cli.command(singleton=True)
def preprocess_data(config: DataProcessConfig):
    # Load the dataset from Hugging Face
    dataset = load_dataset("sentence-transformers/codesearchnet", split="train")

    # Subsample the dataset
    if config.subsample_size:
        dataset = dataset.shuffle(seed=42).select(range(config.subsample_size))

    # Create a vocabulary
    vocab = {"<pad>": 0, "<unk>": 1}
    for example in tqdm(dataset, desc="Building vocab"):
        for char in example["comment"]:
            if char not in vocab:
                vocab[char] = len(vocab)
        for char in example["code"]:
            if char not in vocab:
                vocab[char] = len(vocab)

    # Process the dataset
    inputs = []
    labels = []
    for example in tqdm(dataset, desc="Processing dataset"):
        # Tokenize the comment and code
        input_tokens = [vocab.get(char, vocab["<unk>"]) for char in example["comment"]]
        label_tokens = [vocab.get(char, vocab["<unk>"]) for char in example["code"]]

        # Pad the sequences
        input_tokens = input_tokens[:config.max_seq_len] + [vocab["<pad>"]] * (config.max_seq_len - len(input_tokens))
        label_tokens = label_tokens[:config.max_seq_len] + [vocab["<pad>"]] * (config.max_seq_len - len(label_tokens))

        inputs.append(input_tokens)
        labels.append(label_tokens)

    # Convert to NumPy arrays
    inputs = np.array(inputs, dtype=np.int32)
    labels = np.array(labels, dtype=np.int32)

    # Create the output directory
    os.makedirs(config.output_dir, exist_ok=True)
    train_dir = os.path.join(config.output_dir, "train")
    os.makedirs(train_dir, exist_ok=True)
    test_dir = os.path.join(config.output_dir, "test")
    os.makedirs(test_dir, exist_ok=True)


    # Save the data
    np.save(os.path.join(train_dir, "all__inputs.npy"), inputs)
    np.save(os.path.join(train_dir, "all__labels.npy"), labels)
    np.save(os.path.join(train_dir, "all__group_indices.npy"), np.arange(len(inputs) + 1, dtype=np.int32))
    np.save(os.path.join(train_dir, "all__puzzle_indices.npy"), np.arange(len(inputs) + 1, dtype=np.int32))
    np.save(os.path.join(train_dir, "all__puzzle_identifiers.npy"), np.zeros(len(inputs), dtype=np.int32))
    
    # Create dummy test set
    np.save(os.path.join(test_dir, "all__inputs.npy"), inputs[:1])
    np.save(os.path.join(test_dir, "all__labels.npy"), labels[:1])
    np.save(os.path.join(test_dir, "all__group_indices.npy"), np.arange(2, dtype=np.int32))
    np.save(os.path.join(test_dir, "all__puzzle_indices.npy"), np.arange(2, dtype=np.int32))
    np.save(os.path.join(test_dir, "all__puzzle_identifiers.npy"), np.zeros(1, dtype=np.int32))


    # Create the metadata
    metadata = PuzzleDatasetMetadata(
        seq_len=config.max_seq_len,
        vocab_size=len(vocab),
        pad_id=vocab["<pad>"],
        ignore_label_id=vocab["<pad>"],
        blank_identifier_id=0,
        num_puzzle_identifiers=1,
        total_groups=len(inputs),
        mean_puzzle_examples=1,
        sets=["all"]
    )

    # Save the metadata
    with open(os.path.join(train_dir, "dataset.json"), "w") as f:
        json.dump(metadata.model_dump(), f)
    with open(os.path.join(test_dir, "dataset.json"), "w") as f:
        json.dump(metadata.model_dump(), f)
        
    # Save vocab
    with open(os.path.join(config.output_dir, "vocab.json"), "w") as f:
        json.dump(vocab, f)


if __name__ == "__main__":
    cli()
