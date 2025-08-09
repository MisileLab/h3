from examples.train_model import train_from_parquet

# Train directly from Hugging Face dataset (defaults to 'train' split)
train_from_parquet(
  parquet_path="misilelab/adela-dataset",
  output_dir="models/adela",
  num_epochs=50,
  batch_size=256,
  validation_split=0.1,
  min_elo=1000,
  parse_chunk_size=1000,
  early_stop_patience=5,
  early_stop_min_delta=1e-3,
  device="cuda",
)
