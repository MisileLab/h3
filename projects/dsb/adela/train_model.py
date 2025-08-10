from examples.train_model import train_from_local_data

from sys import argv

# Train from local data folder (expects train/validation/test subfolders)
train_from_local_data(
  data_path=argv[1],  # Local folder with train/validation/test subfolders
  output_dir="models/adela",
  num_epochs=int(argv[2] if len(argv) > 2 else 50),
  batch_size=int(argv[3] if len(argv) > 3 else 256),
  min_elo=1000,
  early_stop_patience=5,
  early_stop_min_delta=1e-3,
)
