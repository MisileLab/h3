from datasets import Dataset, DatasetDict  # pyright: ignore[reportMissingTypeStubs]

from utils import read_cached_avro

df = read_cached_avro("processed.avro")

# Shuffle the data first
df = df.sample(fraction=1.0)

# Calculate split indices
n = len(df)
train_end = int(0.7 * n)
test_end = int(0.9 * n)  # 0.7 + 0.2 = 0.9

# Split the data into train, test, and validation datasets
train_data = df.slice(0, train_end)
test_data = df.slice(train_end, test_end - train_end)
val_data = df.slice(test_end)

# Convert to pandas, then to datasets
train_dataset = Dataset.from_pandas(train_data.to_pandas())
test_dataset = Dataset.from_pandas(test_data.to_pandas())
val_dataset = Dataset.from_pandas(val_data.to_pandas())

# Create DatasetDict
dataset_dict = DatasetDict({
  'train': train_dataset,
  'test': test_dataset,
  'validation': val_dataset
})

# Push to hub
_ = dataset_dict.push_to_hub('MisileLab/youtube-bot-comments') # pyright: ignore[reportUnknownMemberType]
