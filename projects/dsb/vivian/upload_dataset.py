from datasets import Dataset, DatasetDict  # pyright: ignore[reportMissingTypeStubs]

from utils import read_cached_avro

df = read_cached_avro("processed.avro")

# Use all data as the train split
train_dataset = Dataset.from_pandas(df.to_pandas())

# Create a DatasetDict with only the train split
dataset_dict = DatasetDict({"train": train_dataset})

_ = dataset_dict.push_to_hub('MisileLab/youtube-bot-comments')  # pyright: ignore[reportUnknownMemberType]
