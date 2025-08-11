from os import environ
from pathlib import Path

from huggingface_hub import snapshot_download
from tqdm import tqdm

environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

year = int(input("Enter the year: "))

snapshot_download(
  repo_id="Lichess/standard-chess-games",
  repo_type="dataset",
  allow_patterns=[f"data/year={year}/*"],
  local_dir="./lichess_data"
)

for i in tqdm(Path("./lichess_data/data").glob(f"year={year}/*")):
  if i.is_dir():
    for j in i.glob("*.parquet"):
      j.rename(i.parent / f"{i.name}_{j.name}")
