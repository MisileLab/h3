from os import environ

from huggingface_hub import snapshot_download

environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

year = int(input("Enter the year: "))
month = int(input("Enter the month: "))

# Download the specific year=2025/month=07 folder
snapshot_download(
  repo_id="Lichess/standard-chess-games",
  repo_type="dataset",
  allow_patterns=[f"data/year={year}/month={month:02d}/*"],
  local_dir="./lichess_data"
)