"""Module for downloading data from Lichess database."""

import os
import re
import json
import logging
import tempfile
import urllib.request
from pathlib import Path
from typing import Optional, List, Dict, Any, Union, Tuple
from datetime import datetime, timedelta

import polars as pl
from tqdm import tqdm

# Configure logging
logging.basicConfig(
  level=logging.INFO,
  format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Lichess database URL
LICHESS_DB_URL = "https://database.lichess.org"

# Data types available from Lichess
LICHESS_DATA_TYPES = {
  "games": "Chess games in PGN format",
  "puzzles": "Chess puzzles in CSV format",
  "evaluations": "Chess position evaluations in JSONL format",
}


class LichessDownloader:
  """Downloader for Lichess database files."""

  def __init__(self, data_dir: Optional[Union[str, Path]] = None) -> None:
    """Initialize the downloader.

    Args:
      data_dir: Directory to store downloaded files. If None, uses a temporary directory.
    """
    if data_dir is None:
      self.data_dir = Path(tempfile.gettempdir()) / "adela_lichess_data"
    else:
      self.data_dir = Path(data_dir)
    
    # Create data directory if it doesn't exist
    self.data_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories for each data type
    for data_type in LICHESS_DATA_TYPES:
      (self.data_dir / data_type).mkdir(exist_ok=True)
    
    logger.info(f"Data will be stored in {self.data_dir}")
  
  def _download_file(
    self, 
    url: str, 
    output_path: Path,
    show_progress: bool = True
  ) -> Path:
    """Download a file from a URL.

    Args:
      url: URL to download.
      output_path: Path to save the file.
      show_progress: Whether to show a progress bar.

    Returns:
      Path to the downloaded file.
    """
    # Create parent directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if file already exists
    if output_path.exists():
      logger.info(f"File already exists: {output_path}")
      return output_path
    
    logger.info(f"Downloading {url} to {output_path}")
    
    # Download with progress bar
    with urllib.request.urlopen(url) as response:
      file_size = int(response.headers.get("Content-Length", 0))
      
      with open(output_path, "wb") as out_file:
        if show_progress and file_size > 0:
          with tqdm(total=file_size, unit="B", unit_scale=True, desc=output_path.name) as pbar:
            while True:
              buffer = response.read(8192)
              if not buffer:
                break
              out_file.write(buffer)
              pbar.update(len(buffer))
        else:
          # Simple download without progress bar
          out_file.write(response.read())
    
    logger.info(f"Downloaded {url} to {output_path}")
    return output_path
  
  def list_available_files(self, data_type: str) -> List[Dict[str, Any]]:
    """List available files for a data type.

    Args:
      data_type: Type of data to list ("games", "puzzles", "evaluations").

    Returns:
      List of dictionaries with file information.
    """
    if data_type not in LICHESS_DATA_TYPES:
      raise ValueError(f"Invalid data type: {data_type}. Must be one of {list(LICHESS_DATA_TYPES.keys())}")
    
    # Fetch the Lichess database page
    with urllib.request.urlopen(LICHESS_DB_URL) as response:
      html = response.read().decode("utf-8")
    
    # Extract file information based on data type
    if data_type == "games":
      # Find game files (standard chess only)
      pattern = r'href="(lichess_db_standard_rated_(\d{4}-\d{2})\.pgn\.zst)".*?(\d+\.\d+\s+[GMK]B).*?(\d+,\d+,\d+)'
      matches = re.findall(pattern, html)
      
      return [
        {
          "filename": match[0],
          "date": match[1],
          "size": match[2].strip(),
          "games": match[3].replace(",", ""),
          "url": f"{LICHESS_DB_URL}/{match[0]}",
        }
        for match in matches
      ]
    
    elif data_type == "puzzles":
      # Find puzzle file
      pattern = r'href="(lichess_db_puzzle\.csv\.zst)".*?(\d+\.\d+\s+[GMK]B).*?(\d+,\d+,\d+)'
      matches = re.findall(pattern, html)
      
      if matches:
        match = matches[0]
        return [{
          "filename": match[0],
          "size": match[1].strip(),
          "puzzles": match[2].replace(",", ""),
          "url": f"{LICHESS_DB_URL}/{match[0]}",
        }]
      return []
    
    elif data_type == "evaluations":
      # Find evaluation file
      pattern = r'href="(lichess_db_eval\.jsonl\.zst)".*?(\d+\.\d+\s+[GMK]B)'
      matches = re.findall(pattern, html)
      
      if matches:
        match = matches[0]
        return [{
          "filename": match[0],
          "size": match[1].strip(),
          "url": f"{LICHESS_DB_URL}/{match[0]}",
        }]
      return []
    
    return []
  
  def download_latest_data(
    self, 
    data_type: str, 
    num_months: int = 1
  ) -> List[Path]:
    """Download the latest data files.

    Args:
      data_type: Type of data to download ("games", "puzzles", "evaluations").
      num_months: Number of months to download for games (ignored for other data types).

    Returns:
      List of paths to downloaded files.
    """
    if data_type not in LICHESS_DATA_TYPES:
      raise ValueError(f"Invalid data type: {data_type}. Must be one of {list(LICHESS_DATA_TYPES.keys())}")
    
    available_files = self.list_available_files(data_type)
    downloaded_files = []
    
    if data_type == "games":
      # Download the latest N months of games
      files_to_download = sorted(
        available_files, 
        key=lambda x: x["date"], 
        reverse=True
      )[:num_months]
      
      for file_info in files_to_download:
        output_path = self.data_dir / data_type / file_info["filename"]
        downloaded_file = self._download_file(file_info["url"], output_path)
        downloaded_files.append(downloaded_file)
    
    else:
      # For puzzles and evaluations, there's only one file
      if available_files:
        file_info = available_files[0]
        output_path = self.data_dir / data_type / file_info["filename"]
        downloaded_file = self._download_file(file_info["url"], output_path)
        downloaded_files.append(downloaded_file)
    
    return downloaded_files
  
  def download_games_by_date_range(
    self, 
    start_date: Union[str, datetime],
    end_date: Optional[Union[str, datetime]] = None
  ) -> List[Path]:
    """Download games within a date range.

    Args:
      start_date: Start date (YYYY-MM format or datetime object).
      end_date: End date (YYYY-MM format or datetime object). If None, uses start_date.

    Returns:
      List of paths to downloaded files.
    """
    # Convert string dates to datetime objects
    if isinstance(start_date, str):
      start_date = datetime.strptime(start_date, "%Y-%m")
    
    if end_date is None:
      end_date = start_date
    elif isinstance(end_date, str):
      end_date = datetime.strptime(end_date, "%Y-%m")
    
    # List available game files
    available_files = self.list_available_files("games")
    downloaded_files = []
    
    for file_info in available_files:
      file_date = datetime.strptime(file_info["date"], "%Y-%m")
      
      if start_date <= file_date <= end_date:
        output_path = self.data_dir / "games" / file_info["filename"]
        downloaded_file = self._download_file(file_info["url"], output_path)
        downloaded_files.append(downloaded_file)
    
    return downloaded_files
