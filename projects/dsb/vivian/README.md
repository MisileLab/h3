# Vivian

A YouTube bot comment detection system using embeddings and machine learning.

## Versions

- [v0](https://static.marimo.app/static/vivian-jcxs)
- [v1](https://static.marimo.app/static/vivian-44de)

## Overview

Vivian is a tool designed to identify and filter bot comments on YouTube videos. It uses embeddings and machine learning techniques to classify comments as either bot-generated or human-written.

## Features

- YouTube comment collection and processing
- Bot comment classification using transformer-based models
- Interactive UI for model training and evaluation using Marimo notebooks
- Support for Korean text via KcELECTRA model
- Data visualization tools for model performance analysis

## Requirements

- Python 3.13.4 or higher
- YouTube API key for comment collection
- PyTorch (CPU or ROCm version available)

## Installation

1. Clone the repository:
   ```
   git clone https://gith.misile.xyz/h3.git:/projects/dsb/vivian.git
   cd vivian
   ```

2. Set up the environment:
   ```
   uv sync
   ```

3. Set up your YouTube API key as an environment variable:
   ```
   export YOUTUBE_API_KEY=your_api_key_here
   ```

## Usage

### Data Collection

1. Collect YouTube videos:
   ```
   python data/get_videos.py
   ```

2. Collect comments from videos:
   ```
   python data/get_comments.py
   ```

### Classification

1. Run the classification model:
   ```
   python classify.py
   ```

2. Train the model with custom data:
   ```
   python train.py
   ```

### Data Processing

1. Split data into training and test sets:
   ```
   python data/train_test_split.py
   ```

2. Merge processed data:
   ```
   python merge.py
   ```

3. Clear temporary data:
   ```
   python clear.py
   ```

## Project Structure

- `classify.py`: Runs the bot comment classification model
- `clear.py`: Clears temporary OpenAI files
- `data/`: Directory containing data processing scripts
  - `get_comments.py`: Collects comments from YouTube videos
  - `get_videos.py`: Collects YouTube video information
  - `train_test_split.py`: Splits data into training and test sets
- `main.py`: Entry point for running modules
- `merge.py`: Merges processed embedding data
- `train.py`: Trains the bot detection model
- `utils.py`: Utility functions and data models

## Model Architecture

The bot detection system uses a SpamUserClassifier based on the KcELECTRA model with:
- Frozen initial transformer layers
- Custom classification layers with dropout for regularization
- Focal Loss to handle class imbalance
- Combined CLS token and mean pooling for improved performance

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
