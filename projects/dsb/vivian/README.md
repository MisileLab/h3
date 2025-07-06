# Vivian

A YouTube bot comment detection system using embeddings and machine learning.

## Versions

- [v0](https://static.marimo.app/static/vivian-jcxs)
- [v1](https://static.marimo.app/static/vivian-44de)

## Overview

Vivian is a tool designed to identify and filter bot comments on YouTube videos. It uses embeddings and machine learning techniques to classify comments as either bot-generated or human-written.

## Features

- YouTube comment collection and processing
- Embedding generation for comment analysis
- Bot comment classification using machine learning
- Interactive UI for manual comment classification
- Support for multiple AI models (Gemini, Ollama, etc.)
- Data visualization tools

## Requirements

- Python 3.13.4 or higher
- YouTube API key for comment collection

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
   python get_videos.py
   ```

2. Collect comments from videos:
   ```
   python get_comments.py
   ```

### Data Processing

1. Remove duplicate comments:
   ```
   python remove_duplicate.py
   ```

2. Prepare data for embedding:
   ```
   python prepare_embedding.py
   ```

3. Generate embeddings:
   ```
   python embedding.py
   ```

### Classification

1. Manual classification using the UI:
   ```
   python classify.py
   ```

2. Using Gemini for classification:
   ```
   python classify_gemini.py
   ```

3. Using Ollama for classification:
   ```
   python run_ollama.py
   ```

### Data Analysis

1. Split data into training and test sets:
   ```
   python train_test_split.py
   ```

2. Visualize the data:
   ```
   python visualize.py
   ```

## Project Structure

- `batch.py`: Batch processing utilities
- `build_batch.py`: Builds batches for processing
- `classify.py`: UI for manual comment classification
- `classify_gemini.py`: Classification using Google's Gemini model
- `clear.py`: Clears temporary data
- `embedding.py`: Generates embeddings for comments
- `get_comments.py`: Collects comments from YouTube videos
- `get_videos.py`: Collects YouTube video information
- `merge.py`: Merges processed data
- `prepare_embedding.py`: Prepares data for embedding generation
- `remove_duplicate.py`: Removes duplicate comments
- `run_ollama.py`: Runs classification using Ollama
- `train_test_split.py`: Splits data into training and test sets
- `utils.py`: Utility functions
- `visualize.py`: Data visualization tools

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
