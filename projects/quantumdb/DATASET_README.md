# Wikipedia Korean Embeddings Dataset

This directory contains scripts for downloading and analyzing the Wikipedia Korean embeddings dataset from Hugging Face.

## Dataset Information

- **Source**: [chaehoyu/wikipedia-22-12-ko-embeddings-100k](https://huggingface.co/datasets/chaehoyu/wikipedia-22-12-ko-embeddings-100k)
- **Language**: Korean
- **Size**: 100,000 Wikipedia articles
- **Embedding Model**: [jhgan/ko-sbert-nli](https://huggingface.co/jhgan/ko-sbert-nli)
- **Embedding Dimension**: 768
- **Format**: Parquet file with JSON-encoded embeddings

## Data Structure

Each entry contains:
- `title`: Wikipedia article title (string)
- `text`: Article content (string)  
- `embedding_json`: 768-dimensional embedding encoded as JSON string

## Setup

1. Install required dependencies:
```bash
uv add huggingface_hub pandas pyarrow numpy scikit-learn matplotlib seaborn
```

2. Download the dataset:
```bash
uv run python simple_download.py
```

## Usage

### Basic Comparison Script

The main comparison script is `compare_embeddings.py`. Here are the available operations:

#### 1. Get Dataset Statistics
```bash
uv run python compare_embeddings.py --operation stats
```

#### 2. Find Similar Items
```bash
# Find items similar to index 0
uv run python compare_embeddings.py --operation similar --index 0 --top_k 5
```

#### 3. Search by Text
```bash
# Search for articles containing '문화방송'
uv run python compare_embeddings.py --operation search --query "문화방송" --top_k 5
```

#### 4. Compare Two Items
```bash
# Compare items at indices 0 and 1
uv run python compare_embeddings.py --operation compare --index 0 --index2 1
```

#### 5. Find Most Similar Pairs
```bash
# Find top 5 most similar pairs
uv run python compare_embeddings.py --operation pairs --top_k 5
```

#### 6. Visualize Embeddings (PCA)
```bash
# Create PCA visualization with 1000 samples
uv run python compare_embeddings.py --operation visualize --sample_size 1000
```

#### 7. Cluster Analysis
```bash
# Analyze 5 clusters with 1000 samples
uv run python compare_embeddings.py --operation clusters --n_clusters 5 --sample_size 1000
```

### Example Usage

Run the complete example:
```bash
uv run python example_usage.py
```

This will demonstrate all the main features:
- Dataset statistics
- Similarity search
- Text search
- Item comparison
- Most similar pairs
- PCA visualization
- Cluster analysis

## Python API

You can also use the `EmbeddingComparator` class directly in your Python code:

```python
from compare_embeddings import EmbeddingComparator

# Initialize comparator
comparator = EmbeddingComparator("data/wikipedia_ko_embeddings/wikipedia-22-12-ko-embeddings-100k.parquet")

# Find similar items
similar = comparator.find_similar_items(query_idx=0, top_k=5)

# Search by text
results = comparator.search_by_text("한국어", top_k=5)

# Get statistics
stats = comparator.get_embedding_stats()

# Compare two items
comparison = comparator.compare_two_items(0, 1)

# Visualize embeddings
comparator.visualize_embeddings_pca(sample_size=1000)

# Analyze clusters
comparator.analyze_embedding_clusters(n_clusters=5, sample_size=1000)
```

## Output Files

- `embeddings_pca.png`: PCA visualization of embeddings (if visualization is run)
- Dataset is saved as `data/wikipedia_ko_embeddings/wikipedia_embeddings.csv` for easy access

## Memory Considerations

The dataset contains 100,000 embeddings with 768 dimensions each. For memory-intensive operations:
- Similarity calculations use sampling (default 1000 items) to avoid O(n²) complexity
- PCA and clustering operations also use sampling by default
- You can adjust sample sizes using the `--sample_size` parameter

## Dependencies

- `huggingface_hub`: For downloading the dataset
- `pandas`: For data manipulation
- `pyarrow`: For reading parquet files
- `numpy`: For numerical operations
- `scikit-learn`: For similarity calculations, PCA, and clustering
- `matplotlib` & `seaborn`: For visualization

## Troubleshooting

If you encounter import errors, make sure all dependencies are installed:
```bash
uv add huggingface_hub pandas pyarrow numpy scikit-learn matplotlib seaborn
```

If the dataset file is not found, make sure you've downloaded it first:
```bash
uv run python simple_download.py
```