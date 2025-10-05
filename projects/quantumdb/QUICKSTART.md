# Quick Start Guide

Get started with QuantumDB in 5 minutes!

## Prerequisites

- Python 3.8+
- Docker and Docker Compose (for Qdrant)

## 1. Installation

```bash
# Clone the repository
git clone https://github.com/quantumdb/quantumdb.git
cd quantumdb

# Install dependencies
pip install -e ".[dev,evaluation]"
```

## 2. Start Qdrant

```bash
# Start Qdrant with Docker Compose
docker-compose up -d qdrant

# Verify it's running
curl http://localhost:6333/health
```

## 3. Train a Model

```python
# examples/quickstart_train.py
import numpy as np
from quantumdb.training import LearnablePQ, Trainer
from quantumdb.data import SyntheticDataGenerator

# Generate synthetic data
data_generator = SyntheticDataGenerator(random_state=42)
train_vectors = data_generator.generate_text_like_vectors(
    n_samples=10000, dimension=768
)

# Create and train model
model = LearnablePQ(
    input_dim=768,
    target_dim=256,
    n_subvectors=16,
    codebook_size=256,
)

trainer = Trainer(model, experiment_name="quickstart")
history = trainer.fit(train_vectors, epochs=10, save_best=True)

print(f"Model trained! Compression ratio: {model.get_compression_ratio():.1f}x")
```

## 4. Build Vector Index

```python
# examples/quickstart_index.py
import numpy as np
from quantumdb import QuantumDB
from quantumdb.data import SyntheticDataGenerator

# Initialize database
db = QuantumDB(
    model_path="models/quickstart_best.safetensors",
    collection_name="quickstart_demo",
    vector_size=256,
)

# Generate and add vectors
data_generator = SyntheticDataGenerator(random_state=42)
vectors = data_generator.generate_text_like_vectors(
    n_samples=1000, dimension=768
)

# Add with metadata
ids = [f"doc_{i}" for i in range(1000)]
payloads = [{"title": f"Document {i}", "category": f"cat_{i%5}"} for i in range(1000)]

result = db.add(vectors, ids, payloads)
print(f"Added {result['vectors_added']} vectors")
```

## 5. Search Vectors

```python
# examples/quickstart_search.py
import numpy as np
from quantumdb import QuantumDB
from quantumdb.data import SyntheticDataGenerator

# Connect to database
db = QuantumDB(
    model_path="models/quickstart_best.safetensors",
    collection_name="quickstart_demo",
    vector_size=256,
)

# Generate a query
data_generator = SyntheticDataGenerator(random_state=42)
query_vector = data_generator.generate_text_like_vectors(
    n_samples=1, dimension=768
)[0]

# Search
results = db.search(query_vector, limit=5)

print("Search results:")
for i, (doc_id, score, payload) in enumerate(results):
    print(f"{i+1}. {doc_id}: {score:.4f} - {payload['title']}")

# Search with filter
filtered_results = db.search(
    query_vector,
    limit=5,
    filter_params={
        "must": [{"key": "category", "match": {"value": "cat_1"}}]
    }
)

print(f"\nFiltered results (cat_1): {len(filtered_results)} found")
```

## 6. Run the Examples

```bash
# Train model
python examples/quickstart_train.py

# Build index
python examples/quickstart_index.py

# Test search
python examples/quickstart_search.py
```

## What's Next?

- 📖 Read the [full documentation](README.md)
- 🧪 Try the [comprehensive examples](examples/)
- 📊 Run [benchmarks](examples/03_search.py)
- 🔧 [Customize](README.md#api-reference) for your use case

## Troubleshooting

### Qdrant Connection Issues

```bash
# Check if Qdrant is running
docker-compose ps qdrant

# Restart if needed
docker-compose restart qdrant
```

### Import Errors

```bash
# Install missing dependencies
pip install torch sentence-transformers qdrant-client

# Or install all optional dependencies
pip install -e ".[dev,evaluation]"
```

### Model Not Found

```bash
# Make sure the model file exists
ls -la models/

# If not, run the training script first
python examples/quickstart_train.py
```

## Performance Tips

1. **Batch Size**: Use larger batches for better throughput
2. **Vector Dimension**: Start with 256d compressed vectors
3. **Filtering**: Use metadata filters to reduce search space
4. **Hardware**: Use GPU for training, CPU for inference

## Need Help?

- 📖 [Documentation](README.md)
- 🐛 [Issues](https://github.com/quantumdb/quantumdb/issues)
- 💬 [Discussions](https://github.com/quantumdb/quantumdb/discussions)