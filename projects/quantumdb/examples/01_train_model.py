#!/usr/bin/env python3
"""
Example 1: Training a LearnablePQ model.

This script demonstrates how to train a learnable product quantization model
on synthetic data and save it for later use.
"""

import numpy as np
import torch
from pathlib import Path

from quantumdb.training import LearnablePQ, Trainer
from quantumdb.data import SyntheticDataGenerator


def main():
    """Main training script."""
    print("🚀 Starting QuantumDB Model Training")
    print("=" * 50)

    # Configuration
    config = {
        "input_dim": 768,
        "target_dim": 256,
        "n_subvectors": 16,
        "codebook_size": 256,
        "n_samples": 100000,
        "epochs": 50,
        "batch_size": 512,
        "learning_rate": 1e-3,
    }

    print(f"Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()

    # Generate synthetic training data
    print("📊 Generating synthetic training data...")
    data_generator = SyntheticDataGenerator(random_state=42)

    # Generate clustered vectors (more realistic than pure random)
    train_vectors, _ = data_generator.generate_clustered_vectors(
        n_samples=config["n_samples"],
        dimension=config["input_dim"],
        n_clusters=50,
        cluster_std=0.1,
    )

    print(f"Generated {len(train_vectors)} training vectors")
    print(f"Vector dimension: {train_vectors.shape[1]}")
    print()

    # Create model
    print("🧠 Creating LearnablePQ model...")
    model = LearnablePQ(
        input_dim=config["input_dim"],
        target_dim=config["target_dim"],
        n_subvectors=config["n_subvectors"],
        codebook_size=config["codebook_size"],
    )

    print(f"Model compression ratio: {model.get_compression_ratio():.2f}x")
    print(f"Model parameters: {model.get_model_size():,}")
    print()

    # Create trainer
    print("🏋️ Setting up trainer...")
    trainer = Trainer(
        model=model,
        experiment_name="learnablepq_synthetic",
        use_wandb=False,  # Set to True to enable wandb logging
    )

    # Train model
    print("🎯 Starting training...")
    history = trainer.fit(
        train_data=train_vectors,
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        learning_rate=config["learning_rate"],
        save_best=True,
        save_dir="models",
    )

    print(f"✅ Training completed!")
    print(f"Final training loss: {history['train_loss'][-1]:.4f}")
    if "val_loss" in history:
        print(f"Final validation loss: {history['val_loss'][-1]:.4f}")
    print()

    # Test model compression
    print("🔍 Testing model compression...")
    model.eval()
    with torch.no_grad():
        # Sample test vectors
        test_vectors = torch.randn(100, config["input_dim"])

        # Original size
        original_size = test_vectors.numel() * 4  # float32 = 4 bytes

        # Compressed vectors
        encoded = model.encode(test_vectors)
        quantized, codes = model.quantize(encoded)

        # Compressed size (codes only)
        compressed_size = codes.numel() * 1  # int8 = 1 byte

        compression_ratio = original_size / compressed_size

        print(f"Original size: {original_size:,} bytes")
        print(f"Compressed size: {compressed_size:,} bytes")
        print(f"Actual compression ratio: {compression_ratio:.2f}x")
        print()

    # Save final model
    print("💾 Saving final model...")
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    final_model_path = models_dir / "learnablepq_final.safetensors"
    trainer.save_model("models", "learnablepq_final")

    print(f"Model saved to: {final_model_path}")
    print()

    print("🎉 Training pipeline completed successfully!")
    print("Next steps:")
    print("1. Use the trained model with QuantumDB API")
    print("2. Run example 02 to build a vector index")
    print("3. Run example 03 to test search functionality")


if __name__ == "__main__":
    main()
