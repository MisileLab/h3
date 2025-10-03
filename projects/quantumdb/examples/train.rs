//! Training example for QuantumDB
//! 
//! Demonstrates how to train a neural compression model.

use quantumdb_core::{
    training::{Trainer, TrainingConfig},
    models::QuantumCompressorConfig,
    backend::NdArray,
};
use std::path::PathBuf;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt::init();
    
    println!("🚀 QuantumDB Training Example");
    
    // Configuration
    let config = TrainingConfig {
        input_dim: 768,
        target_dim: 256,
        n_subvectors: 16,
        epochs: 10,
        batch_size: 256,
        learning_rate: 5e-4,
        auto_tune: false, // Disable for quick example
        ..Default::default()
    };
    
    // Initialize trainer
    let device = NdArray::Device::default();
    let mut trainer = Trainer::new(config, device);
    
    // Generate synthetic training data
    println!("📊 Generating synthetic training data...");
    let train_data = generate_synthetic_data(1000, 768)?;
    println!("✅ Generated {} training samples", train_data.len());
    
    // Train model
    println!("🏃‍♂️ Starting training...");
    let model = trainer.train(train_data).await?;
    
    // Save model
    let output_path = PathBuf::from("examples/model.safetensors");
    println!("💾 Saving model to {:?}", output_path);
    trainer.save_model(&model, &output_path)?;
    
    println!("🎉 Training completed successfully!");
    
    // Test compression
    println!("🧪 Testing compression...");
    let test_embedding = generate_synthetic_data(1, 768)?.remove(0);
    let device = NdArray::Device::default();
    let test_tensor = quantumdb_core::tensor::Tensor::from_data(
        quantumdb_core::tensor::TensorData::new(test_embedding, quantumdb_core::tensor::Shape::new([768])),
        &device,
    );
    
    let compressed = model.compress(test_tensor.unsqueeze::<2>(0));
    println!("📐 Compressed shape: {:?}", compressed.dims());
    println!("🗜️  Compression ratio: {}x", model.compression_ratio());
    
    Ok(())
}

/// Generate synthetic training data
fn generate_synthetic_data(num_samples: usize, dim: usize) -> Result<Vec<Vec<f32>>, Box<dyn std::error::Error>> {
    let mut data = Vec::with_capacity(num_samples);
    
    for _ in 0..num_samples {
        let mut embedding = Vec::with_capacity(dim);
        for i in 0..dim {
            // Generate some structured data with patterns
            let value = if i % 100 == 0 {
                // Some dimensions are more important
                (i as f32).sin() * 2.0
            } else {
                // Random noise
                rand::random::<f32>() * 2.0 - 1.0
            };
            embedding.push(value);
        }
        data.push(embedding);
    }
    
    Ok(data)
}