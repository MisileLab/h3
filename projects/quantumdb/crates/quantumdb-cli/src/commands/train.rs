//! Training command implementation

use quantumdb_core::{
    training::{Trainer, TrainingConfig},
    backend::NdArray,
};
use anyhow::Result;

/// Train a new compression model
pub async fn train_command(config: TrainConfig) -> Result<()> {
    println!("🚀 Starting QuantumDB training...");
    
    // Create training configuration
    let training_config = TrainingConfig {
        input_dim: config.input_dim,
        target_dim: config.target_dim,
        n_subvectors: config.n_subvectors,
        epochs: config.epochs,
        batch_size: config.batch_size,
        learning_rate: config.learning_rate,
        auto_tune: config.auto_tune,
        ..Default::default()
    };
    
    // Initialize trainer
    let device = NdArray::Device::default();
    let mut trainer = Trainer::new(training_config, device);
    
    // Load data
    println!("📂 Loading training data...");
    let train_data = trainer.load_data(&config.data_path)?;
    println!("✅ Loaded {} training samples", train_data.len());
    
    // Train model
    println!("🏃‍♂️ Starting training...");
    let model = trainer.train(train_data).await?;
    
    // Save model
    println!("💾 Saving model to {}", config.output_path);
    trainer.save_model(&model, &config.output_path)?;
    
    println!("🎉 Training completed successfully!");
    Ok(())
}

/// Training configuration
pub struct TrainConfig {
    pub data_path: String,
    pub output_path: String,
    pub input_dim: usize,
    pub target_dim: usize,
    pub n_subvectors: usize,
    pub epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f32,
    pub auto_tune: bool,
    pub gpu: Option<usize>,
}