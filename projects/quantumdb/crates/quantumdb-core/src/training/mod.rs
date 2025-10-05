//! Training module for QuantumDB
//! 
//! Implements training pipeline for neural compression models including
//! loss functions, optimizers, and auto-tuning.

pub mod losses;
pub mod trainer;
pub mod optimizer;
pub mod autotuning;
pub mod data;

pub use losses::*;
pub use trainer::*;
pub use optimizer::*;
pub use autotuning::*;
pub use data::*;

#[derive(Clone, Debug)]
pub struct TrainingConfig {
    pub input_dim: usize,
    pub target_dim: usize,
    pub n_subvectors: usize,
    pub epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub auto_tune: bool,
    pub num_samples: Option<usize>,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            input_dim: 768,
            target_dim: 256,
            n_subvectors: 16,
            epochs: 25,
            batch_size: 512,
            learning_rate: 5e-4,
            auto_tune: false,
            num_samples: None,
        }
    }
}