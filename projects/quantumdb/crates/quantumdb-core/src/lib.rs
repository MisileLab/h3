#![feature(portable_simd)]
//! QuantumDB Core Library
//! 
//! Neural compression-powered vector database with 192x compression ratio.
//! 
//! # Features
//! 
//! - **Neural Compression**: Learnable Product Quantization with end-to-end training
//! - **HNSW Index**: Lock-free concurrent approximate nearest neighbor search
//! - **SIMD Optimization**: Vectorized distance computation with portable-simd
//! - **Memory Efficiency**: Zero-copy SafeTensors and memory-mapped storage
//! - **Cross-platform**: CPU, GPU (CUDA/ROCm/Metal), and WebGPU support via Burn

pub mod models;
pub mod training;
pub mod data;
pub mod index;
pub mod storage;
pub mod utils;

// Re-export main types
pub use models::{QuantumCompressor, NeuralEncoder, LearnablePQ};
pub use index::{HNSWGraph, SearchConfig};
pub use storage::{QuantumDB};
pub use training::{Trainer, TrainingConfig};
pub use utils::{SIMDDistance, Metrics};

use thiserror::Error;

/// Main error type for QuantumDB
#[derive(Error, Debug)]
pub enum QuantumDBError {
    #[error("Model error: {0}")]
    Model(#[from] anyhow::Error),
    
    #[error("Index error: {0}")]
    Index(String),
    
    #[error("Storage error: {0}")]
    Storage(#[from] std::io::Error),
    
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    
    #[error("SafeTensors error: {0}")]
    SafeTensors(String),
    
    #[error("Invalid configuration: {0}")]
    Config(String),
}

pub type Result<T> = std::result::Result<T, QuantumDBError>;

/// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const NAME: &str = env!("CARGO_PKG_NAME");

/// Default configuration constants
pub mod config {
    pub const DEFAULT_DIM: usize = 768;
    pub const DEFAULT_TARGET_DIM: usize = 256;
    pub const DEFAULT_N_SUBVECTORS: usize = 16;
    pub const DEFAULT_CODEBOOK_SIZE: usize = 256;
    pub const DEFAULT_COMPRESSION_RATIO: usize = 192; // 3072 / 16
    
    pub const DEFAULT_HNSW_M: usize = 16;
    pub const DEFAULT_HNSW_EF_CONSTRUCT: usize = 200;
    pub const DEFAULT_HNSW_EF_SEARCH: usize = 100;
    
    pub const DEFAULT_BATCH_SIZE: usize = 512;
    pub const DEFAULT_LEARNING_RATE: f32 = 5e-4;
    pub const DEFAULT_EPOCHS: usize = 25;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
        assert_eq!(NAME, "quantumdb-core");
    }

    #[test]
    fn test_config_constants() {
        assert_eq!(config::DEFAULT_DIM, 768);
        assert_eq!(config::DEFAULT_COMPRESSION_RATIO, 192);
    }
}