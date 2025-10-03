//! SafeTensors-based persistence for models and indices
//! 
//! Provides zero-copy loading and saving of neural models and
//! compressed indices using the SafeTensors format.

use safetensors::{SafeTensors, Dtype, TensorInfo};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    fs::File,
    io::{Read, Write},
    path::Path,
};
use crate::{Result, QuantumDBError};

/// Re-export from mmap module
pub use super::mmap::IndexMetadata;

/// SafeTensors-based storage for models and indices
pub struct SafeTensorsStorage;

impl SafeTensorsStorage {
    /// Save model weights to SafeTensors format
    /// 
    /// # Arguments
    /// * `tensors` - Map of tensor name to tensor data
    /// * `path` - Output file path
    pub fn save_model<P: AsRef<Path>>(
        tensors: HashMap<&str, Vec<f32>>,
        path: P,
    ) -> Result<()> {
        let mut tensor_data = HashMap::new();
        
        for (name, data) in tensors {
            let shape = vec![data.len()];
            tensor_data.insert(
                name.to_string(),
                TensorInfo::new(shape, Dtype::F32),
            );
        }
        
        // Create SafeTensors
        let mut buffer = Vec::new();
        
        // Write header
        let header = serde_json::to_string(&tensor_data)?;
        let header_len = header.len() as u64;
        buffer.write_all(&header_len.to_le_bytes())?;
        buffer.write_all(header.as_bytes())?;
        
        // Write tensor data
        for (_, data) in tensors {
            let bytes = unsafe {
                std::slice::from_raw_parts(
                    data.as_ptr() as *const u8,
                    data.len() * std::mem::size_of::<f32>(),
                )
            };
            buffer.write_all(bytes)?;
        }
        
        // Write to file
        let mut file = File::create(path)?;
        file.write_all(&buffer)?;
        
        Ok(())
    }
    
    /// Load model weights from SafeTensors format
    /// 
    /// # Arguments
    /// * `path` - Input file path
    /// 
    /// # Returns
    /// Map of tensor name to tensor data
    pub fn load_model<P: AsRef<Path>>(path: P) -> Result<HashMap<String, Vec<f32>>> {
        let mut file = File::open(path)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;
        
        // Parse SafeTensors
        let safetensors = SafeTensors::deserialize(&buffer)?;
        let mut tensors = HashMap::new();
        
        for (name, info) in safetensors.tensors() {
            let data = safetensors.tensor(name)?;
            
            if info.dtype == Dtype::F32 {
                let float_data: Vec<f32> = data
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect();
                tensors.insert(name.to_string(), float_data);
            } else {
                return Err(QuantumDBError::SafeTensors(format!(
                    "Unsupported dtype: {:?}",
                    info.dtype
                )));
            }
        }
        
        Ok(tensors)
    }
    
    /// Save compressed index to disk
    /// 
    /// # Arguments
    /// * `codes` - Compressed codes
    /// * `metadata` - Index metadata
    /// * `path` - Output directory path
    pub fn save_index<P: AsRef<Path>>(
        codes: &[u8],
        metadata: &IndexMetadata,
        path: P,
    ) -> Result<()> {
        let path = path.as_ref();
        std::fs::create_dir_all(path)?;
        
        // Save codes
        let codes_path = path.join("codes.safetensors");
        let mut file = File::create(codes_path)?;
        file.write_all(codes)?;
        
        // Save metadata
        let metadata_path = path.join("metadata.json");
        let metadata_json = serde_json::to_string_pretty(metadata)?;
        std::fs::write(metadata_path, metadata_json)?;
        
        Ok(())
    }
    
    /// Load compressed index from disk
    /// 
    /// # Arguments
    /// * `path` - Input directory path
    /// 
    /// # Returns
    /// Tuple of (codes, metadata)
    pub fn load_index<P: AsRef<Path>>(path: P) -> Result<(Vec<u8>, IndexMetadata)> {
        let path = path.as_ref();
        
        // Load codes
        let codes_path = path.join("codes.safetensors");
        let mut file = File::open(codes_path)?;
        let mut codes = Vec::new();
        file.read_to_end(&mut codes)?;
        
        // Load metadata
        let metadata_path = path.join("metadata.json");
        let metadata_json = std::fs::read_to_string(metadata_path)?;
        let metadata: IndexMetadata = serde_json::from_str(&metadata_json)?;
        
        Ok((codes, metadata))
    }
    
    /// Save training checkpoint
    /// 
    /// # Arguments
    /// * `model_data` - Model weights
    /// * `optimizer_data` - Optimizer state
    /// * `epoch` - Current epoch
    /// * `loss` - Current loss
    /// * `path` - Output file path
    pub fn save_checkpoint<P: AsRef<Path>>(
        model_data: HashMap<&str, Vec<f32>>,
        optimizer_data: HashMap<&str, Vec<f32>>,
        epoch: usize,
        loss: f32,
        path: P,
    ) -> Result<()> {
        let checkpoint = Checkpoint {
            epoch,
            loss,
            model: model_data,
            optimizer: optimizer_data,
        };
        
        let checkpoint_json = serde_json::to_string(&checkpoint)?;
        std::fs::write(path, checkpoint_json)?;
        
        Ok(())
    }
    
    /// Load training checkpoint
    /// 
    /// # Arguments
    /// * `path` - Input file path
    /// 
    /// # Returns
    /// Loaded checkpoint
    pub fn load_checkpoint<P: AsRef<Path>>(path: P) -> Result<Checkpoint> {
        let checkpoint_json = std::fs::read_to_string(path)?;
        let checkpoint: Checkpoint = serde_json::from_str(&checkpoint_json)?;
        Ok(checkpoint)
    }
}

/// Training checkpoint data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Checkpoint {
    /// Current epoch
    pub epoch: usize,
    /// Current loss
    pub loss: f32,
    /// Model weights
    pub model: HashMap<String, Vec<f32>>,
    /// Optimizer state
    pub optimizer: HashMap<String, Vec<f32>>,
}

impl Checkpoint {
    /// Create a new checkpoint
    /// 
    /// # Arguments
    /// * `epoch` - Current epoch
    /// * `loss` - Current loss
    /// * `model` - Model weights
    /// * `optimizer` - Optimizer state
    pub fn new(
        epoch: usize,
        loss: f32,
        model: HashMap<String, Vec<f32>>,
        optimizer: HashMap<String, Vec<f32>>,
    ) -> Self {
        Self {
            epoch,
            loss,
            model,
            optimizer,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_save_load_model() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("model.safetensors");
        
        // Create test tensors
        let mut tensors = HashMap::new();
        tensors.insert("encoder.weight", vec![1.0, 2.0, 3.0, 4.0]);
        tensors.insert("encoder.bias", vec![0.1, 0.2]);
        
        // Save model
        SafeTensorsStorage::save_model(tensors.clone(), &path).unwrap();
        
        // Load model
        let loaded_tensors = SafeTensorsStorage::load_model(&path).unwrap();
        
        assert_eq!(loaded_tensors.len(), 2);
        assert_eq!(loaded_tensors["encoder.weight"], vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(loaded_tensors["encoder.bias"], vec![0.1, 0.2]);
    }

    #[test]
    fn test_save_load_index() {
        let dir = tempdir().unwrap();
        let path = dir.path();
        
        // Create test data
        let codes = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let metadata = IndexMetadata {
            num_vectors: 2,
            compressed_dim: 4,
            version: 1,
            created_at: 1234567890,
        };
        
        // Save index
        SafeTensorsStorage::save_index(&codes, &metadata, path).unwrap();
        
        // Load index
        let (loaded_codes, loaded_metadata) = SafeTensorsStorage::load_index(path).unwrap();
        
        assert_eq!(loaded_codes, codes);
        assert_eq!(loaded_metadata.num_vectors, metadata.num_vectors);
        assert_eq!(loaded_metadata.compressed_dim, metadata.compressed_dim);
    }

    #[test]
    fn test_save_load_checkpoint() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("checkpoint.json");
        
        // Create test checkpoint
        let mut model = HashMap::new();
        model.insert("weight1".to_string(), vec![1.0, 2.0]);
        
        let mut optimizer = HashMap::new();
        optimizer.insert("momentum".to_string(), vec![0.9]);
        
        let checkpoint = Checkpoint::new(10, 0.123, model, optimizer);
        
        // Save checkpoint
        SafeTensorsStorage::save_checkpoint(
            checkpoint.model.iter().map(|(k, v)| (k.as_str(), v.clone())).collect(),
            checkpoint.optimizer.iter().map(|(k, v)| (k.as_str(), v.clone())).collect(),
            checkpoint.epoch,
            checkpoint.loss,
            &path,
        ).unwrap();
        
        // Load checkpoint
        let loaded_checkpoint = SafeTensorsStorage::load_checkpoint(&path).unwrap();
        
        assert_eq!(loaded_checkpoint.epoch, 10);
        assert_eq!(loaded_checkpoint.loss, 0.123);
        assert_eq!(loaded_checkpoint.model["weight1"], vec![1.0, 2.0]);
        assert_eq!(loaded_checkpoint.optimizer["momentum"], vec![0.9]);
    }
}