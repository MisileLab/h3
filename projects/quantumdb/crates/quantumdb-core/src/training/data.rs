//! Data loading and preprocessing for training

use std::path::Path;
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowReader;
use parquet::file::reader::SerializedFileReader;
use crate::{Result, QuantumDBError};

/// Data loader for training
pub struct DataLoader {
    /// Record batches
    batches: Vec<RecordBatch>,
    /// Current batch index
    current_batch: usize,
    /// Batch size
    batch_size: usize,
}

impl DataLoader {
    /// Create a new data loader from Parquet file
    /// 
    /// # Arguments
    /// * `path` - Path to Parquet file
    /// * `batch_size` - Batch size
    /// 
    /// # Returns
    /// Data loader
    pub fn from_parquet<P: AsRef<Path>>(path: P, batch_size: usize) -> Result<Self> {
        let file = std::fs::File::open(path)?;
        let reader = SerializedFileReader::new(file)?;
        let mut arrow_reader = arrow::ParquetFileArrowReader::new(reader);
        
        let mut batches = Vec::new();
        while let Some(batch) = arrow_reader.get_next_record_batch(1024)? {
            batches.push(batch);
        }
        
        Ok(Self {
            batches,
            current_batch: 0,
            batch_size,
        })
    }
    
    /// Get the next batch
    /// 
    /// # Returns
    /// Optional batch of embeddings
    pub fn next_batch(&mut self) -> Option<Vec<Vec<f32>>> {
        if self.current_batch >= self.batches.len() {
            return None;
        }
        
        let batch = &self.batches[self.current_batch];
        self.current_batch += 1;
        
        // Extract embeddings from batch
        // This assumes there's an 'embedding' column with float32 values
        if let Some(embedding_array) = batch.column_by_name("embedding") {
            let embeddings = self.extract_embeddings(embedding_array)?;
            Some(embeddings)
        } else {
            None
        }
    }
    
    /// Extract embeddings from arrow array
    fn extract_embeddings(&self, array: &arrow::array::ArrayRef) -> Result<Vec<Vec<f32>>> {
        use arrow::array::Float32Array;
        
        let float_array = array.as_any().downcast_ref::<Float32Array>()
            .ok_or_else(|| QuantumDBError::Config("Expected Float32Array".to_string()))?;
        
        let mut embeddings = Vec::new();
        let values = float_array.values();
        
        // Assuming each embedding is a fixed size (e.g., 768)
        let embedding_dim = 768;
        
        for chunk in values.chunks_exact(embedding_dim) {
            embeddings.push(chunk.to_vec());
        }
        
        Ok(embeddings)
    }
    
    /// Reset the data loader
    pub fn reset(&mut self) {
        self.current_batch = 0;
    }
    
    /// Get the total number of batches
    pub fn num_batches(&self) -> usize {
        self.batches.len()
    }
    
    /// Get the current batch index
    pub fn current_batch(&self) -> usize {
        self.current_batch
    }
}

/// Data augmentation utilities
pub struct DataAugmentation;

impl DataAugmentation {
    /// Apply random noise to embeddings
    /// 
    /// # Arguments
    /// * `embeddings` - Input embeddings
    /// * `noise_level` - Noise level (standard deviation)
    /// 
    /// # Returns
    /// Augmented embeddings
    pub fn add_noise(embeddings: &[Vec<f32>], noise_level: f32) -> Vec<Vec<f32>> {
        embeddings
            .iter()
            .map(|embedding| {
                embedding
                    .iter()
                    .map(|&x| x + rand::random::<f32>() * noise_level - noise_level / 2.0)
                    .collect()
            })
            .collect()
    }
    
    /// Apply random masking to embeddings
    /// 
    /// # Arguments
    /// * `embeddings` - Input embeddings
    /// * `mask_ratio` - Ratio of dimensions to mask
    /// 
    /// # Returns
    /// Augmented embeddings
    pub fn random_mask(embeddings: &[Vec<f32>], mask_ratio: f32) -> Vec<Vec<f32>> {
        embeddings
            .iter()
            .map(|embedding| {
                embedding
                    .iter()
                    .map(|&x| {
                        if rand::random::<f32>() < mask_ratio {
                            0.0
                        } else {
                            x
                        }
                    })
                    .collect()
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_data_augmentation_noise() {
        let embeddings = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
        ];
        
        let augmented = DataAugmentation::add_noise(&embeddings, 0.1);
        
        assert_eq!(augmented.len(), embeddings.len());
        assert_eq!(augmented[0].len(), embeddings[0].len());
        
        // Values should be different (with high probability)
        let mut different = false;
        for (orig, aug) in embeddings.iter().zip(augmented.iter()) {
            for (o, a) in orig.iter().zip(aug.iter()) {
                if (o - a).abs() > 1e-6 {
                    different = true;
                    break;
                }
            }
        }
        assert!(different);
    }

    #[test]
    fn test_data_augmentation_mask() {
        let embeddings = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
        ];
        
        let augmented = DataAugmentation::random_mask(&embeddings, 0.5);
        
        assert_eq!(augmented.len(), embeddings.len());
        assert_eq!(augmented[0].len(), embeddings[0].len());
        
        // Some values should be masked (set to 0)
        let mut has_zeros = false;
        for embedding in &augmented {
            for &value in embedding {
                if value == 0.0 {
                    has_zeros = true;
                    break;
                }
            }
        }
        // With mask_ratio=0.5, probability of no zeros is very low
        assert!(has_zeros);
    }

    #[test]
    fn test_data_loader_reset() {
        // This test would require creating a real Parquet file
        // For now, we'll just test the reset logic
        let mut loader = DataLoader {
            batches: Vec::new(),
            current_batch: 5,
            batch_size: 32,
        };
        
        assert_eq!(loader.current_batch(), 5);
        loader.reset();
        assert_eq!(loader.current_batch(), 0);
    }
}