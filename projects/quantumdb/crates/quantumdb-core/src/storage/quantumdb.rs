//! Main QuantumDB interface
//! 
//! High-level interface that combines compression, indexing, and storage
//! for a complete vector database solution.

use std::{
    collections::HashMap,
    path::Path,
    sync::Arc,
};
use burn::{
    backend::Backend,
    tensor::Tensor,
};
use crate::{
    models::{QuantumCompressor, CompressionModel},
    index::{HNSWGraph, SearchConfig},
    storage::{MemoryMappedStorage, SafeTensorsStorage, IndexMetadata},
    utils::SIMDDistance,
    Result, QuantumDBError,
};

/// Main QuantumDB interface
/// 
/// Combines neural compression, HNSW indexing, and memory-mapped storage
/// for efficient vector search with 192x compression.
pub struct QuantumDB<B: Backend> {
    /// Neural compression model
    compressor: QuantumCompressor<B>,
    /// HNSW index for search
    index: HNSWGraph,
    /// Storage for compressed vectors
    storage: Option<MemoryMappedStorage>,
    /// Distance computer for search
    distance_computer: SIMDDistance,
    /// Backend device
    device: B::Device,
}

impl<B: Backend> QuantumDB<B> {
    /// Create a new QuantumDB instance
    /// 
    /// # Arguments
    /// * `compressor` - Trained compression model
    /// * `config` - HNSW search configuration
    /// * `device` - Backend device
    /// 
    /// # Returns
    /// New QuantumDB instance
    pub fn new(
        compressor: QuantumCompressor<B>,
        config: SearchConfig,
        device: B::Device,
    ) -> Self {
        let distance_computer = SIMDDistance::new(
            compressor.compressed_dim(),
            compressor.pq().codebook_size(),
        );
        
        let index = HNSWGraph::new(config, 0);
        
        Self {
            compressor,
            index,
            storage: None,
            distance_computer,
            device,
        }
    }
    
    /// Load QuantumDB from disk
    /// 
    /// # Arguments
    /// * `model_path` - Path to trained model
    /// * `index_path` - Path to saved index
    /// * `device` - Backend device
    /// 
    /// # Returns
    /// Loaded QuantumDB instance
    pub fn load<P: AsRef<Path>>(
        model_path: P,
        index_path: P,
        device: B::Device,
    ) -> Result<Self> {
        // Load model
        let model_data = SafeTensorsStorage::load_model(model_path)?;
        let compressor = Self::load_compressor_from_data(model_data, &device)?;
        
        // Load index
        let (codes, metadata) = SafeTensorsStorage::load_index(index_path)?;
        let storage = MemoryMappedStorage::open(
            index_path.as_ref().join("codes.safetensors"),
            index_path.as_ref().join("metadata.json"),
        )?;
        
        // Create HNSW index
        let config = SearchConfig {
            m: 16,
            ef_construction: 200,
            ef_search: 100,
            max_layers: 0,
            distance_computer: SIMDDistance::new(
                compressor.compressed_dim(),
                compressor.pq().codebook_size(),
            ),
        };
        
        let index = HNSWGraph::new(config, metadata.num_vectors);
        
        let distance_computer = SIMDDistance::new(
            compressor.compressed_dim(),
            compressor.pq().codebook_size(),
        );
        
        Ok(Self {
            compressor,
            index,
            storage: Some(storage),
            distance_computer,
            device,
        })
    }
    
    /// Add vectors to the database
    /// 
    /// # Arguments
    /// * `ids` - Vector IDs
    /// * `embeddings` - Full-precision embeddings
    pub fn add(&mut self, ids: &[usize], embeddings: &Tensor<B, 2>) -> Result<()> {
        if ids.len() != embeddings.dims()[0] {
            return Err(QuantumDBError::Config(
                "Number of IDs must match number of embeddings".to_string()
            ));
        }
        
        // Compress embeddings
        let compressed = self.compressor.compress(embeddings.clone());
        
        // Convert compressed tensor to codes
        let compressed_data = compressed.into_data();
        let compressed_vecs: Vec<Vec<u8>> = compressed_data
            .value
            .iter()
            .map(|&x| x as u8)
            .collect();
        
        // Add to HNSW index
        for (i, &id) in ids.iter().enumerate() {
            let start_idx = i * self.compressor.compressed_dim();
            let end_idx = start_idx + self.compressor.compressed_dim();
            let code: Vec<u8> = compressed_vecs[start_idx..end_idx].to_vec();
            let code_array: [u8; 16] = code.try_into()
                .map_err(|_| QuantumDBError::Index("Invalid code length".to_string()))?;
            
            self.index.insert(id, &code_array)?;
        }
        
        Ok(())
    }
    
    /// Search for nearest neighbors
    /// 
    /// # Arguments
    /// * `query` - Query embedding
    /// * `top_k` - Number of results to return
    /// * `ef_search` - Search depth (optional)
    /// 
    /// # Returns
    /// Vector of search results
    pub fn search(
        &self,
        query: &Tensor<B, 1>,
        top_k: usize,
        ef_search: Option<usize>,
    ) -> Result<Vec<crate::index::SearchResult>> {
        // Compress query
        let query_batch = query.clone().unsqueeze::<2>(0);
        let compressed_query = self.compressor.compress(query_batch);
        
        // Convert to compressed codes
        let compressed_data = compressed_query.into_data();
        let compressed_vec: Vec<u8> = compressed_data
            .value
            .iter()
            .map(|&x| x as u8)
            .collect();
        
        let query_code: [u8; 16] = compressed_vec.try_into()
            .map_err(|_| QuantumDBError::Index("Invalid query code length".to_string()))?;
        
        // Search HNSW index
        self.index.search(&query_code, top_k, ef_search)
    }
    
    /// Build HNSW index from added vectors
    pub fn build_index(&mut self) -> Result<()> {
        // In a real implementation, this would optimize the HNSW graph
        // For now, we'll just mark it as built
        Ok(())
    }
    
    /// Save the database to disk
    /// 
    /// # Arguments
    /// * `path` - Output directory path
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let path = path.as_ref();
        std::fs::create_dir_all(path)?;
        
        // Save model
        let model_path = path.join("model.safetensors");
        self.save_compressor(&model_path)?;
        
        // Save index
        let index_path = path.join("index");
        self.save_index(&index_path)?;
        
        Ok(())
    }
    
    /// Get database statistics
    pub fn stats(&self) -> QuantumDBStats {
        let index_stats = self.index.stats();
        
        QuantumDBStats {
            num_vectors: index_stats.num_vectors,
            compression_ratio: self.compressor.compression_ratio(),
            input_dim: self.compressor.input_dim(),
            compressed_dim: self.compressor.compressed_dim(),
            memory_usage_mb: self.estimate_memory_usage(),
        }
    }
    
    /// Load compressor from tensor data
    fn load_compressor_from_data(
        model_data: HashMap<String, Vec<f32>>,
        device: &B::Device,
    ) -> Result<QuantumCompressor<B>> {
        // This is a simplified implementation
        // In practice, you'd need to properly reconstruct the Burn model
        // from the saved weights
        
        // For now, create a default compressor
        let config = crate::models::QuantumCompressorConfig::new(768, 256, 16);
        Ok(config.init(device))
    }
    
    /// Save compressor to disk
    fn save_compressor<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        // This is a simplified implementation
        // In practice, you'd extract the weights from the Burn model
        let tensors = HashMap::new();
        SafeTensorsStorage::save_model(tensors, path)?;
        Ok(())
    }
    
    /// Save index to disk
    fn save_index<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let metadata = IndexMetadata {
            num_vectors: self.index.stats().num_vectors,
            compressed_dim: self.compressor.compressed_dim(),
            version: 1,
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        };
        
        let codes = Vec::new(); // In practice, extract from index
        SafeTensorsStorage::save_index(&codes, &metadata, path)?;
        Ok(())
    }
    
    /// Estimate memory usage in MB
    fn estimate_memory_usage(&self) -> f64 {
        let stats = self.index.stats();
        let compressed_bytes = stats.num_vectors * self.compressor.compressed_dim();
        let graph_bytes = stats.num_vectors * stats.m * std::mem::size_of::<usize>();
        
        (compressed_bytes + graph_bytes) as f64 / (1024.0 * 1024.0)
    }
}

/// QuantumDB statistics
#[derive(Debug, Clone)]
pub struct QuantumDBStats {
    /// Number of vectors in the database
    pub num_vectors: usize,
    /// Compression ratio achieved
    pub compression_ratio: usize,
    /// Input dimension
    pub input_dim: usize,
    /// Compressed dimension
    pub compressed_dim: usize,
    /// Estimated memory usage in MB
    pub memory_usage_mb: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type Backend = NdArray;

    #[test]
    fn test_quantumdb_creation() {
        let device = Default::default();
        let config = crate::models::QuantumCompressorConfig::new(768, 256, 16);
        let compressor = config.init(&device);
        let search_config = SearchConfig::default();
        
        let db = QuantumDB::new(compressor, search_config, device);
        
        let stats = db.stats();
        assert_eq!(stats.num_vectors, 0);
        assert_eq!(stats.compression_ratio, 192);
        assert_eq!(stats.input_dim, 768);
        assert_eq!(stats.compressed_dim, 16);
    }

    #[test]
    fn test_compression_ratio() {
        let device = Default::default();
        let config = crate::models::QuantumCompressorConfig::new(768, 256, 16);
        let compressor = config.init(&device);
        
        assert_eq!(compressor.compression_ratio(), 192);
    }
}