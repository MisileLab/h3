//! QuantumDB service implementation
//! 
//! Core service logic that handles vector operations and search requests.

use std::sync::Arc;
use tokio::sync::RwLock;
use quantumdb_core::{
    QuantumDB, 
    backend::NdArray,
    Result as QuantumResult,
    QuantumDBError,
};
use serde::{Deserialize, Serialize};

/// QuantumDB service that handles all operations
pub struct QuantumDBService {
    /// Database instance
    db: Arc<RwLock<QuantumDB<NdArray>>>,
    /// Service configuration
    config: ServiceConfig,
}

/// Service configuration
#[derive(Debug, Clone)]
pub struct ServiceConfig {
    /// Maximum batch size for operations
    pub max_batch_size: usize,
    /// Default search parameters
    pub default_top_k: usize,
    pub default_ef_search: usize,
    /// Request timeout in seconds
    pub request_timeout_secs: u64,
}

impl Default for ServiceConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 1000,
            default_top_k: 10,
            default_ef_search: 100,
            request_timeout_secs: 30,
        }
    }
}

/// Search request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchRequest {
    /// Query embedding
    pub embedding: Vec<f32>,
    /// Number of results to return
    pub top_k: Option<usize>,
    /// Search depth
    pub ef_search: Option<usize>,
}

/// Search response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResponse {
    /// Search results
    pub results: Vec<SearchResult>,
    /// Query time in milliseconds
    pub query_time_ms: f64,
}

/// Single search result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    /// Vector ID
    pub id: usize,
    /// Distance to query
    pub distance: f32,
}

/// Add vectors request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AddRequest {
    /// Vector IDs
    pub ids: Vec<usize>,
    /// Embeddings to add
    pub embeddings: Vec<Vec<f32>>,
}

/// Add vectors response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AddResponse {
    /// Number of vectors added
    pub count: usize,
    /// Time taken in milliseconds
    pub time_ms: f64,
}

/// Database statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseStats {
    /// Number of vectors in database
    pub num_vectors: usize,
    /// Compression ratio
    pub compression_ratio: usize,
    /// Input dimension
    pub input_dim: usize,
    /// Compressed dimension
    pub compressed_dim: usize,
    /// Memory usage in MB
    pub memory_usage_mb: f64,
}

impl QuantumDBService {
    /// Create a new service instance
    /// 
    /// # Arguments
    /// * `db` - QuantumDB instance
    /// * `config` - Service configuration
    pub fn new(db: QuantumDB<NdArray>, config: ServiceConfig) -> Self {
        Self {
            db: Arc::new(RwLock::new(db)),
            config,
        }
    }
    
    /// Search for similar vectors
    /// 
    /// # Arguments
    /// * `request` - Search request
    /// 
    /// # Returns
    /// Search response
    pub async fn search(&self, request: SearchRequest) -> Result<SearchResponse, ServiceError> {
        let start_time = std::time::Instant::now();
        
        let top_k = request.top_k.unwrap_or(self.config.default_top_k);
        let ef_search = request.ef_search.unwrap_or(self.config.default_ef_search);
        
        // Validate request
        if request.embedding.is_empty() {
            return Err(ServiceError::InvalidRequest("Empty embedding".to_string()));
        }
        
        if top_k == 0 {
            return Err(ServiceError::InvalidRequest("top_k must be > 0".to_string()));
        }
        
        // Perform search
        let db = self.db.read().await;
        
        // Convert embedding to tensor
        let device = burn::backend::NdArray::Device::default();
        let embedding_tensor = burn::tensor::Tensor::from_data(
            burn::tensor::TensorData::new(request.embedding, burn::tensor::Shape::new([request.embedding.len()])),
            &device,
        );
        
        let results = db.search(&embedding_tensor, top_k, Some(ef_search))
            .await
            .map_err(|e| ServiceError::Database(e.to_string()))?;
        
        let query_time_ms = start_time.elapsed().as_secs_f64() * 1000.0;
        
        // Convert results
        let search_results: Vec<SearchResult> = results
            .into_iter()
            .map(|r| SearchResult {
                id: r.id,
                distance: r.distance,
            })
            .collect();
        
        Ok(SearchResponse {
            results: search_results,
            query_time_ms,
        })
    }
    
    /// Add vectors to the database
    /// 
    /// # Arguments
    /// * `request` - Add request
    /// 
    /// # Returns
    /// Add response
    pub async fn add(&self, request: AddRequest) -> Result<AddResponse, ServiceError> {
        let start_time = std::time::Instant::now();
        
        // Validate request
        if request.ids.is_empty() {
            return Err(ServiceError::InvalidRequest("Empty IDs".to_string()));
        }
        
        if request.embeddings.is_empty() {
            return Err(ServiceError::InvalidRequest("Empty embeddings".to_string()));
        }
        
        if request.ids.len() != request.embeddings.len() {
            return Err(ServiceError::InvalidRequest(
                "Number of IDs must match number of embeddings".to_string()
            ));
        }
        
        if request.ids.len() > self.config.max_batch_size {
            return Err(ServiceError::InvalidRequest(
                format!("Batch size {} exceeds maximum {}", request.ids.len(), self.config.max_batch_size)
            ));
        }
        
        // Check embedding dimensions
        let embedding_dim = request.embeddings[0].len();
        for embedding in &request.embeddings {
            if embedding.len() != embedding_dim {
                return Err(ServiceError::InvalidRequest(
                    "All embeddings must have the same dimension".to_string()
                ));
            }
        }
        
        // Add vectors
        let mut db = self.db.write().await;
        
        // Convert embeddings to tensor
        let device = burn::backend::NdArray::Device::default();
        let embeddings_flat: Vec<f32> = request.embeddings.iter().flatten().copied().collect();
        let embeddings_tensor = burn::tensor::Tensor::from_data(
            burn::tensor::TensorData::new(
                embeddings_flat,
                burn::tensor::Shape::new([request.embeddings.len(), embedding_dim])
            ),
            &device,
        );
        
        db.add(&request.ids, &embeddings_tensor)
            .await
            .map_err(|e| ServiceError::Database(e.to_string()))?;
        
        let time_ms = start_time.elapsed().as_secs_f64() * 1000.0;
        
        Ok(AddResponse {
            count: request.ids.len(),
            time_ms,
        })
    }
    
    /// Get database statistics
    /// 
    /// # Returns
    /// Database statistics
    pub async fn stats(&self) -> Result<DatabaseStats, ServiceError> {
        let db = self.db.read().await;
        let stats = db.stats();
        
        Ok(DatabaseStats {
            num_vectors: stats.num_vectors,
            compression_ratio: stats.compression_ratio,
            input_dim: stats.input_dim,
            compressed_dim: stats.compressed_dim,
            memory_usage_mb: stats.memory_usage_mb,
        })
    }
    
    /// Build the HNSW index
    /// 
    /// # Returns
    /// Success response
    pub async fn build_index(&self) -> Result<(), ServiceError> {
        let mut db = self.db.write().await;
        db.build_index()
            .await
            .map_err(|e| ServiceError::Database(e.to_string()))?;
        Ok(())
    }
}

/// Service errors
#[derive(Debug, thiserror::Error)]
pub enum ServiceError {
    #[error("Invalid request: {0}")]
    InvalidRequest(String),
    
    #[error("Database error: {0}")]
    Database(String),
    
    #[error("Internal error: {0}")]
    Internal(String),
}

impl From<ServiceError> for tonic::Status {
    fn from(err: ServiceError) -> Self {
        match err {
            ServiceError::InvalidRequest(msg) => {
                tonic::Status::invalid_argument(msg)
            }
            ServiceError::Database(msg) => {
                tonic::Status::internal(format!("Database error: {}", msg))
            }
            ServiceError::Internal(msg) => {
                tonic::Status::internal(format!("Internal error: {}", msg))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quantumdb_core::{
        models::QuantumCompressorConfig,
        index::SearchConfig,
    };

    #[tokio::test]
    async fn test_service_creation() {
        let device = burn::backend::NdArray::Device::default();
        let compressor_config = QuantumCompressorConfig::new(768, 256, 16);
        let compressor = compressor_config.init(&device);
        let search_config = SearchConfig::default();
        
        let db = QuantumDB::new(compressor, search_config, device);
        let service = QuantumDBService::new(db, ServiceConfig::default());
        
        let stats = service.stats().await.unwrap();
        assert_eq!(stats.num_vectors, 0);
    }

    #[tokio::test]
    async fn test_search_validation() {
        let device = burn::backend::NdArray::Device::default();
        let compressor_config = QuantumCompressorConfig::new(768, 256, 16);
        let compressor = compressor_config.init(&device);
        let search_config = SearchConfig::default();
        
        let db = QuantumDB::new(compressor, search_config, device);
        let service = QuantumDBService::new(db, ServiceConfig::default());
        
        // Empty embedding should fail
        let request = SearchRequest {
            embedding: vec![],
            top_k: Some(10),
            ef_search: Some(100),
        };
        
        let result = service.search(request).await;
        assert!(matches!(result, Err(ServiceError::InvalidRequest(_))));
    }

    #[tokio::test]
    async fn test_add_validation() {
        let device = burn::backend::NdArray::Device::default();
        let compressor_config = QuantumCompressorConfig::new(768, 256, 16);
        let compressor = compressor_config.init(&device);
        let search_config = SearchConfig::default();
        
        let db = QuantumDB::new(compressor, search_config, device);
        let service = QuantumDBService::new(db, ServiceConfig::default());
        
        // Mismatched IDs and embeddings should fail
        let request = AddRequest {
            ids: vec![1, 2],
            embeddings: vec![vec![1.0; 768]], // Only one embedding
        };
        
        let result = service.add(request).await;
        assert!(matches!(result, Err(ServiceError::InvalidRequest(_))));
    }
}