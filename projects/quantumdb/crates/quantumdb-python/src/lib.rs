//! Python bindings for QuantumDB
//! 
//! Provides a Python interface to the Rust QuantumDB library using PyO3.

use pyo3::prelude::*;
use numpy::{IntoPyArray, PyArray1, PyArray2};
use std::sync::Arc;
use tokio::sync::RwLock;

/// QuantumDB Python wrapper
#[pyclass]
pub struct QuantumDB {
    inner: Arc<RwLock<quantumdb_core::QuantumDB<quantumdb_core::backend::NdArray>>>,
}

#[pymethods]
impl QuantumDB {
    /// Create a new QuantumDB instance
    #[new]
    fn new(
        input_dim: usize,
        target_dim: usize,
        n_subvectors: usize,
    ) -> PyResult<Self> {
        let device = quantumdb_core::backend::NdArray::Device::default();
        let compressor_config = quantumdb_core::models::QuantumCompressorConfig::new(
            input_dim,
            target_dim,
            n_subvectors,
        );
        let compressor = compressor_config.init(&device);
        
        let search_config = quantumdb_core::index::SearchConfig::default();
        let db = quantumdb_core::QuantumDB::new(compressor, search_config, device);
        
        Ok(Self {
            inner: Arc::new(RwLock::new(db)),
        })
    }
    
    /// Load QuantumDB from disk
    #[staticmethod]
    fn load(model_path: &str, index_path: &str) -> PyResult<Self> {
        let device = quantumdb_core::backend::NdArray::Device::default();
        let db = quantumdb_core::QuantumDB::load(model_path, index_path, device)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        
        Ok(Self {
            inner: Arc::new(RwLock::new(db)),
        })
    }
    
    /// Add vectors to the database
    fn add<'py>(
        &self,
        py: Python<'py>,
        ids: Vec<usize>,
        embeddings: &'py PyArray2<f32>,
    ) -> PyResult<()> {
        let embeddings_vec = embeddings.to_vec()?;
        let shape = embeddings.shape();
        
        if ids.len() != shape[0] {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Number of IDs must match number of embeddings"
            ));
        }
        
        // Convert to tensor
        let device = quantumdb_core::backend::NdArray::Device::default();
        let tensor = quantumdb_core::tensor::Tensor::from_data(
            quantumdb_core::tensor::TensorData::new(embeddings_vec, quantumdb_core::tensor::Shape::new(shape)),
            &device,
        );
        
        // Add to database
        let db = self.inner.clone();
        pyo3_asyncio::tokio::future_into_py(py, async move {
            let mut db = db.write().await;
            db.add(&ids, &tensor).await
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
            Ok(())
        })?;
        
        Ok(())
    }
    
    /// Search for similar vectors
    fn search<'py>(
        &self,
        py: Python<'py>,
        query: &'py PyArray1<f32>,
        top_k: usize,
        ef_search: Option<usize>,
    ) -> PyResult<&'py PyArray2<f32>> {
        let query_vec = query.to_vec()?;
        
        // Convert to tensor
        let device = quantumdb_core::backend::NdArray::Device::default();
        let query_tensor = quantumdb_core::tensor::Tensor::from_data(
            quantumdb_core::tensor::TensorData::new(query_vec, quantumdb_core::tensor::Shape::new([query_vec.len()])),
            &device,
        );
        
        // Perform search
        let db = self.inner.clone();
        let results = pyo3_asyncio::tokio::get_runtime().block_on(async move {
            let db = db.read().await;
            db.search(&query_tensor, top_k, ef_search).await
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
        })?;
        
        // Convert results to numpy array
        let result_data: Vec<f32> = results
            .into_iter()
            .flat_map(|r| [r.id as f32, r.distance])
            .collect();
        
        let result_array = result_data.into_pyarray(py).reshape([top_k, 2])?;
        Ok(result_array)
    }
    
    /// Build the HNSW index
    fn build_index<'py>(&self, py: Python<'py>) -> PyResult<()> {
        let db = self.inner.clone();
        pyo3_asyncio::tokio::future_into_py(py, async move {
            let mut db = db.write().await;
            db.build_index().await
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
            Ok(())
        })?;
        
        Ok(())
    }
    
    /// Get database statistics
    fn stats(&self) -> PyResult<QuantumDBStats> {
        let db = self.inner.clone();
        let stats = pyo3_asyncio::tokio::get_runtime().block_on(async move {
            let db = db.read().await;
            Ok(db.stats())
        })?;
        
        Ok(QuantumDBStats {
            num_vectors: stats.num_vectors,
            compression_ratio: stats.compression_ratio,
            input_dim: stats.input_dim,
            compressed_dim: stats.compressed_dim,
            memory_usage_mb: stats.memory_usage_mb,
        })
    }
    
    /// Save the database to disk
    fn save(&self, path: &str) -> PyResult<()> {
        let db = self.inner.clone();
        pyo3_asyncio::tokio::get_runtime().block_on(async move {
            let db = db.read().await;
            db.save(path).await
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
            Ok(())
        })?;
        
        Ok(())
    }
}

/// Database statistics for Python
#[pyclass]
#[derive(Clone)]
pub struct QuantumDBStats {
    #[pyo3(get)]
    pub num_vectors: usize,
    #[pyo3(get)]
    pub compression_ratio: usize,
    #[pyo3(get)]
    pub input_dim: usize,
    #[pyo3(get)]
    pub compressed_dim: usize,
    #[pyo3(get)]
    pub memory_usage_mb: f64,
}

#[pymethods]
impl QuantumDBStats {
    /// String representation
    fn __repr__(&self) -> String {
        format!(
            "QuantumDBStats(num_vectors={}, compression_ratio={}x, memory_usage_mb={:.2})",
            self.num_vectors, self.compression_ratio, self.memory_usage_mb
        )
    }
    
    /// Dictionary representation
    fn as_dict(&self) -> std::collections::HashMap<String, pyo3::PyObject> {
        Python::with_gil(|py| {
            let dict = pyo3::types::PyDict::new(py);
            dict.set_item("num_vectors", self.num_vectors).unwrap();
            dict.set_item("compression_ratio", self.compression_ratio).unwrap();
            dict.set_item("input_dim", self.input_dim).unwrap();
            dict.set_item("compressed_dim", self.compressed_dim).unwrap();
            dict.set_item("memory_usage_mb", self.memory_usage_mb).unwrap();
            
            dict.into()
        })
    }
}

/// QuantumDB module
#[pymodule]
fn quantumdb(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<QuantumDB>()?;
    m.add_class::<QuantumDBStats>()?;
    
    // Add version info
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::Python;

    #[test]
    fn test_quantumdb_creation() {
        Python::with_gil(|py| {
            let db = QuantumDB::new(768, 256, 16).unwrap();
            let stats = db.stats().unwrap();
            assert_eq!(stats.num_vectors, 0);
            assert_eq!(stats.compression_ratio, 192);
        });
    }

    #[test]
    fn test_quantumdb_stats_repr() {
        let stats = QuantumDBStats {
            num_vectors: 1000,
            compression_ratio: 192,
            input_dim: 768,
            compressed_dim: 16,
            memory_usage_mb: 10.5,
        };
        
        let repr = stats.__repr__();
        assert!(repr.contains("1000"));
        assert!(repr.contains("192x"));
        assert!(repr.contains("10.5"));
    }
}