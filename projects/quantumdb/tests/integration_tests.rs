//! Integration tests for QuantumDB

use quantumdb_core::{
    models::QuantumCompressorConfig,
    index::SearchConfig,
    backend::NdArray,
};

#[test]
fn test_end_to_end_workflow() {
    // Create model
    let device = NdArray::Device::default();
    let compressor_config = QuantumCompressorConfig::new(768, 256, 16);
    let compressor = compressor_config.init(&device);
    
    // Create search configuration
    let search_config = SearchConfig::default();
    
    // Create database
    let mut db = quantumdb_core::QuantumDB::new(compressor, search_config, device);
    
    // Add some test vectors
    let num_vectors = 100;
    let dim = 768;
    
    for i in 0..num_vectors {
        let mut embedding = Vec::with_capacity(dim);
        for j in 0..dim {
            let value = (i as f32 / num_vectors as f32) * (j as f32 + 1.0);
            embedding.push(value);
        }
        
        let tensor = quantumdb_core::tensor::Tensor::from_data(
            quantumdb_core::tensor::TensorData::new(embedding, quantumdb_core::tensor::Shape::new([dim])),
            &device,
        );
        
        db.add(&[i], &tensor.unsqueeze::<2>(0)).unwrap();
    }
    
    // Build index
    db.build_index().unwrap();
    
    // Test search
    let mut query_embedding = Vec::with_capacity(dim);
    for j in 0..dim {
        let value = (50.0 / num_vectors as f32) * (j as f32 + 1.0);
        query_embedding.push(value);
    }
    
    let query_tensor = quantumdb_core::tensor::Tensor::from_data(
        quantumdb_core::tensor::TensorData::new(query_embedding, quantumdb_core::tensor::Shape::new([dim])),
        &device,
    );
    
    let results = db.search(&query_tensor, 10, Some(100)).unwrap();
    
    // Should find some results
    assert!(!results.is_empty());
    assert!(results.len() <= 10);
    
    // Check that results are sorted by distance
    for i in 1..results.len() {
        assert!(results[i].distance >= results[i-1].distance);
    }
    
    // Check statistics
    let stats = db.stats();
    assert_eq!(stats.num_vectors, num_vectors);
    assert_eq!(stats.compression_ratio, 192);
}

#[test]
fn test_compression_pipeline() {
    let device = NdArray::Device::default();
    let config = QuantumCompressorConfig::new(768, 256, 16);
    let compressor = config.init(&device);
    
    // Create test embedding
    let mut embedding = Vec::with_capacity(768);
    for i in 0..768 {
        embedding.push(i as f32 / 768.0);
    }
    
    let tensor = quantumdb_core::tensor::Tensor::from_data(
        quantumdb_core::tensor::TensorData::new(embedding, quantumdb_core::tensor::Shape::new([768])),
        &device,
    );
    
    // Test compression
    let compressed = compressor.compress(tensor.clone().unsqueeze::<2>(0));
    assert_eq!(compressed.dims(), [1, 16]);
    
    // Test decompression
    let decompressed = compressor.decompress(compressed);
    assert_eq!(decompressed.dims(), [1, 768]);
    
    // Check compression ratio
    assert_eq!(compressor.compression_ratio(), 192);
}

#[test]
fn test_hnsw_index() {
    let config = SearchConfig::default();
    let hnsw = quantumdb_core::index::HNSWGraph::new(config, 1000);
    
    let stats = hnsw.stats();
    assert_eq!(stats.num_vectors, 0);
    assert!(stats.num_layers > 0);
}

#[test]
fn test_simd_distance() {
    let mut simd_dist = quantumdb_core::utils::SIMDDistance::new(16, 256);
    
    // Mock query subvectors and codebooks
    let query_subvectors: Vec<[f32; 16]> = (0..16)
        .map(|i| [i as f32; 16])
        .collect();
    
    let codebooks: Vec<[[f32; 16]; 256]> = (0..16)
        .map(|_| {
            (0..256)
                .map(|j| [j as f32; 16])
                .collect::<Vec<_>>()
                .try_into()
                .unwrap()
        })
        .collect();
    
    simd_dist.build_lookup_table(&query_subvectors, &codebooks);
    
    // Test distance computation
    let codes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
    let distance = simd_dist.compute(&codes);
    
    assert!(distance >= 0.0);
}

#[test]
fn test_metrics() {
    use quantumdb_core::utils::Metrics;
    use std::collections::HashSet;
    
    let results = vec![vec![1, 2, 3, 4, 5], vec![2, 1, 4, 3, 5]];
    let ground_truth = vec![
        [1, 3].iter().cloned().collect(),
        [2, 4].iter().cloned().collect(),
    ];
    let query_times = vec![0.1, 0.2];
    let k_values = vec![1, 3, 5];
    
    let metrics = Metrics::compute(&results, &ground_truth, &k_values, &query_times);
    
    assert_eq!(metrics.recall_at_k.len(), 3);
    assert_eq!(metrics.ndcg_at_k.len(), 3);
    assert!(metrics.mrr > 0.0);
    assert!(metrics.qps > 0.0);
    assert!(metrics.avg_latency_ms > 0.0);
}