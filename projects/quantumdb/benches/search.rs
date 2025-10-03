//! Search performance benchmarks

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use quantumdb_core::{
    models::QuantumCompressorConfig,
    index::SearchConfig,
    backend::NdArray,
};

fn bench_search(c: &mut Criterion) {
    let device = NdArray::Device::default();
    let compressor_config = QuantumCompressorConfig::new(768, 256, 16);
    let compressor = compressor_config.init(&device);
    
    let search_config = SearchConfig::default();
    let mut db = quantumdb_core::QuantumDB::new(compressor, search_config, device);
    
    // Add test vectors
    let num_vectors = 10000;
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
    
    db.build_index().unwrap();
    
    // Create query
    let mut query_embedding = Vec::with_capacity(dim);
    for j in 0..dim {
        let value = (5000.0 / num_vectors as f32) * (j as f32 + 1.0);
        query_embedding.push(value);
    }
    
    let query_tensor = quantumdb_core::tensor::Tensor::from_data(
        quantumdb_core::tensor::TensorData::new(query_embedding, quantumdb_core::tensor::Shape::new([dim])),
        &device,
    );
    
    c.bench_function("search_10", |b| {
        b.iter(|| {
            black_box(db.search(black_box(&query_tensor), 10, Some(100)).unwrap())
        })
    });
    
    c.bench_function("search_100", |b| {
        b.iter(|| {
            black_box(db.search(black_box(&query_tensor), 100, Some(100)).unwrap())
        })
    });
}

fn bench_compression(c: &mut Criterion) {
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
    
    c.bench_function("compress", |b| {
        b.iter(|| {
            black_box(compressor.compress(black_box(tensor.clone().unsqueeze::<2>(0))))
        })
    });
    
    let compressed = compressor.compress(tensor.clone().unsqueeze::<2>(0));
    
    c.bench_function("decompress", |b| {
        b.iter(|| {
            black_box(compressor.decompress(black_box(compressed.clone())))
        })
    });
}

fn bench_distance(c: &mut Criterion) {
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
    
    let codes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
    
    c.bench_function("simd_distance", |b| {
        b.iter(|| {
            black_box(simd_dist.compute(black_box(&codes)))
        })
    });
    
    // Standard distance for comparison
    let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0];
    let b = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0];
    
    c.bench_function("standard_distance", |b| {
        b.iter(|| {
            black_box(quantumdb_core::utils::distance::l2_distance(black_box(&a), black_box(&b)))
        })
    });
}

criterion_group!(benches, bench_search, bench_compression, bench_distance);
criterion_main!(benches);