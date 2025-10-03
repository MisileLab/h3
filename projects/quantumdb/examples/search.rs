//! Search example for QuantumDB
//! 
//! Demonstrates how to perform vector search.

use quantumdb_core::{
    models::QuantumCompressorConfig,
    index::SearchConfig,
    backend::NdArray,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt::init();
    
    println!("🔍 QuantumDB Search Example");
    
    // Create a simple model for demonstration
    println!("📦 Creating compression model...");
    let device = NdArray::Device::default();
    let compressor_config = QuantumCompressorConfig::new(768, 256, 16);
    let compressor = compressor_config.init(&device);
    
    // Create search configuration
    let search_config = SearchConfig {
        m: 16,
        ef_construction: 200,
        ef_search: 100,
        max_layers: 5,
        distance_computer: quantumdb_core::utils::SIMDDistance::new(16, 256),
    };
    
    // Create database
    println!("🗄️  Creating database...");
    let mut db = quantumdb_core::QuantumDB::new(compressor, search_config, device);
    
    // Generate some test vectors
    println!("📊 Adding test vectors...");
    let num_vectors = 1000;
    let dim = 768;
    
    for i in 0..num_vectors {
        let mut embedding = Vec::with_capacity(dim);
        for j in 0..dim {
            // Create structured patterns
            let value = if j < 50 {
                // First 50 dimensions encode the ID
                (i as f32 / num_vectors as f32) * (j as f32 + 1.0)
            } else {
                // Rest is noise
                rand::random::<f32>() * 0.1
            };
            embedding.push(value);
        }
        
        // Convert to tensor and add to database
        let tensor = quantumdb_core::tensor::Tensor::from_data(
            quantumdb_core::tensor::TensorData::new(embedding, quantumdb_core::tensor::Shape::new([dim])),
            &device,
        );
        
        db.add(&[i], &tensor.unsqueeze::<2>(0))?;
        
        if (i + 1) % 100 == 0 {
            println!("  Added {} vectors", i + 1);
        }
    }
    
    // Build index
    println!("🏗️  Building HNSW index...");
    db.build_index()?;
    
    // Perform search
    println!("🔍 Performing search...");
    
    // Create a query vector similar to vector 42
    let mut query_embedding = Vec::with_capacity(dim);
    for j in 0..dim {
        let value = if j < 50 {
            (42.0 / num_vectors as f32) * (j as f32 + 1.0) + rand::random::<f32>() * 0.01
        } else {
            rand::random::<f32>() * 0.1
        };
        query_embedding.push(value);
    }
    
    let query_tensor = quantumdb_core::tensor::Tensor::from_data(
        quantumdb_core::tensor::TensorData::new(query_embedding, quantumdb_core::tensor::Shape::new([dim])),
        &device,
    );
    
    // Search for top 10 similar vectors
    let results = db.search(&query_tensor, 10, Some(100))?;
    
    println!("📊 Search Results:");
    println!("  Query: Vector similar to ID 42");
    println!("  Top {} results:", results.len());
    
    for (i, result) in results.iter().enumerate() {
        println!("    {}. ID: {}, Distance: {:.4}", i + 1, result.id, result.distance);
    }
    
    // Show database statistics
    let stats = db.stats();
    println!();
    println!("📈 Database Statistics:");
    println!("  📊 Vectors: {}", stats.num_vectors);
    println!("  🗜️  Compression ratio: {}x", stats.compression_ratio);
    println!("  💾 Memory usage: {:.2} MB", stats.memory_usage_mb);
    
    Ok(())
}