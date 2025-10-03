//! Server example for QuantumDB
//! 
//! Demonstrates how to start a QuantumDB server.

use quantumdb_core::{
    models::QuantumCompressorConfig,
    index::SearchConfig,
    backend::NdArray,
};
use quantumdb_server::{QuantumDBService, QuantumDBHttpServer, ServiceConfig};
use std::net::SocketAddr;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt::init();
    
    println!("🌐 QuantumDB Server Example");
    
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
    let db = quantumdb_core::QuantumDB::new(compressor, search_config, device);
    
    // Create service
    let service_config = ServiceConfig::default();
    let service = QuantumDBService::new(db, service_config);
    
    // Configure server
    let addr: SocketAddr = "127.0.0.1:6333".parse()?;
    
    println!("🚀 Starting HTTP server on {}", addr);
    println!("📖 API Documentation:");
    println!("  POST /search - Search for similar vectors");
    println!("  POST /add - Add vectors to database");
    println!("  GET /stats - Get database statistics");
    println!("  POST /build_index - Build HNSW index");
    println!("  GET /health - Health check");
    println!();
    println!("💡 Try these commands:");
    println!("  curl -X GET http://localhost:6333/health");
    println!("  curl -X GET http://localhost:6333/stats");
    
    // Start HTTP server
    let http_server = QuantumDBHttpServer::new(service, addr);
    http_server.serve().await?;
    
    Ok(())
}