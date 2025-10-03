//! QuantumDB CLI Tool
//! 
//! Command-line interface for training, building, and serving QuantumDB.

use clap::{Parser, Subcommand};
use quantumdb_core::{
    models::QuantumCompressorConfig,
    index::SearchConfig,
    training::{Trainer, TrainingConfig},
    backend::NdArray,
};
use std::net::SocketAddr;
use tracing_subscriber;

/// QuantumDB CLI - Neural compression-powered vector database
#[derive(Parser)]
#[command(name = "quantumdb")]
#[command(about = "QuantumDB CLI - Neural compression-powered vector database", long_about = None)]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Train a new compression model
    Train {
        /// Path to training data (Parquet format)
        #[arg(short, long)]
        data: String,
        
        /// Output path for trained model
        #[arg(short, long)]
        output: String,
        
        /// Input dimension (default: 768)
        #[arg(long, default_value = "768")]
        input_dim: usize,
        
        /// Target dimension (default: 256)
        #[arg(long, default_value = "256")]
        target_dim: usize,
        
        /// Number of subvectors (default: 16)
        #[arg(long, default_value = "16")]
        n_subvectors: usize,
        
        /// Number of training epochs (default: 25)
        #[arg(long, default_value = "25")]
        epochs: usize,
        
        /// Batch size (default: 512)
        #[arg(long, default_value = "512")]
        batch_size: usize,
        
        /// Learning rate (default: 5e-4)
        #[arg(long, default_value = "5e-4")]
        learning_rate: f32,
        
        /// Enable auto-tuning
        #[arg(long)]
        auto_tune: bool,
        
        /// GPU device ID (optional)
        #[arg(long)]
        gpu: Option<usize>,
    },
    
    /// Build an index from embeddings
    Build {
        /// Path to trained model
        #[arg(short, long)]
        model: String,
        
        /// Path to embeddings file
        #[arg(short, long)]
        embeddings: String,
        
        /// Output path for index
        #[arg(short, long)]
        output: String,
        
        /// HNSW M parameter (default: 16)
        #[arg(long, default_value = "16")]
        hnsw_m: usize,
        
        /// HNSW ef_construction parameter (default: 200)
        #[arg(long, default_value = "200")]
        hnsw_ef_construct: usize,
    },
    
    /// Start a server
    Serve {
        /// Path to index
        #[arg(short, long)]
        index: String,
        
        /// Server port (default: 6333)
        #[arg(short, long, default_value = "6333")]
        port: u16,
        
        /// Number of worker threads (default: CPU count)
        #[arg(long)]
        workers: Option<usize>,
        
        /// Enable gRPC server
        #[arg(long)]
        grpc: bool,
        
        /// Enable HTTP server
        #[arg(long)]
        http: bool,
    },
    
    /// Query the database
    Query {
        /// Path to index
        #[arg(short, long)]
        index: String,
        
        /// Query text (will be embedded)
        #[arg(short, long)]
        text: Option<String>,
        
        /// Path to query embedding file
        #[arg(long)]
        embedding_file: Option<String>,
        
        /// Number of results to return (default: 10)
        #[arg(short, long, default_value = "10")]
        top_k: usize,
        
        /// Search depth (default: 100)
        #[arg(long, default_value = "100")]
        ef_search: usize,
    },
    
    /// Benchmark performance
    Benchmark {
        /// Path to index
        #[arg(short, long)]
        index: String,
        
        /// Path to test queries
        #[arg(short, long)]
        queries: String,
        
        /// Path to ground truth file
        #[arg(short, long)]
        ground_truth: String,
        
        /// Number of queries to run (default: all)
        #[arg(long)]
        limit: Option<usize>,
    },
    
    /// Show database statistics
    Stats {
        /// Path to index
        #[arg(short, long)]
        index: String,
    },
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt::init();
    
    let cli = Cli::parse();
    
    match cli.command {
        Commands::Train {
            data,
            output,
            input_dim,
            target_dim,
            n_subvectors,
            epochs,
            batch_size,
            learning_rate,
            auto_tune,
            gpu,
        } => {
            println!("🚀 Starting QuantumDB training...");
            println!("📊 Data: {}", data);
            println!("🎯 Output: {}", output);
            println!("📐 Input dim: {}, Target dim: {}", input_dim, target_dim);
            println!("🔢 Subvectors: {}", n_subvectors);
            println!("🏋️  Epochs: {}, Batch size: {}", epochs, batch_size);
            println!("📈 Learning rate: {}", learning_rate);
            println!("🔧 Auto-tuning: {}", auto_tune);
            
            // Create training configuration
            let training_config = TrainingConfig {
                input_dim,
                target_dim,
                n_subvectors,
                epochs,
                batch_size,
                learning_rate,
                auto_tune,
                ..Default::default()
            };
            
            // Initialize trainer
            let device = NdArray::Device::default();
            let mut trainer = Trainer::new(training_config, device);
            
            // Load data
            println!("📂 Loading training data...");
            let train_data = trainer.load_data(&data)?;
            println!("✅ Loaded {} training samples", train_data.len());
            
            // Train model
            println!("🏃‍♂️ Starting training...");
            let model = trainer.train(train_data).await?;
            
            // Save model
            println!("💾 Saving model to {}", output);
            trainer.save_model(&model, &output)?;
            
            println!("🎉 Training completed successfully!");
        }
        
        Commands::Build {
            model,
            embeddings,
            output,
            hnsw_m,
            hnsw_ef_construct,
        } => {
            println!("🏗️  Building QuantumDB index...");
            println!("📦 Model: {}", model);
            println!("📊 Embeddings: {}", embeddings);
            println!("🎯 Output: {}", output);
            println!("⚙️  HNSW M: {}, ef_construct: {}", hnsw_m, hnsw_ef_construct);
            
            // Load model
            println!("📂 Loading model...");
            let device = NdArray::Device::default();
            let compressor = QuantumCompressorConfig::new(768, 256, 16).init(&device);
            
            // Create search configuration
            let search_config = SearchConfig {
                m: hnsw_m,
                ef_construction: hnsw_ef_construct,
                ..Default::default()
            };
            
            // Create database
            let mut db = quantumdb_core::QuantumDB::new(compressor, search_config, device);
            
            // Load embeddings and build index
            println!("📂 Loading embeddings...");
            // TODO: Implement embedding loading
            println!("🏗️  Building HNSW index...");
            db.build_index().await?;
            
            // Save index
            println!("💾 Saving index to {}", output);
            db.save(&output)?;
            
            println!("🎉 Index built successfully!");
        }
        
        Commands::Serve {
            index,
            port,
            workers,
            grpc,
            http,
        } => {
            println!("🌐 Starting QuantumDB server...");
            println!("📦 Index: {}", index);
            println!("🌍 Port: {}", port);
            if let Some(w) = workers {
                println!("👷 Workers: {}", w);
            }
            println!("🔧 gRPC: {}, HTTP: {}", grpc, http);
            
            // Load database
            println!("📂 Loading index...");
            let device = NdArray::Device::default();
            let db = quantumdb_core::QuantumDB::load(&index, &index, device)?;
            
            // Create service
            let service_config = quantumdb_server::ServiceConfig::default();
            let service = quantumdb_server::QuantumDBService::new(db, service_config);
            
            let addr: SocketAddr = format!("0.0.0.0:{}", port).parse()?;
            
            if http {
                println!("🚀 Starting HTTP server on {}", addr);
                let http_server = quantumdb_server::QuantumDBHttpServer::new(service, addr);
                http_server.serve().await?;
            } else if grpc {
                println!("🚀 Starting gRPC server on {}", addr);
                let grpc_server = quantumdb_server::QuantumDBGrpcServer::new(service, addr);
                grpc_server.serve().await?;
            } else {
                println!("⚠️  No server type specified. Use --http or --grpc");
            }
        }
        
        Commands::Query {
            index,
            text,
            embedding_file,
            top_k,
            ef_search,
        } => {
            println!("🔍 Querying QuantumDB...");
            println!("📦 Index: {}", index);
            println!("🎯 Top-K: {}, ef_search: {}", top_k, ef_search);
            
            if text.is_some() {
                println!("💬 Query text: {:?}", text);
            }
            if embedding_file.is_some() {
                println!("📄 Embedding file: {:?}", embedding_file);
            }
            
            // Load database
            println!("📂 Loading index...");
            let device = NdArray::Device::default();
            let db = quantumdb_core::QuantumDB::load(&index, &index, device)?;
            
            // TODO: Implement query logic
            println!("🔍 Query functionality not yet implemented");
        }
        
        Commands::Benchmark {
            index,
            queries,
            ground_truth,
            limit,
        } => {
            println!("🏃‍♂️ Running benchmark...");
            println!("📦 Index: {}", index);
            println!("📊 Queries: {}", queries);
            println!("🎯 Ground truth: {}", ground_truth);
            if let Some(l) = limit {
                println!("📊 Limit: {}", l);
            }
            
            // TODO: Implement benchmark logic
            println!("🏃‍♂️ Benchmark functionality not yet implemented");
        }
        
        Commands::Stats { index } => {
            println!("📊 QuantumDB Statistics");
            println!("📦 Index: {}", index);
            
            // Load database
            println!("📂 Loading index...");
            let device = NdArray::Device::default();
            let db = quantumdb_core::QuantumDB::load(&index, &index, device)?;
            
            // Get statistics
            let stats = db.stats();
            
            println!();
            println!("📈 Database Statistics:");
            println!("  📊 Vectors: {}", stats.num_vectors);
            println!("  🗜️  Compression ratio: {}x", stats.compression_ratio);
            println!("  📐 Input dimension: {}", stats.input_dim);
            println!("  📏 Compressed dimension: {}", stats.compressed_dim);
            println!("  💾 Memory usage: {:.2} MB", stats.memory_usage_mb);
            
            if stats.num_vectors > 0 {
                let estimated_original_size = stats.num_vectors * stats.input_dim * 4; // float32
                let compressed_size = stats.num_vectors * stats.compressed_dim;
                let space_savings = (1.0 - compressed_size as f64 / estimated_original_size as f64) * 100.0;
                
                println!();
                println!("💾 Storage Analysis:");
                println!("  📊 Original size: {:.2} MB", estimated_original_size as f64 / (1024.0 * 1024.0));
                println!("  🗜️  Compressed size: {:.2} MB", compressed_size as f64 / (1024.0 * 1024.0));
                println!("  💰 Space savings: {:.1}%", space_savings);
            }
        }
    }
    
    Ok(())
}