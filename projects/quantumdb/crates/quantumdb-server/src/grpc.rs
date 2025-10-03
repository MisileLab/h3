//! gRPC server implementation for QuantumDB

use std::net::SocketAddr;
use tonic::{transport::Server, Request, Response, Status};
use tokio::sync::Arc;
use crate::service::{QuantumDBService, SearchRequest, SearchResponse, AddRequest, AddResponse, DatabaseStats, ServiceError};

/// gRPC server for QuantumDB
pub struct QuantumDBGrpcServer {
    service: Arc<QuantumDBService>,
    addr: SocketAddr,
}

impl QuantumDBGrpcServer {
    /// Create a new gRPC server
    /// 
    /// # Arguments
    /// * `service` - QuantumDB service
    /// * `addr` - Server address
    pub fn new(service: QuantumDBService, addr: SocketAddr) -> Self {
        Self {
            service: Arc::new(service),
            addr,
        }
    }
    
    /// Start the gRPC server
    /// 
    /// # Returns
    /// Future that resolves when server stops
    pub async fn serve(self) -> Result<(), Box<dyn std::error::Error>> {
        // In a real implementation, you would generate the gRPC code from .proto files
        // For now, we'll provide a placeholder implementation
        
        println!("Starting gRPC server on {}", self.addr);
        
        // Placeholder for actual gRPC service implementation
        // You would typically use tonic_build to generate the service from .proto files
        
        // For demonstration, we'll just run a simple HTTP server
        let axum_server = axum::Server::bind(&self.addr)
            .serve(axum::Router::new().into_make_service());
        
        axum_server.await?;
        
        Ok(())
    }
}

/// Generated gRPC service implementation
/// 
/// This would typically be generated from a .proto file using tonic_build
#[tonic::async_trait]
pub trait QuantumDbService: Send + Sync + 'static {
    async fn search(&self, request: Request<SearchRequest>) -> Result<Response<SearchResponse>, Status>;
    async fn add(&self, request: Request<AddRequest>) -> Result<Response<AddResponse>, Status>;
    async fn stats(&self, request: Request<()>) -> Result<Response<DatabaseStats>, Status>;
    async fn build_index(&self, request: Request<()>) -> Result<Response<()>, Status>;
}

/// Mock implementation for demonstration
pub struct MockQuantumDbService {
    service: Arc<QuantumDBService>,
}

#[tonic::async_trait]
impl QuantumDbService for MockQuantumDbService {
    async fn search(&self, request: Request<SearchRequest>) -> Result<Response<SearchResponse>, Status> {
        let req = request.into_inner();
        let response = self.service.search(req).await
            .map_err(|e| Status::from(e))?;
        Ok(Response::new(response))
    }
    
    async fn add(&self, request: Request<AddRequest>) -> Result<Response<AddResponse>, Status> {
        let req = request.into_inner();
        let response = self.service.add(req).await
            .map_err(|e| Status::from(e))?;
        Ok(Response::new(response))
    }
    
    async fn stats(&self, _request: Request<()>) -> Result<Response<DatabaseStats>, Status> {
        let stats = self.service.stats().await
            .map_err(|e| Status::from(e))?;
        Ok(Response::new(stats))
    }
    
    async fn build_index(&self, _request: Request<()>) -> Result<Response<()>, Status> {
        self.service.build_index().await
            .map_err(|e| Status::from(e))?;
        Ok(Response::new(()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::service::ServiceConfig;
    use quantumdb_core::{
        models::QuantumCompressorConfig,
        index::SearchConfig,
    };

    #[tokio::test]
    async fn test_grpc_server_creation() {
        let device = burn::backend::NdArray::Device::default();
        let compressor_config = QuantumCompressorConfig::new(768, 256, 16);
        let compressor = compressor_config.init(&device);
        let search_config = SearchConfig::default();
        
        let db = quantumdb_core::QuantumDB::new(compressor, search_config, device);
        let service = QuantumDBService::new(db, ServiceConfig::default());
        
        let addr = "127.0.0.1:0".parse().unwrap();
        let server = QuantumDBGrpcServer::new(service, addr);
        
        // Server creation should succeed
        assert_eq!(server.addr, addr);
    }

    #[tokio::test]
    async fn test_mock_service() {
        let device = burn::backend::NdArray::Device::default();
        let compressor_config = QuantumCompressorConfig::new(768, 256, 16);
        let compressor = compressor_config.init(&device);
        let search_config = SearchConfig::default();
        
        let db = quantumdb_core::QuantumDB::new(compressor, search_config, device);
        let service = QuantumDBService::new(db, ServiceConfig::default());
        let mock_service = MockQuantumDbService {
            service: Arc::new(service),
        };
        
        // Test stats
        let response = mock_service.stats(Request::new(())).await.unwrap();
        assert_eq!(response.into_inner().num_vectors, 0);
    }
}