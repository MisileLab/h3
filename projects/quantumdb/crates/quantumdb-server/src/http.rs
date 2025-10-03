//! HTTP server implementation for QuantumDB
//! 
//! REST API server using Axum framework for HTTP-based access.

use std::net::SocketAddr;
use std::sync::Arc;
use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::Json,
    routing::{get, post},
    Router,
};
use serde::Deserialize;
use tower::ServiceBuilder;
use tower_http::cors::CorsLayer;
use crate::service::{QuantumDBService, SearchRequest, AddRequest, ServiceError};

/// HTTP server for QuantumDB
pub struct QuantumDBHttpServer {
    service: Arc<QuantumDBService>,
    addr: SocketAddr,
}

impl QuantumDBHttpServer {
    /// Create a new HTTP server
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
    
    /// Start the HTTP server
    /// 
    /// # Returns
    /// Future that resolves when server stops
    pub async fn serve(self) -> Result<(), Box<dyn std::error::Error>> {
        let app = Router::new()
            .route("/search", post(search_handler))
            .route("/add", post(add_handler))
            .route("/stats", get(stats_handler))
            .route("/build_index", post(build_index_handler))
            .route("/health", get(health_handler))
            .layer(
                ServiceBuilder::new()
                    .layer(CorsLayer::permissive())
            )
            .with_state(self.service);
        
        println!("Starting HTTP server on {}", self.addr);
        
        let listener = tokio::net::TcpListener::bind(self.addr).await?;
        axum::serve(listener, app).await?;
        
        Ok(())
    }
}

/// Query parameters for search
#[derive(Debug, Deserialize)]
pub struct SearchQuery {
    /// Number of results to return
    pub k: Option<usize>,
    /// Search depth
    pub ef: Option<usize>,
}

/// Search handler
async fn search_handler(
    State(service): State<Arc<QuantumDBService>>,
    Query(query): Query<SearchQuery>,
    Json(request): Json<SearchRequest>,
) -> Result<Json<crate::service::SearchResponse>, ApiError> {
    let mut search_request = request;
    search_request.top_k = search_request.top_k.or(query.k);
    search_request.ef_search = search_request.ef_search.or(query.ef);
    
    let response = service.search(search_request).await
        .map_err(|e| ApiError::from(e))?;
    
    Ok(Json(response))
}

/// Add vectors handler
async fn add_handler(
    State(service): State<Arc<QuantumDBService>>,
    Json(request): Json<AddRequest>,
) -> Result<Json<crate::service::AddResponse>, ApiError> {
    let response = service.add(request).await
        .map_err(|e| ApiError::from(e))?;
    
    Ok(Json(response))
}

/// Stats handler
async fn stats_handler(
    State(service): State<Arc<QuantumDBService>>,
) -> Result<Json<crate::service::DatabaseStats>, ApiError> {
    let stats = service.stats().await
        .map_err(|e| ApiError::from(e))?;
    
    Ok(Json(stats))
}

/// Build index handler
async fn build_index_handler(
    State(service): State<Arc<QuantumDBService>>,
) -> Result<Json<BuildIndexResponse>, ApiError> {
    service.build_index().await
        .map_err(|e| ApiError::from(e))?;
    
    Ok(Json(BuildIndexResponse {
        success: true,
        message: "Index built successfully".to_string(),
    }))
}

/// Health check handler
async fn health_handler() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "healthy".to_string(),
        timestamp: chrono::Utc::now().to_rfc3339(),
    })
}

/// Build index response
#[derive(Debug, serde::Serialize)]
pub struct BuildIndexResponse {
    pub success: bool,
    pub message: String,
}

/// Health check response
#[derive(Debug, serde::Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub timestamp: String,
}

/// API error response
#[derive(Debug, serde::Serialize)]
pub struct ApiErrorResponse {
    pub error: String,
    pub message: String,
}

/// API error type
#[derive(Debug)]
pub enum ApiError {
    BadRequest(String),
    InternalError(String),
}

impl From<ServiceError> for ApiError {
    fn from(err: ServiceError) -> Self {
        match err {
            ServiceError::InvalidRequest(msg) => ApiError::BadRequest(msg),
            ServiceError::Database(msg) => ApiError::InternalError(format!("Database error: {}", msg)),
            ServiceError::Internal(msg) => ApiError::InternalError(msg),
        }
    }
}

impl axum::response::IntoResponse for ApiError {
    fn into_response(self) -> axum::response::Response {
        let (status, error, message) = match self {
            ApiError::BadRequest(msg) => (StatusCode::BAD_REQUEST, "bad_request", msg),
            ApiError::InternalError(msg) => (StatusCode::INTERNAL_SERVER_ERROR, "internal_error", msg),
        };
        
        let body = Json(ApiErrorResponse {
            error: error.to_string(),
            message,
        });
        
        (status, body).into_response()
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
    async fn test_http_server_creation() {
        let device = burn::backend::NdArray::Device::default();
        let compressor_config = QuantumCompressorConfig::new(768, 256, 16);
        let compressor = compressor_config.init(&device);
        let search_config = SearchConfig::default();
        
        let db = quantumdb_core::QuantumDB::new(compressor, search_config, device);
        let service = QuantumDBService::new(db, ServiceConfig::default());
        
        let addr = "127.0.0.1:0".parse().unwrap();
        let server = QuantumDBHttpServer::new(service, addr);
        
        // Server creation should succeed
        assert_eq!(server.addr, addr);
    }

    #[tokio::test]
    async fn test_health_handler() {
        let response = health_handler().await;
        assert_eq!(response.status, "healthy");
        assert!(!response.timestamp.is_empty());
    }

    #[tokio::test]
    async fn test_api_error_conversion() {
        let service_error = ServiceError::InvalidRequest("test error".to_string());
        let api_error: ApiError = service_error.into();
        
        match api_error {
            ApiError::BadRequest(msg) => assert_eq!(msg, "test error"),
            _ => panic!("Expected BadRequest"),
        }
    }
}